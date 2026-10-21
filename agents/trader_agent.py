"""Trading agent: reads the news agent's signals, applies risk rules, places
orders on Alpaca (paper by default), and records P&L for the dashboard."""
from __future__ import annotations

import os
import sys

from common import Alpaca, load_config, now_utc, parse_time, read_json, write_json


def decide(positions: list[dict], signals: dict, equity: float, cash: float, day_pnl_pct: float,
           cfg: dict, pending: set[str], now) -> tuple[list[dict], list[str]]:
    """Pure decision function (unit-tested). Returns (actions, halt_reasons)."""
    risk = cfg["risk"]
    watch = {s.upper() for s in cfg["watchlist"]}
    tickers = signals.get("tickers", {})
    actions, halts = [], []

    held = [p for p in positions if p["symbol"] in watch]

    if cfg.get("kill_switch"):
        halts.append("Kill switch is on")
        if cfg.get("liquidate_on_kill"):
            actions += [{"action": "sell", "symbol": p["symbol"], "reason": "kill switch liquidation"} for p in held]
        return actions, halts

    # ---- sells: protect capital first -----------------------------------
    selling = set()
    for p in held:
        sym, plpc = p["symbol"], float(p["unrealized_plpc"])
        sig = tickers.get(sym, {})
        reason = None
        if plpc <= -risk["stop_loss_pct"]:
            reason = f"stop loss ({plpc:+.1%})"
        elif plpc >= risk["take_profit_pct"]:
            reason = f"take profit ({plpc:+.1%})"
        elif sig.get("articles", 0) >= risk["min_articles"] and sig.get("score", 0) <= risk["sell_threshold"]:
            reason = f"negative news ({sig['score']:+.2f})"
        if reason and sym not in pending:
            actions.append({"action": "sell", "symbol": sym, "reason": reason})
            selling.add(sym)

    # ---- buy gates ------------------------------------------------------
    if day_pnl_pct <= -risk["daily_loss_limit_pct"]:
        halts.append(f"Daily loss limit hit ({day_pnl_pct:+.2%})")
    if signals.get("market_risk", 0) > risk["max_market_risk"]:
        halts.append(f"News agent flags high market risk ({signals['market_risk']:.2f})")
    gen = signals.get("generated_at")
    if not gen:
        halts.append("No news signals yet")
    elif (now - parse_time(gen)).total_seconds() / 3600 > risk["max_signal_age_hours"]:
        halts.append("News signals are stale")
    if halts:
        return actions, halts

    # ---- buys -----------------------------------------------------------
    held_syms = {p["symbol"] for p in held}
    slots = risk["max_positions"] - (len(held_syms) - len(selling))
    budget = cash - equity * risk["min_cash_pct"]
    candidates = sorted(
        ((s, v) for s, v in tickers.items()
         if s in watch and s not in held_syms and s not in pending
         and v.get("articles", 0) >= risk["min_articles"] and v.get("score", 0) >= risk["buy_threshold"]),
        key=lambda kv: kv[1]["score"], reverse=True)
    for sym, v in candidates:
        if slots <= 0:
            break
        notional = min(equity * risk["max_position_pct"], budget)
        if notional < 1:
            break
        actions.append({"action": "buy", "symbol": sym, "notional": round(notional, 2),
                        "reason": f"positive news ({v['score']:+.2f}, {v['articles']} articles)"})
        budget -= notional
        slots -= 1
    return actions, halts


def main() -> int:
    cfg = load_config()
    if os.environ.get("KILL_SWITCH") == "1":
        cfg["kill_switch"] = True
    now = now_utc()
    alpaca = Alpaca()
    acct = alpaca.account()
    clock = alpaca.clock()
    positions = alpaca.positions()

    equity, cash = float(acct["equity"]), float(acct["cash"])
    last_equity = float(acct["last_equity"])
    day_pnl = equity - last_equity
    day_pnl_pct = day_pnl / last_equity if last_equity else 0.0

    state = read_json("state.json", {})
    state.setdefault("starting_equity", equity)
    state.setdefault("started_at", now.isoformat())
    signals = read_json("signals.json", {})

    actions, halts, results = [], [], []
    if not clock["is_open"]:
        halts.append("Market closed")
    else:
        pending = {o["symbol"] for o in alpaca.open_orders()}
        actions, halts = decide(positions, signals, equity, cash, day_pnl_pct, cfg, pending, now)
        trades = read_json("trades.json", [])
        for a in actions:
            sym = a["symbol"]
            score = signals.get("tickers", {}).get(sym, {}).get("score")
            rec = {"time": now.isoformat(), "symbol": sym, "side": a["action"], "reason": a["reason"],
                   "news_score": score, "notional": a.get("notional")}
            try:
                order = alpaca.buy_notional(sym, a["notional"]) if a["action"] == "buy" else alpaca.close_position(sym)
                rec.update(status="submitted", order_id=order.get("id"))
                if a["action"] == "sell":
                    p = next((p for p in positions if p["symbol"] == sym), None)
                    if p:
                        rec["realized_pl"] = round(float(p["unrealized_pl"]), 2)
                        rec["notional"] = round(float(p["market_value"]), 2)
            except Exception as e:
                rec.update(status="error", error=str(e)[:300])
            trades.append(rec)
            results.append(rec)
            print(f"{rec['side'].upper()} {sym}: {rec['status']} - {a['reason']}")
        write_json("trades.json", trades[-1000:])
        positions = alpaca.positions()  # refresh after orders
        acct = alpaca.account()
        equity, cash = float(acct["equity"]), float(acct["cash"])

    snap = {"time": now.isoformat(), "equity": round(equity, 2), "cash": round(cash, 2)}
    equity_hist = read_json("equity.json", [])
    equity_hist.append(snap)
    write_json("equity.json", equity_hist[-5000:])
    write_json("state.json", state)

    status = {
        "last_run": now.isoformat(),
        "paper": alpaca.is_paper,
        "market_open": clock["is_open"],
        "next_open": clock.get("next_open"),
        "equity": round(equity, 2),
        "cash": round(cash, 2),
        "day_pnl": round(equity - last_equity, 2),
        "day_pnl_pct": round((equity - last_equity) / last_equity, 5) if last_equity else 0,
        "starting_equity": state["starting_equity"],
        "total_pnl": round(equity - state["starting_equity"], 2),
        "total_pnl_pct": round((equity - state["starting_equity"]) / state["starting_equity"], 5),
        "halts": halts,
        "last_actions": results,
        "market_risk": signals.get("market_risk"),
        "signals_at": signals.get("generated_at"),
        "positions": [{"symbol": p["symbol"], "qty": float(p["qty"]),
                       "avg_entry": float(p["avg_entry_price"]), "price": float(p["current_price"]),
                       "value": float(p["market_value"]), "pl": float(p["unrealized_pl"]),
                       "pl_pct": float(p["unrealized_plpc"])} for p in positions],
    }
    write_json("status.json", status)
    print(f"Equity ${equity:,.2f} | today {status['day_pnl']:+,.2f} | total {status['total_pnl']:+,.2f}")
    if halts:
        print("Buys paused:", "; ".join(halts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
