"""Trading agent: combines the news agent's and chart agent's views, applies
risk rules, places orders on Alpaca (paper by default), keeps a trailing stop
on every position at the broker, and records P&L for the dashboard.

Swing style: positions are meant to last days to ~2 weeks."""
from __future__ import annotations

import datetime as dt
import math
import os
import sys
import time

from common import Alpaca, load_config, now_utc, parse_time, read_json, write_json


def thresholds(cfg: dict, analysis: dict) -> tuple[float, float]:
    risk, learned = cfg["risk"], analysis.get("thresholds", {})
    buy = risk.get("buy_threshold", "auto")
    sell = risk.get("exit_threshold", "auto")
    buy = learned.get("buy", 0.25) if buy == "auto" else float(buy)
    sell = learned.get("sell", -0.25) if sell == "auto" else float(sell)
    return buy, sell


def hours_since(ts: str | None, now: dt.datetime) -> float:
    return (now - parse_time(ts)).total_seconds() / 3600 if ts else float("inf")


def decide(positions: list[dict], analysis: dict, signals: dict, equity: float, cash: float,
           day_pnl_pct: float, cfg: dict, pending: set[str], now: dt.datetime,
           state: dict) -> tuple[list[dict], list[str]]:
    """Pure decision function (unit-tested). Returns (actions, halt_reasons)."""
    risk = cfg["risk"]
    watch = {s.upper() for s in cfg["watchlist"]}
    syms = analysis.get("symbols", {})
    news = signals.get("tickers", {})
    entries, exits = state.get("entries", {}), state.get("exits", {})
    buy_thr, exit_thr = thresholds(cfg, analysis)
    news_sell = risk.get("news_sell_threshold", risk.get("sell_threshold", -0.30))
    key = "articles" if risk.get("allow_keyword_trades") else "ai_articles"
    actions, halts = [], []
    held = [p for p in positions if p["symbol"] in watch]

    if cfg.get("kill_switch"):
        halts.append("Kill switch is on")
        if cfg.get("liquidate_on_kill"):
            actions += [{"action": "sell", "symbol": p["symbol"], "reason": "kill switch liquidation"} for p in held]
        return actions, halts

    # ---- sells (the 4% trailing stop itself lives at the broker) ----------
    selling = set()
    for p in held:
        sym, plpc = p["symbol"], float(p["unrealized_plpc"])
        a, sig = syms.get(sym), news.get(sym, {})
        held_days = hours_since(entries.get(sym, {}).get("time"), now) / 24
        reason = None
        if plpc <= -risk["stop_loss_pct"]:
            reason = f"stop loss ({plpc:+.1%})"
        elif risk.get("take_profit_pct") and plpc >= risk["take_profit_pct"]:
            reason = f"take profit ({plpc:+.1%})"
        elif sig.get(key, 0) >= risk["min_articles"] and sig.get("score", 0) <= news_sell:
            reason = f"negative news ({sig['score']:+.2f})"
        elif a and a["combined"] <= exit_thr:
            reason = f"signals turned negative ({a['combined']:+.2f})"
        elif held_days != float("inf") and held_days >= risk["max_hold_days"] and not (a and a["combined"] >= buy_thr):
            reason = f"max hold reached ({held_days:.0f} days, no fresh buy signal)"
        if reason and sym not in pending:
            actions.append({"action": "sell", "symbol": sym, "reason": reason})
            selling.add(sym)

    # ---- buy gates ------------------------------------------------------
    if day_pnl_pct <= -risk["daily_loss_limit_pct"]:
        halts.append(f"Daily loss limit hit ({day_pnl_pct:+.2%})")
    if signals.get("market_risk", 0) > risk["max_market_risk"]:
        halts.append(f"News agent flags high market risk ({signals['market_risk']:.2f})")
    if not analysis.get("generated_at"):
        halts.append("No chart analysis yet")
    elif hours_since(analysis["generated_at"], now) > risk["max_signal_age_hours"]:
        halts.append("Chart analysis is stale")
    if not signals.get("generated_at") or hours_since(signals["generated_at"], now) > risk["max_signal_age_hours"]:
        halts.append("News signals are stale")
    if not risk.get("allow_keyword_trades") and signals.get("scorer") == "keywords":
        halts.append("Claude could not read the news (check Anthropic API credits)")
    if halts:
        return actions, halts

    # ---- buys -----------------------------------------------------------
    held_syms = {p["symbol"] for p in held}
    slots = risk["max_positions"] - (len(held_syms) - len(selling))
    budget = cash - equity * risk["min_cash_pct"]
    candidates = []
    for sym, a in syms.items():
        sig = news.get(sym, {})
        if (sym not in watch or sym in held_syms or sym in pending or a["combined"] < buy_thr
                or a["agreeing"] < risk.get("min_agreeing_factors", 3)):
            continue
        if sig.get(key, 0) >= risk["min_articles"] and sig.get("score", 0) <= news_sell:
            continue  # never buy into clearly bad news
        if hours_since(exits.get(sym), now) < risk.get("reentry_cooldown_hours", 20):
            continue  # just sold it; wait before buying back
        candidates.append((sym, a))
    candidates.sort(key=lambda kv: kv[1]["combined"], reverse=True)
    for sym, a in candidates:
        if slots <= 0:
            break
        notional = min(equity * risk["max_position_pct"], budget)
        qty = math.floor(notional / a["price"]) if a["price"] > 0 else 0
        if qty < 1:
            continue
        f = a["factors"]
        why = ", ".join(f"{k} {v:+.2f}" for k, v in sorted(f.items(), key=lambda kv: -kv[1]) if v > 0.1)
        actions.append({"action": "buy", "symbol": sym, "qty": qty, "notional": round(qty * a["price"], 2),
                        "reason": f"score {a['combined']:+.2f} ({a['agreeing']}/{len(f)} agree: {why})"})
        budget -= qty * a["price"]
        slots -= 1
    return actions, halts


def protection_needed(positions: list[dict], open_orders: list[dict], watch: set[str]) -> list[tuple[str, int]]:
    """Positions with whole shares not yet covered by a trailing stop."""
    out = []
    for p in positions:
        sym, whole = p["symbol"], int(float(p["qty"]))
        if sym not in watch or whole < 1:
            continue
        if any(o["symbol"] == sym and o["side"] == "sell" for o in open_orders):
            continue
        out.append((sym, whole))
    return out


def sync_fills(alpaca: Alpaca, state: dict, trades: list[dict], now: dt.datetime) -> list[dict]:
    """Log sells that happened at the broker between runs (trailing stops)."""
    after = state.get("last_order_check") or (now - dt.timedelta(days=3)).isoformat()
    logged = set(state.get("logged_orders", []))
    new = []
    for o in alpaca.orders(status="closed", after=after, limit=500, direction="asc"):
        if o.get("status") != "filled" or o.get("side") != "sell" or o["id"] in logged:
            continue
        sym, qty, px = o["symbol"], float(o["filled_qty"]), float(o["filled_avg_price"])
        entry = state.get("entries", {}).get(sym, {}).get("avg")
        rec = {"time": o.get("filled_at") or now.isoformat(), "symbol": sym, "side": "sell", "status": "filled",
               "order_id": o["id"], "notional": round(qty * px, 2),
               "reason": ("trailing stop hit (fell from its high)" if o.get("type") == "trailing_stop"
                          else "sell order filled")}
        if entry:
            rec["realized_pl"] = round((px - entry) * qty, 2)
        trades.append(rec)
        new.append(rec)
        logged.add(o["id"])
        state.setdefault("exits", {})[sym] = rec["time"]
        print(f"FILLED SELL {sym} at {px:.2f}: {rec['reason']}")
    state["logged_orders"] = list(logged)[-1000:]
    state["last_order_check"] = now.isoformat()
    return new


def main() -> int:
    cfg = load_config()
    if os.environ.get("KILL_SWITCH") == "1":
        cfg["kill_switch"] = True
    risk = cfg["risk"]
    watch = {s.upper() for s in cfg["watchlist"]}
    now = now_utc()
    alpaca = Alpaca()
    acct, clock = alpaca.account(), alpaca.clock()
    positions = alpaca.positions()
    equity, cash = float(acct["equity"]), float(acct["cash"])
    last_equity = float(acct["last_equity"])

    state = read_json("state.json", {})
    state.setdefault("starting_equity", equity)
    state.setdefault("started_at", now.isoformat())
    entries = state.setdefault("entries", {})
    state.setdefault("exits", {})
    signals = read_json("signals.json", {})
    analysis = read_json("analysis.json", {})
    trades = read_json("trades.json", [])
    logged = set(state.get("logged_orders", []))
    results, halts, alerts = [], [], []

    def log(rec: dict):
        trades.append(rec)
        results.append(rec)
        if rec.get("order_id"):
            logged.add(rec["order_id"])
        print(f"{rec['side'].upper()} {rec['symbol']}: {rec['status']} - {rec['reason']}")

    try:
        results += sync_fills(alpaca, state, trades, now)
        logged |= set(state.get("logged_orders", []))
    except Exception as e:
        alerts.append(f"Could not check recent fills: {e}")

    # Track when each position was opened (for the max-hold rule).
    buys = {t["symbol"]: t["time"] for t in trades if t["side"] == "buy" and t.get("status") != "error"}
    for p in positions:
        e = entries.setdefault(p["symbol"], {"time": buys.get(p["symbol"], now.isoformat())})
        e["avg"] = float(p["avg_entry_price"])
    for sym in list(entries):
        if sym not in {p["symbol"] for p in positions}:
            entries.pop(sym)
            state["exits"].setdefault(sym, now.isoformat())

    open_orders = alpaca.open_orders()
    if not clock["is_open"]:
        halts.append("Market closed")
    else:
        # Sell fractional leftovers so every share can sit under a trailing stop.
        for p in positions:
            frac = float(p["qty"]) - int(float(p["qty"]))
            if p["symbol"] in watch and frac > 1e-6 and not any(
                    o["symbol"] == p["symbol"] and o["side"] == "sell" and o["type"] != "trailing_stop"
                    for o in open_orders):
                rec = {"time": now.isoformat(), "symbol": p["symbol"], "side": "sell",
                       "reason": f"tidy {frac:.4f} fractional share", "notional": round(frac * float(p["current_price"]), 2)}
                try:
                    rec.update(status="submitted", order_id=alpaca.sell_shares(p["symbol"], frac)["id"])
                except Exception as e:
                    rec.update(status="error", error=str(e)[:300])
                log(rec)

        pending = {o["symbol"] for o in open_orders if o["type"] != "trailing_stop"}
        actions, halts = decide(positions, analysis, signals, equity, cash, (equity - last_equity) / last_equity
                                if last_equity else 0.0, cfg, pending, now, state)
        for a in actions:
            sym = a["symbol"]
            info = analysis.get("symbols", {}).get(sym, {})
            rec = {"time": now.isoformat(), "symbol": sym, "side": a["action"], "reason": a["reason"],
                   "score": info.get("combined"), "factors": info.get("factors"),
                   "news_score": info.get("news_score"), "notional": a.get("notional")}
            try:
                if a["action"] == "sell":
                    for o in open_orders:  # release shares held by the trailing stop first
                        if o["symbol"] == sym:
                            alpaca.cancel_order(o["id"])
                    p = next((p for p in positions if p["symbol"] == sym), None)
                    order = alpaca.close_position(sym)
                    if p:
                        rec["realized_pl"] = round(float(p["unrealized_pl"]), 2)
                        rec["notional"] = round(float(p["market_value"]), 2)
                    state["exits"][sym] = now.isoformat()
                else:
                    order = alpaca.buy_shares(sym, a["qty"])
                    entries[sym] = {"time": now.isoformat(), "avg": info.get("price")}
                    for _ in range(10):  # wait briefly for the fill, then protect it
                        o = alpaca.get_order(order["id"])
                        if o["status"] == "filled":
                            alpaca.trailing_stop(sym, int(float(o["filled_qty"])), risk["trailing_stop_pct"] * 100)
                            rec["trailing_stop"] = f"{risk['trailing_stop_pct']:.0%}"
                            break
                        time.sleep(1)
                rec.update(status="submitted", order_id=order.get("id"))
            except Exception as e:
                rec.update(status="error", error=str(e)[:300])
            log(rec)
        if actions:
            time.sleep(2)
            positions = alpaca.positions()
            open_orders = alpaca.open_orders()

    # Every whole-share position gets a trailing stop at the broker. GTC
    # orders are accepted while the market is closed and activate at the open.
    for sym, qty in protection_needed(positions, open_orders, watch):
        try:
            alpaca.trailing_stop(sym, qty, risk["trailing_stop_pct"] * 100)
            print(f"PROTECT {sym}: trailing stop {risk['trailing_stop_pct']:.0%} on {qty} shares")
        except Exception as e:
            alerts.append(f"Could not place trailing stop on {sym}: {str(e)[:120]}")
    open_orders = alpaca.open_orders()
    acct = alpaca.account()
    equity, cash = float(acct["equity"]), float(acct["cash"])

    state["logged_orders"] = list(logged)[-1000:]
    write_json("trades.json", trades[-1000:])
    equity_hist = read_json("equity.json", [])
    equity_hist.append({"time": now.isoformat(), "equity": round(equity, 2), "cash": round(cash, 2)})
    write_json("equity.json", equity_hist[-5000:])
    write_json("state.json", state)

    stops = {o["symbol"]: o for o in open_orders if o["type"] == "trailing_stop"}
    buy_thr, exit_thr = thresholds(cfg, analysis)
    status = {
        "last_run": now.isoformat(), "paper": alpaca.is_paper, "market_open": clock["is_open"],
        "next_open": clock.get("next_open"), "equity": round(equity, 2), "cash": round(cash, 2),
        "day_pnl": round(equity - last_equity, 2),
        "day_pnl_pct": round((equity - last_equity) / last_equity, 5) if last_equity else 0,
        "starting_equity": state["starting_equity"],
        "total_pnl": round(equity - state["starting_equity"], 2),
        "total_pnl_pct": round((equity - state["starting_equity"]) / state["starting_equity"], 5),
        "halts": halts, "alerts": alerts, "last_actions": results,
        "market_risk": signals.get("market_risk"), "signals_at": signals.get("generated_at"),
        "analysis_at": analysis.get("generated_at"),
        "buy_threshold": buy_thr, "exit_threshold": exit_thr, "trailing_stop_pct": risk["trailing_stop_pct"],
        "max_hold_days": risk["max_hold_days"],
        "positions": [{
            "symbol": p["symbol"], "qty": float(p["qty"]), "avg_entry": float(p["avg_entry_price"]),
            "price": float(p["current_price"]), "value": float(p["market_value"]),
            "pl": float(p["unrealized_pl"]), "pl_pct": float(p["unrealized_plpc"]),
            "opened": entries.get(p["symbol"], {}).get("time"),
            "stop_price": float(stops[p["symbol"]]["stop_price"]) if p["symbol"] in stops and stops[p["symbol"]].get("stop_price") else None,
            "high_water": float(stops[p["symbol"]]["hwm"]) if p["symbol"] in stops and stops[p["symbol"]].get("hwm") else None,
        } for p in positions],
    }
    write_json("status.json", status)
    print(f"Equity ${equity:,.2f} | today {status['day_pnl']:+,.2f} | total {status['total_pnl']:+,.2f}")
    for msg in halts + alerts:
        print("Note:", msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
