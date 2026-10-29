"""Options agent ("options" book): buys calls on strong bullish setups and
puts on strong bearish ones, using the same combined news + chart score.

Risk is capped per trade: the most it can lose is the premium paid, sized to
~1.5% of the account. Contracts are 21-45 days out, near the money (delta
~0.55), with tight bid/ask spreads. Exits: +50% (take profit), -50% (stop),
signal flips against the trade, or 7 days before expiry (never held into
expiration, where time decay is fastest and exercise rules kick in).

Limit orders throughout: option spreads are wide enough that market orders
can give away a lot."""
from __future__ import annotations

import datetime as dt
import math
import os
import sys

import books
from common import Alpaca, load_config, now_utc, read_json, write_json
from trader_agent import hours_since

BOOK = "options"


def tick_round(p: float) -> float:
    """Options trade in $0.05 steps above $3 (pennies below for most
    liquid names); $0.05 above $3 is always valid."""
    return round(round(p / 0.05) * 0.05, 2) if p >= 3 else round(p, 2)


def signal_side(a: dict, buy_thr: float, sell_thr: float, o: dict) -> str | None:
    if a["combined"] >= buy_thr and a["agreeing"] >= o["min_agreeing_factors"]:
        return "call"
    if a["combined"] <= sell_thr and a["opposing"] >= o["min_agreeing_factors"]:
        return "put"
    return None


def choose_contract(contracts: list[dict], snaps: dict, kind: str, price: float, today: dt.date,
                    o: dict) -> dict | None:
    """Pure: pick a liquid contract near the target delta and ~30 days out.
    (unit-tested)"""
    best, best_key = None, None
    for c in contracts:
        q = snaps.get(c["symbol"]) or {}
        quote = q.get("latestQuote") or {}
        bid, ask = float(quote.get("bp") or 0), float(quote.get("ap") or 0)
        if bid <= 0 or ask <= 0 or ask < bid:
            continue
        mid = (bid + ask) / 2
        if (ask - bid) / mid > o["max_spread_pct"]:
            continue
        oi = c.get("open_interest")
        if oi is not None and float(oi) < o["min_open_interest"]:
            continue
        dte = (dt.date.fromisoformat(c["expiration_date"]) - today).days
        strike = float(c["strike_price"])
        delta = (q.get("greeks") or {}).get("delta")
        if delta is not None:
            miss = abs(abs(float(delta)) - o["target_delta"])
        else:  # no greeks: aim slightly in the money
            target = price * (0.98 if kind == "call" else 1.02)
            miss = abs(strike - target) / price * 5
        key = (round(miss, 3), abs(dte - 30), (ask - bid) / mid)
        if best_key is None or key < best_key:
            best, best_key = {"symbol": c["symbol"], "strike": strike, "expiry": c["expiration_date"], "dte": dte,
                              "bid": bid, "ask": ask, "mid": mid, "delta": delta,
                              "iv": q.get("impliedVolatility")}, key
    return best


def exit_reason(pos: dict, info: dict, a: dict | None, today: dt.date, o: dict) -> str | None:
    """Pure: why to close an option position, if at all. (unit-tested)"""
    occ = books.parse_occ(pos["symbol"])
    dte = (occ["expiry"] - today).days
    plpc = float(pos["unrealized_plpc"])
    if dte <= o["exit_dte"]:
        return f"{dte} days to expiry (time exit)"
    if plpc >= o["take_profit_pct"]:
        return f"take profit ({plpc:+.0%})"
    if plpc <= -o["stop_loss_pct"]:
        return f"stop loss ({plpc:+.0%})"
    if a is not None:
        if occ["type"] == "call" and a["combined"] < o["flip_exit"] * -1:
            return f"signals turned against the call ({a['combined']:+.2f})"
        if occ["type"] == "put" and a["combined"] > o["flip_exit"]:
            return f"signals turned against the put ({a['combined']:+.2f})"
    return None


def main() -> int:
    cfg = load_config()
    if os.environ.get("KILL_SWITCH") == "1":
        cfg["kill_switch"] = True
    o = cfg["books"][BOOK]
    now = now_utc()
    today = now.date()
    state = read_json("options_state.json", {})
    trades = read_json("trades.json", [])
    halts, alerts, results = [], [], []
    if not o.get("enabled", True):
        write_json("options_status.json", {"time": now.isoformat(), "halts": ["Options book is turned off"]})
        return 0

    alpaca = Alpaca()
    acct, clock = alpaca.account(), alpaca.clock()
    equity, last_equity = float(acct["equity"]), float(acct["last_equity"])
    state.setdefault("started_at", now.isoformat())
    state.setdefault("capital", round(equity * o["allocation"], 2))
    entries = state.setdefault("entries", {})
    state.setdefault("exits", {})
    analysis, signals = read_json("analysis.json", {}), read_json("signals.json", {})
    th = analysis.get("thresholds", {})
    buy_thr, sell_thr = th.get("buy", 0.4), th.get("sell", -0.3)
    syms = analysis.get("symbols", {})
    owners = books.load_owners()

    def log(rec):
        rec.update(book=BOOK, time=now.isoformat())
        trades.append(rec)
        results.append(rec)
        if rec.get("order_id"):
            state.setdefault("logged_orders", []).append(rec["order_id"])
        print(f"[options] {rec['side'].upper()} {rec['symbol']}: {rec['status']} - {rec['reason']}")

    try:
        results += books.sync_fills(alpaca, BOOK, state, trades, now, multiplier=100, include_buys=True,
                                    describe=lambda od: f"{od['side']} order filled")
    except Exception as e:
        alerts.append(f"Could not check recent fills: {e}")

    level = int(acct.get("options_trading_level") or 0)
    if level < 2:
        halts.append(f"Account options level is {level}; buying calls/puts needs level 2")

    positions = alpaca.positions()
    mine = books.book_positions(BOOK, positions, owners)
    held = {p["symbol"] for p in mine}
    for p in mine:
        e = entries.setdefault(p["symbol"], {"time": now.isoformat()})
        e["avg"] = float(p["avg_entry_price"])
    for sym in list(entries):
        if sym not in held:
            info = entries.pop(sym)
            state["exits"].setdefault(info.get("underlying") or books.parse_occ(sym)["underlying"], now.isoformat())

    orders = alpaca.open_orders()
    my_orders = [x for x in orders if books.book_of_order(x) == BOOK]
    if cfg.get("kill_switch"):
        halts.append("Kill switch is on")
    if not clock["is_open"]:
        halts.append("Market closed")
    else:
        # Unfilled limit orders from earlier runs: cancel and re-price below.
        for x in my_orders:
            try:
                alpaca.cancel_order(x["id"])
            except Exception:
                pass
        snaps = {}
        if mine:
            try:
                snaps = alpaca.option_snapshots(sorted(held))
            except Exception as e:
                alerts.append(f"Could not get option quotes: {str(e)[:120]}")

        # ---- exits --------------------------------------------------------
        for p in mine:
            occ = books.parse_occ(p["symbol"])
            reason = exit_reason(p, entries.get(p["symbol"], {}), syms.get(occ["underlying"]), today, o)
            if cfg.get("kill_switch") and cfg.get("liquidate_on_kill"):
                reason = "kill switch liquidation"
            if not reason:
                continue
            qty = int(float(p["qty"]))
            quote = (snaps.get(p["symbol"]) or {}).get("latestQuote") or {}
            bid = float(quote.get("bp") or 0)
            rec = {"symbol": p["symbol"], "side": "sell", "reason": reason, "qty": qty,
                   "underlying": occ["underlying"], "notional": round(float(p["market_value"]), 2)}
            try:
                dte = (occ["expiry"] - today).days
                if bid > 0 and dte > 2:
                    od = alpaca.limit_order(p["symbol"], "sell", qty, tick_round(bid), books.cid(BOOK))
                else:   # no bid, or about to expire: get out at market
                    od = alpaca.sell_shares(p["symbol"], qty, books.cid(BOOK))
                rec.update(status="order placed", order_id=od["id"], limit=tick_round(bid) if bid else None)
                state.setdefault("order_reasons", {})[od["id"]] = reason
                state["exits"][occ["underlying"]] = now.isoformat()
                print(f"[options] SELL {p['symbol']} x{qty}: {reason}")
                results.append({**rec, "book": BOOK, "time": now.isoformat()})
            except Exception as e:
                rec.update(status="error", error=str(e)[:300])
                log(rec)

        # ---- entries ------------------------------------------------------
        risk = cfg["risk"]
        if (equity - last_equity) / last_equity <= -risk["daily_loss_limit_pct"]:
            halts.append("Daily loss limit hit")
        if signals.get("market_risk", 0) > risk["max_market_risk"]:
            halts.append("High market risk")
        if not analysis.get("generated_at") or hours_since(analysis["generated_at"], now) > risk["max_signal_age_hours"]:
            halts.append("Chart analysis is stale")
        if signals.get("scorer") == "keywords" and not risk.get("allow_keyword_trades"):
            halts.append("Claude could not read the news")
        held_under = {books.parse_occ(s)["underlying"] for s in held}
        slots = o["max_positions"] - len(held)
        if not halts and slots > 0:
            orders = alpaca.open_orders()
            avail = books.budget(BOOK, cfg, equity, alpaca.positions(), owners, orders)
            news = signals.get("tickers", {})
            ideas = []
            for u, a in syms.items():
                side = signal_side(a, buy_thr, sell_thr, o)
                if not side or u in held_under or hours_since(state["exits"].get(u), now) < o["reentry_cooldown_hours"]:
                    continue
                sig = news.get(u, {})
                if side == "call" and sig.get("ai_articles", 0) >= risk["min_articles"] and sig.get("score", 0) <= risk["news_sell_threshold"]:
                    continue
                ideas.append((abs(a["combined"]), u, side, a))
            for _, u, side, a in sorted(ideas, reverse=True)[:slots]:
                px = a["price"]
                try:
                    lo, hi = (px * 0.85, px * 1.05) if side == "call" else (px * 0.95, px * 1.15)
                    cs = alpaca.option_contracts(u, side, today + dt.timedelta(days=o["dte_min"]),
                                                 today + dt.timedelta(days=o["dte_max"]), lo, hi)
                    cs.sort(key=lambda c: (abs((dt.date.fromisoformat(c["expiration_date"]) - today).days - 30),
                                           abs(float(c["strike_price"]) - px)))
                    pick = choose_contract(cs, alpaca.option_snapshots([c["symbol"] for c in cs[:40]]),
                                           side, px, today, o)
                except Exception as e:
                    alerts.append(f"{u}: could not look up options ({str(e)[:100]})")
                    continue
                if not pick:
                    print(f"[options] {u}: no liquid {side} contract found")
                    continue
                limit = tick_round(pick["mid"])
                per_contract = limit * 100
                qty = math.floor(min(equity * o["risk_per_trade_pct"], avail) / per_contract)
                if qty < 1:
                    print(f"[options] {u}: one {side} costs ${per_contract:,.0f}, above the per-trade risk cap")
                    continue
                d = f"{pick['delta']:+.2f}" if pick["delta"] is not None else "n/a"
                rec = {"symbol": pick["symbol"], "side": "buy", "qty": qty, "limit": limit, "underlying": u,
                       "notional": round(qty * per_contract, 2), "score": a["combined"], "factors": a["factors"],
                       "reason": f"{side.upper()} {u} ${pick['strike']:g} exp {pick['expiry']} ({pick['dte']}d, "
                                 f"delta {d}) on score {a['combined']:+.2f}; max loss ${qty * per_contract:,.0f}"}
                try:
                    od = alpaca.limit_order(pick["symbol"], "buy", qty, limit, books.cid(BOOK))
                    entries[pick["symbol"]] = {"time": now.isoformat(), "underlying": u, "type": side,
                                               "score": a["combined"], "delta": pick["delta"], "iv": pick["iv"],
                                               "dte": pick["dte"], "limit": limit}
                    state.setdefault("order_reasons", {})[od["id"]] = rec["reason"]
                    rec.update(status="order placed", order_id=od["id"], book=BOOK, time=now.isoformat())
                    results.append(rec)
                    print(f"[options] BUY {pick['symbol']} x{qty} @ {limit:.2f}: {rec['reason']}")
                    avail -= qty * per_contract
                except Exception as e:
                    rec.update(status="error", error=str(e)[:300])
                    log(rec)

    # What the options book has learned from its own closed trades.
    closed = [t for t in trades if t.get("book") == BOOK and t.get("realized_pl") is not None]
    by_type = {}
    for t in closed:
        k = "call" if books.is_option(t["symbol"]) and books.parse_occ(t["symbol"])["type"] == "call" else "put"
        by_type.setdefault(k, []).append(t["realized_pl"])
    learned = {"closed_trades": len(closed), "min_for_conclusions": 20,
               "by_type": {k: {"n": len(v), "win_rate": round(sum(x > 0 for x in v) / len(v), 3),
                               "total": round(sum(v), 2)} for k, v in by_type.items()}}

    state["logged_orders"] = state.get("logged_orders", [])[-2000:]
    write_json("trades.json", trades[-3000:])
    write_json("options_state.json", state)
    write_json("options_status.json", {"time": now.isoformat(), "halts": halts, "alerts": alerts,
                                       "last_actions": results, "options_level": level, "learned": learned})
    for m in halts + alerts:
        print("[options] Note:", m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
