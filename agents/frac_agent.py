"""Fractional-share agent ("frac" book): a dollar-sized swing strategy that
works for any account size.

  Entry: when a stock is in an uptrend and its combined score clears this
         book's bar, place a LIMIT buy a little below the current price
         (k x ATR, learned from history) so it buys small dips.
  Exit:  a STOP-LIMIT sell kept m x ATR under the highest price since buying
         (also learned), raised every run as the stock climbs. If the price
         gaps through the stop and the stop-limit can't fill, the agent sells
         at MARKET on its next run. Also exits on negative signals, clearly
         bad news, or after max_hold_days.

Fractional orders at Alpaca must be day orders, so protective stops are
re-placed every trading morning."""
from __future__ import annotations

import datetime as dt
import os
import sys
import time

import books
from common import Alpaca, load_config, now_utc, parse_time, read_json, write_json
from trader_agent import hours_since

BOOK = "frac"


def settings(cfg: dict, weights: dict) -> dict:
    f = cfg["books"][BOOK]
    learned = weights.get("frac") or {}
    pick = lambda key, lk, dflt: learned.get(lk, f.get(dflt)) if f.get(key, "auto") == "auto" else f[key]
    return {"limit_atr": float(pick("limit_atr", "limit_atr", "default_limit_atr")),
            "stop_atr": float(pick("stop_atr", "stop_atr", "default_stop_atr")),
            "buy": float(pick("buy_threshold", "buy_threshold", "default_buy_threshold"))}


def plan_entries(analysis: dict, signals: dict, cfg: dict, st: dict, holding: set[str], blocked: set[str],
                 pending_buys: set[str], state: dict, avail: float, book_capital: float,
                 now: dt.datetime) -> list[dict]:
    """Pure: which limit buys to place. (unit-tested)"""
    f, risk = cfg["books"][BOOK], cfg["risk"]
    news = signals.get("tickers", {})
    out, slots = [], f["max_positions"] - len(holding | pending_buys)
    cands = []
    for sym, a in analysis.get("symbols", {}).items():
        sig = news.get(sym, {})
        if sym in holding or sym in blocked or sym in pending_buys:
            continue
        if a["combined"] < st["buy"] or a["factors"].get("trend", 0) <= 0 or a["agreeing"] < f["min_agreeing_factors"]:
            continue
        if sig.get("ai_articles", 0) >= risk["min_articles"] and sig.get("score", 0) <= risk["news_sell_threshold"]:
            continue
        if hours_since(state.get("exits", {}).get(sym), now) < f["reentry_cooldown_hours"]:
            continue
        atr = (a.get("levels") or {}).get("atr")
        if not atr:
            continue
        cands.append((sym, a, atr))
    cands.sort(key=lambda x: -x[1]["combined"])
    for sym, a, atr in cands:
        if slots <= 0:
            break
        dollars = min(book_capital * f["position_pct"], avail)
        if dollars < f.get("min_order_dollars", 25):
            break
        limit = round(a["price"] - st["limit_atr"] * atr, 2)
        qty = round(dollars / limit, 4)
        out.append({"symbol": sym, "limit": limit, "qty": qty, "dollars": round(qty * limit, 2), "atr": atr,
                    "reason": f"limit buy {st['limit_atr']:g} ATR below ${a['price']:.2f} "
                              f"(score {a['combined']:+.2f}, trend {a['factors']['trend']:+.2f})"})
        avail -= dollars
        slots -= 1
    return out


def exit_plan(pos: dict, entry: dict, a: dict | None, sig: dict, cfg: dict, st: dict, exit_thr: float,
              now: dt.datetime) -> tuple[str | None, float]:
    """Pure: returns (sell reason or None, protective stop price). (unit-tested)"""
    f, risk = cfg["books"][BOOK], cfg["risk"]
    price, avg = float(pos["current_price"]), float(pos["avg_entry_price"])
    atr = entry.get("atr") or avg * 0.02
    hwm = max(entry.get("hwm", avg), price)
    stop = round(max(avg - st["stop_atr"] * atr, hwm - st["stop_atr"] * atr), 2)
    held_days = hours_since(entry.get("time"), now) / 24
    if price <= stop:
        return f"price {price:.2f} fell through stop {stop:.2f} (market sell fallback)", stop
    if a and a["combined"] <= exit_thr:
        return f"signals turned negative ({a['combined']:+.2f})", stop
    if sig.get("ai_articles", 0) >= risk["min_articles"] and sig.get("score", 0) <= risk["news_sell_threshold"]:
        return f"negative news ({sig['score']:+.2f})", stop
    if held_days >= f["max_hold_days"] and not (a and a["combined"] >= st["buy"]):
        return f"max hold reached ({held_days:.0f} days)", stop
    return None, stop


def main() -> int:
    cfg = load_config()
    if os.environ.get("KILL_SWITCH") == "1":
        cfg["kill_switch"] = True
    f = cfg["books"][BOOK]
    now = now_utc()
    state = read_json("frac_state.json", {})
    trades = read_json("trades.json", [])
    halts, alerts, results = [], [], []
    if not f.get("enabled", True):
        write_json("frac_status.json", {"time": now.isoformat(), "halts": ["Fractional book is turned off"]})
        return 0

    alpaca = Alpaca()
    acct, clock = alpaca.account(), alpaca.clock()
    equity, last_equity = float(acct["equity"]), float(acct["last_equity"])
    state.setdefault("started_at", now.isoformat())
    state.setdefault("capital", round(equity * f["allocation"], 2))
    entries = state.setdefault("entries", {})
    state.setdefault("exits", {})
    analysis, signals = read_json("analysis.json", {}), read_json("signals.json", {})
    weights = read_json("weights.json", {})
    st = settings(cfg, weights)
    _, exit_thr = (analysis.get("thresholds", {}).get("buy", 0.25), analysis.get("thresholds", {}).get("sell", -0.25))
    owners = books.load_owners()

    def log(rec):
        rec.update(book=BOOK, time=now.isoformat())
        trades.append(rec)
        results.append(rec)
        if rec.get("order_id"):
            state.setdefault("logged_orders", []).append(rec["order_id"])
        print(f"[frac] {rec['side'].upper()} {rec['symbol']}: {rec['status']} - {rec['reason']}")

    try:
        results += books.sync_fills(alpaca, BOOK, state, trades, now, include_buys=True, describe=lambda o: (
            "stop-limit exit hit" if o.get("type") in ("stop_limit", "stop") else f"{o['side']} order filled"))
    except Exception as e:
        alerts.append(f"Could not check recent fills: {e}")

    positions = alpaca.positions()
    orders = alpaca.open_orders()
    mine = books.book_positions(BOOK, positions, owners)
    my_orders = [o for o in orders if books.book_of_order(o) == BOOK]
    held = {p["symbol"] for p in mine}

    # Bookkeeping for positions (entry time, ATR, highest price since buying).
    for p in mine:
        e = entries.setdefault(p["symbol"], {"time": now.isoformat()})
        e.setdefault("atr", (state.get("pending", {}).get(p["symbol"]) or {}).get("atr"))
        e["avg"] = float(p["avg_entry_price"])
        e["hwm"] = max(e.get("hwm", 0), float(p["current_price"]), e["avg"])
    for sym in list(entries):
        if sym not in held:
            entries.pop(sym)
            state["exits"].setdefault(sym, now.isoformat())

    if cfg.get("kill_switch"):
        halts.append("Kill switch is on")
    if not clock["is_open"]:
        halts.append("Market closed")
    else:
        # Daily highs since entry, so the stop also rises on moves that
        # happened between hourly runs.
        if mine:
            try:
                start = min(parse_time(entries[p["symbol"]]["time"]) for p in mine) - dt.timedelta(days=1)
                for sym, bs in alpaca.bars(sorted(held), "1Day", start).items():
                    since = parse_time(entries[sym]["time"]).date()
                    highs = [b["h"] for b in bs if parse_time(b["t"]).date() > since]
                    if highs:
                        entries[sym]["hwm"] = max(entries[sym]["hwm"], max(highs))
            except Exception as e:
                alerts.append(f"Could not refresh highs: {str(e)[:120]}")

        news = signals.get("tickers", {})
        for p in mine:
            sym = p["symbol"]
            reason, stop = exit_plan(p, entries[sym], analysis.get("symbols", {}).get(sym), news.get(sym, {}),
                                     cfg, st, exit_thr, now)
            if cfg.get("kill_switch") and cfg.get("liquidate_on_kill"):
                reason = "kill switch liquidation"
            sells = [o for o in my_orders if o["symbol"] == sym and o["side"] == "sell"]
            qty = float(p["qty"])
            if reason:
                rec = {"symbol": sym, "side": "sell", "reason": reason, "qty": qty,
                       "notional": round(float(p["market_value"]), 2),
                       "realized_pl": round(float(p["unrealized_pl"]), 2)}
                try:
                    for o in sells:
                        alpaca.cancel_order(o["id"])
                    time.sleep(0.5)
                    rec.update(status="submitted", order_id=alpaca.sell_shares(sym, qty, books.cid(BOOK))["id"])
                    state["exits"][sym] = now.isoformat()
                except Exception as e:
                    rec.update(status="error", error=str(e)[:300])
                log(rec)
                continue
            # Keep a stop-limit at the current stop; only ever move it up.
            cur = max((float(o.get("stop_price") or 0) for o in sells), default=None)
            if cur is None or stop > cur * 1.001:
                try:
                    for o in sells:
                        alpaca.cancel_order(o["id"])
                    lim = round(stop * (1 - f["stop_limit_gap"]), 2)
                    alpaca.stop_limit_sell(sym, qty, stop, lim, books.cid(BOOK))
                    print(f"[frac] PROTECT {sym}: stop {stop:.2f} / limit {lim:.2f} (high {entries[sym]['hwm']:.2f})")
                except Exception as e:
                    alerts.append(f"Could not place stop on {sym}: {str(e)[:120]}")
            entries[sym]["stop"] = stop

        # Cancel limit buys whose signal has faded.
        syms = analysis.get("symbols", {})
        for o in [o for o in my_orders if o["side"] == "buy"]:
            a = syms.get(o["symbol"])
            if not a or a["combined"] < st["buy"] or a["factors"].get("trend", 0) <= 0:
                try:
                    alpaca.cancel_order(o["id"])
                    print(f"[frac] CANCEL limit buy {o['symbol']}: signal faded")
                except Exception:
                    pass

        # Gates for new buys.
        risk = cfg["risk"]
        if (equity - last_equity) / last_equity <= -risk["daily_loss_limit_pct"]:
            halts.append("Daily loss limit hit")
        if signals.get("market_risk", 0) > risk["max_market_risk"]:
            halts.append("High market risk")
        if not analysis.get("generated_at") or hours_since(analysis["generated_at"], now) > risk["max_signal_age_hours"]:
            halts.append("Chart analysis is stale")
        if signals.get("scorer") == "keywords" and not risk.get("allow_keyword_trades"):
            halts.append("Claude could not read the news")
        if not halts:
            orders = alpaca.open_orders()
            pending = {o["symbol"] for o in orders if o["side"] == "buy" and books.book_of_order(o) == BOOK}
            pos_syms = {p["symbol"] for p in positions}
            blocked = {s for s in pos_syms | set(owners) if books.owner_of(owners, s, pos_syms) not in (None, BOOK)}
            avail = books.budget(BOOK, cfg, equity, positions, owners, orders)
            plan = plan_entries(analysis, signals, cfg, st, held, blocked, pending, state, avail,
                                state["capital"], now)
            for b in plan:
                rec = {"symbol": b["symbol"], "side": "buy", "reason": b["reason"], "qty": b["qty"],
                       "limit": b["limit"], "notional": b["dollars"],
                       "score": syms[b["symbol"]]["combined"], "factors": syms[b["symbol"]]["factors"]}
                try:
                    owners[b["symbol"]] = BOOK
                    o = alpaca.limit_order(b["symbol"], "buy", b["qty"], b["limit"], books.cid(BOOK))
                    state.setdefault("pending", {})[b["symbol"]] = {"atr": b["atr"], "time": now.isoformat()}
                    state.setdefault("order_reasons", {})[o["id"]] = b["reason"]
                    print(f"[frac] LIMIT BUY {b['symbol']} {b['qty']:g} @ {b['limit']:.2f}: {b['reason']}")
                    results.append({**rec, "status": "order placed", "order_id": o["id"], "time": now.isoformat()})
                except Exception as e:
                    owners.pop(b["symbol"], None)
                    rec.update(status="error", error=str(e)[:300])
                    log(rec)

    books.release_unused(owners, BOOK, alpaca.positions(), alpaca.open_orders())
    books.save_owners(owners)
    state["logged_orders"] = state.get("logged_orders", [])[-2000:]
    write_json("trades.json", trades[-3000:])
    write_json("frac_state.json", state)
    write_json("frac_status.json", {"time": now.isoformat(), "halts": halts, "alerts": alerts,
                                    "last_actions": results, "settings": st,
                                    "learned": {k: v for k, v in (weights.get("frac") or {}).items() if k != "grid"}})
    for m in halts + alerts:
        print("[frac] Note:", m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
