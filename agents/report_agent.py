"""Report step: runs last. Pulls the account, splits P&L by book (stock,
fractional, options) and writes data/status.json + data/equity.json for the
dashboard."""
from __future__ import annotations

import sys

import books
from common import Alpaca, load_config, now_utc, read_json, write_json


def main() -> int:
    cfg = load_config()
    now = now_utc()
    alpaca = Alpaca()
    acct, clock = alpaca.account(), alpaca.clock()
    positions, orders = alpaca.positions(), alpaca.open_orders()
    equity, cash, last_equity = float(acct["equity"]), float(acct["cash"]), float(acct["last_equity"])
    owners = books.load_owners()
    trades = read_json("trades.json", [])
    signals, analysis = read_json("signals.json", {}), read_json("analysis.json", {})

    stock_state = read_json("state.json", {})
    starting = stock_state.get("starting_equity", equity)
    st = {"stock": read_json("stock_status.json", {}), "frac": read_json("frac_status.json", {}),
          "options": read_json("options_status.json", {})}
    bstate = {"stock": stock_state, "frac": read_json("frac_state.json", {}),
              "options": read_json("options_state.json", {})}
    since = {"stock": stock_state.get("started_at"), "frac": bstate["frac"].get("started_at"),
             "options": bstate["options"].get("started_at")}
    capital = {"stock": round(starting * cfg["books"]["stock"]["allocation"], 2),
               "frac": bstate["frac"].get("capital", equity * cfg["books"]["frac"]["allocation"]),
               "options": bstate["options"].get("capital", equity * cfg["books"]["options"]["allocation"])}
    summary = {}
    for b in books.BOOKS:
        summary[b] = {**books.book_summary(b, positions, trades, owners, capital[b], since[b]),
                      "since": since[b], "allocation": cfg["books"][b]["allocation"],
                      "enabled": cfg["books"][b].get("enabled", True),
                      "halts": [h for h in st[b].get("halts", []) if h != "Market closed"],
                      "alerts": st[b].get("alerts", [])}
    summary["frac"]["settings"] = st["frac"].get("settings")
    summary["frac"]["learned"] = st["frac"].get("learned")
    summary["options"]["learned"] = st["options"].get("learned")
    summary["options"]["options_level"] = st["options"].get("options_level")

    stops = {}
    for o in orders:
        if o["side"] == "sell" and o["type"] in ("trailing_stop", "stop_limit", "stop"):
            stops[o["symbol"]] = o
    pos_syms = {p["symbol"] for p in positions}
    rows = []
    for p in positions:
        sym = p["symbol"]
        book = "options" if books.is_option(sym) else (books.owner_of(owners, sym, pos_syms) or "stock")
        so = stops.get(sym) or {}
        row = {"symbol": sym, "book": book, "qty": float(p["qty"]), "avg_entry": float(p["avg_entry_price"]),
               "price": float(p["current_price"]), "value": float(p["market_value"]),
               "pl": float(p["unrealized_pl"]), "pl_pct": float(p["unrealized_plpc"]),
               "opened": (bstate[book].get("entries", {}).get(sym) or {}).get("time"),
               "stop_price": float(so["stop_price"]) if so.get("stop_price") else None,
               "high_water": float(so["hwm"]) if so.get("hwm") else
               (bstate["frac"].get("entries", {}).get(sym) or {}).get("hwm") if book == "frac" else None,
               "stop_type": so.get("type")}
        if book == "options":
            occ = books.parse_occ(sym)
            row.update(underlying=occ["underlying"], option_type=occ["type"], strike=occ["strike"],
                       expiry=occ["expiry"].isoformat(), dte=(occ["expiry"] - now.date()).days)
        rows.append(row)

    status = {
        "last_run": now.isoformat(), "paper": "paper" in alpaca.base, "market_open": clock["is_open"],
        "next_open": clock.get("next_open"), "equity": round(equity, 2), "cash": round(cash, 2),
        "day_pnl": round(equity - last_equity, 2),
        "day_pnl_pct": round((equity - last_equity) / last_equity, 5) if last_equity else 0,
        "starting_equity": starting, "total_pnl": round(equity - starting, 2),
        "total_pnl_pct": round((equity - starting) / starting, 5) if starting else 0,
        "halts": st["stock"].get("halts", []), "alerts": st["stock"].get("alerts", []),
        "last_actions": sum((st[b].get("last_actions", []) for b in books.BOOKS), []),
        "market_risk": signals.get("market_risk"), "signals_at": signals.get("generated_at"),
        "analysis_at": analysis.get("generated_at"),
        "buy_threshold": st["stock"].get("buy_threshold"), "exit_threshold": st["stock"].get("exit_threshold"),
        "trailing_stop_pct": st["stock"].get("trailing_stop_pct"), "max_hold_days": st["stock"].get("max_hold_days"),
        "books": summary, "positions": rows,
    }
    write_json("status.json", status)
    hist = read_json("equity.json", [])
    hist.append({"time": now.isoformat(), "equity": round(equity, 2), "cash": round(cash, 2),
                 "books": {b: summary[b]["pnl"] for b in books.BOOKS}})
    write_json("equity.json", hist[-5000:])
    print(f"Equity ${equity:,.2f} | today {status['day_pnl']:+,.2f} | total {status['total_pnl']:+,.2f}")
    for b in books.BOOKS:
        s = summary[b]
        print(f"  {b:8} P&L {s['pnl']:+,.2f} ({s['pnl_pct']:+.2%} of ${s['capital']:,.0f}) | open {s['open']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
