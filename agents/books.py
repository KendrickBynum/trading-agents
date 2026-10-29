"""Bookkeeping that lets three strategies share one Alpaca account:

  stock    - whole shares, broker-side trailing stops (trader_agent.py)
  frac     - fractional shares, limit entries, stop-limit exits (frac_agent.py)
  options  - long calls/puts with capped risk (options_agent.py)

Every order is tagged with its book in client_order_id. A stock symbol can
only be held by one share book at a time (owners.json), so the books never
trip over each other's shares. Option contracts are separate symbols."""
from __future__ import annotations

import datetime as dt
import re
import uuid

from common import read_json, write_json

BOOKS = ["stock", "frac", "options"]
PREFIX = {"stock": "stk", "frac": "frc", "options": "opt"}
OCC = re.compile(r"^([A-Z.]{1,6})(\d{6})([CP])(\d{8})$")


def cid(book: str) -> str:
    return f"{PREFIX[book]}-{uuid.uuid4().hex[:24]}"


def is_option(symbol: str) -> bool:
    return bool(OCC.match(symbol or ""))


def parse_occ(symbol: str) -> dict:
    m = OCC.match(symbol)
    u, d, t, k = m.groups()
    return {"underlying": u, "expiry": dt.date(2000 + int(d[:2]), int(d[2:4]), int(d[4:])),
            "type": "call" if t == "C" else "put", "strike": int(k) / 1000}


def book_of_order(o: dict) -> str:
    c = o.get("client_order_id") or ""
    for b, p in PREFIX.items():
        if c.startswith(p + "-"):
            return b
    return "options" if is_option(o.get("symbol", "")) else "stock"   # orders from before tagging


# ------------------------------------------------------------- ownership
def load_owners() -> dict:
    return read_json("owners.json", {})


def save_owners(owners: dict) -> None:
    write_json("owners.json", owners)


def owner_of(owners: dict, symbol: str, positions_syms: set[str]) -> str | None:
    """Which share book holds (or is buying) this symbol. Positions from
    before books existed belong to the stock book."""
    if symbol in owners:
        return owners[symbol]
    return "stock" if symbol in positions_syms else None


def release_unused(owners: dict, book: str, positions: list[dict], open_orders: list[dict]) -> None:
    held = {p["symbol"] for p in positions}
    buying = {o["symbol"] for o in open_orders if o["side"] == "buy" and book_of_order(o) == book}
    for sym in [s for s, b in owners.items() if b == book]:
        if sym not in held and sym not in buying:
            owners.pop(sym)


# ------------------------------------------------------- fills & ledger
def sync_fills(alpaca, book: str, state: dict, trades: list[dict], now: dt.datetime,
               multiplier: float = 1.0, describe=None, include_buys: bool = False) -> list[dict]:
    """Log this book's orders that filled at the broker since the last run
    (stops hitting, limit orders filling) with realized P&L on sells. Books
    that use limit orders log trades here, at fill time, so an order that is
    re-priced a few times is still counted once."""
    after = state.get("last_order_check") or (now - dt.timedelta(days=3)).isoformat()
    logged = set(state.get("logged_orders", []))
    reasons = state.get("order_reasons", {})
    new = []
    for o in alpaca.orders(status="closed", after=after, limit=500, direction="asc"):
        side = o.get("side")
        if (o.get("status") not in ("filled", "partially_filled") or o["id"] in logged
                or book_of_order(o) != book or float(o.get("filled_qty") or 0) <= 0):
            continue
        if side == "buy" and not include_buys:
            continue
        sym, qty, px = o["symbol"], float(o["filled_qty"]), float(o["filled_avg_price"])
        reason = reasons.pop(o["id"], None) or (describe(o) if describe else f"{side} order filled")
        rec = {"time": o.get("filled_at") or now.isoformat(), "book": book, "symbol": sym, "side": side,
               "status": "filled", "order_id": o["id"], "qty": qty, "price": px,
               "notional": round(qty * px * multiplier, 2), "reason": reason}
        if is_option(sym):
            rec["underlying"] = parse_occ(sym)["underlying"]
        if side == "sell":
            entry = state.get("entries", {}).get(sym, {}).get("avg")
            if entry:
                rec["realized_pl"] = round((px - entry) * qty * multiplier, 2)
            state.setdefault("exits", {})[rec.get("underlying", sym)] = rec["time"]
        trades.append(rec)
        new.append(rec)
        logged.add(o["id"])
        print(f"[{book}] FILLED {side.upper()} {sym} {qty:g} @ {px:.2f}: {reason}")
    state["order_reasons"] = dict(list(reasons.items())[-500:])
    state["logged_orders"] = list(logged)[-2000:]
    state["last_order_check"] = now.isoformat()
    return new


def book_positions(book: str, positions: list[dict], owners: dict) -> list[dict]:
    syms = {p["symbol"] for p in positions}
    if book == "options":
        return [p for p in positions if is_option(p["symbol"])]
    return [p for p in positions if not is_option(p["symbol"]) and owner_of(owners, p["symbol"], syms) == book]


def book_summary(book: str, positions: list[dict], trades: list[dict], owners: dict, capital: float,
                 since: str | None) -> dict:
    mine = book_positions(book, positions, owners)
    tr = [t for t in trades if t.get("book", "stock") == book]
    realized = sum(t.get("realized_pl") or 0 for t in tr)
    closed = [t for t in tr if t.get("realized_pl") is not None]
    unreal = sum(float(p["unrealized_pl"]) for p in mine)
    pnl = realized + unreal
    return {"capital": round(capital, 2), "value": round(sum(float(p["market_value"]) for p in mine), 2),
            "realized": round(realized, 2), "unrealized": round(unreal, 2), "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / capital, 5) if capital else 0.0, "open": len(mine),
            "closed_trades": len(closed),
            "win_rate": round(sum(1 for t in closed if t["realized_pl"] > 0) / len(closed), 3) if closed else None,
            "buys": sum(1 for t in tr if t["side"] == "buy" and t.get("status") != "error")}


def budget(book: str, cfg: dict, equity: float, positions: list[dict], owners: dict,
           open_orders: list[dict]) -> float:
    """Money this book may still put to work: its share of the account, minus
    what it already holds and has on order, minus its cash reserve."""
    alloc = cfg["books"][book]["allocation"]
    reserve = cfg["books"][book].get("cash_reserve", 0.0)
    held = sum(abs(float(p["market_value"])) for p in book_positions(book, positions, owners))
    pending = 0.0
    for o in open_orders:
        if o["side"] == "buy" and book_of_order(o) == book:
            px = float(o.get("limit_price") or 0)
            q = float(o.get("qty") or 0)
            pending += float(o.get("notional") or 0) or px * q * (100 if is_option(o["symbol"]) else 1)
    return equity * alloc * (1 - reserve) - held - pending
