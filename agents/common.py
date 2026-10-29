"""Shared helpers: config, JSON storage, and a small Alpaca REST client."""
from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_time(s: str) -> dt.datetime:
    """ISO timestamps from us or Alpaca (which uses up to 9 decimal places)."""
    s = re.sub(r"(\.\d{6})\d+", r"\1", s.replace("Z", "+00:00"))
    return dt.datetime.fromisoformat(s)


def read_json(name: str, default):
    p = DATA / name
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return default


def write_json(name: str, obj) -> None:
    p = DATA / name
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(p)


class Alpaca:
    """Minimal Alpaca client. Uses the PAPER endpoint unless LIVE_TRADING is
    set to the exact string 'i-understand-the-risk'."""

    PAPER = "https://paper-api.alpaca.markets"
    LIVE = "https://api.alpaca.markets"
    DATA_URL = "https://data.alpaca.markets"

    def __init__(self):
        key = os.environ["ALPACA_API_KEY"]
        secret = os.environ["ALPACA_SECRET_KEY"]
        live = os.environ.get("LIVE_TRADING", "") == "i-understand-the-risk"
        self.base = self.LIVE if live else self.PAPER
        self.is_paper = not live
        self.s = requests.Session()
        self.s.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})

    def _req(self, method: str, url: str, **kw):
        r = self.s.request(method, url, timeout=30, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"Alpaca {method} {url} -> {r.status_code}: {r.text[:300]}")
        return r.json() if r.text else None

    # --- trading -------------------------------------------------------
    def account(self) -> dict:
        return self._req("GET", f"{self.base}/v2/account")

    def clock(self) -> dict:
        return self._req("GET", f"{self.base}/v2/clock")

    def positions(self) -> list[dict]:
        return self._req("GET", f"{self.base}/v2/positions")

    def open_orders(self) -> list[dict]:
        return self._req("GET", f"{self.base}/v2/orders", params={"status": "open"})

    def latest_price(self, symbol: str) -> float:
        d = self._req("GET", f"{self.DATA_URL}/v2/stocks/{symbol}/trades/latest")
        return float(d["trade"]["p"])

    def buy_notional(self, symbol: str, notional: float) -> dict:
        """Buy a dollar amount. Falls back to whole shares if fractional
        trading isn't available for the symbol."""
        body = {"symbol": symbol, "side": "buy", "type": "market", "time_in_force": "day",
                "notional": f"{notional:.2f}"}
        try:
            return self._req("POST", f"{self.base}/v2/orders", json=body)
        except RuntimeError:
            qty = int(notional // self.latest_price(symbol))
            if qty < 1:
                raise
            body.pop("notional")
            body["qty"] = str(qty)
            return self._req("POST", f"{self.base}/v2/orders", json=body)

    def close_position(self, symbol: str) -> dict:
        return self._req("DELETE", f"{self.base}/v2/positions/{symbol}")

    # --- orders ----------------------------------------------------------
    def submit_order(self, **body) -> dict:
        return self._req("POST", f"{self.base}/v2/orders", json=body)

    def orders(self, **params) -> list[dict]:
        return self._req("GET", f"{self.base}/v2/orders", params=params)

    def get_order(self, order_id: str) -> dict:
        return self._req("GET", f"{self.base}/v2/orders/{order_id}")

    def cancel_order(self, order_id: str) -> None:
        self._req("DELETE", f"{self.base}/v2/orders/{order_id}")

    @staticmethod
    def fmt_qty(qty: float) -> str:
        return f"{qty:.9f}".rstrip("0").rstrip(".")

    @staticmethod
    def _cid(cid):
        return {"client_order_id": cid} if cid else {}

    def buy_shares(self, symbol: str, qty: float, cid: str | None = None) -> dict:
        return self.submit_order(symbol=symbol, qty=self.fmt_qty(qty), side="buy", type="market",
                                 time_in_force="day", **self._cid(cid))

    def sell_shares(self, symbol: str, qty: float, cid: str | None = None) -> dict:
        return self.submit_order(symbol=symbol, qty=self.fmt_qty(qty), side="sell", type="market",
                                 time_in_force="day", **self._cid(cid))

    def limit_order(self, symbol: str, side: str, qty: float, limit: float, cid: str | None = None,
                    tif: str = "day") -> dict:
        return self.submit_order(symbol=symbol, qty=self.fmt_qty(qty), side=side, type="limit",
                                 limit_price=f"{limit:.2f}", time_in_force=tif, **self._cid(cid))

    def stop_limit_sell(self, symbol: str, qty: float, stop: float, limit: float, cid: str | None = None) -> dict:
        """Fractional-friendly protective exit: once the price trades at or
        below `stop`, a limit sell at `limit` is sent. Day orders only."""
        return self.submit_order(symbol=symbol, qty=self.fmt_qty(qty), side="sell", type="stop_limit",
                                 stop_price=f"{stop:.2f}", limit_price=f"{limit:.2f}", time_in_force="day",
                                 **self._cid(cid))

    def trailing_stop(self, symbol: str, qty: int, trail_percent: float, cid: str | None = None) -> dict:
        """Broker-side trailing stop: sells if the price falls trail_percent
        below its highest point since the order was placed. Runs at Alpaca
        continuously, not just when our hourly job runs."""
        return self.submit_order(symbol=symbol, qty=str(qty), side="sell", type="trailing_stop",
                                 trail_percent=f"{trail_percent:.2f}", time_in_force="gtc", **self._cid(cid))

    # --- options -------------------------------------------------------
    def option_contracts(self, underlying: str, kind: str, exp_from: dt.date, exp_to: dt.date,
                         strike_lo: float, strike_hi: float) -> list[dict]:
        out, token = [], None
        while True:
            params = {"underlying_symbols": underlying, "type": kind, "status": "active",
                      "expiration_date_gte": exp_from.isoformat(), "expiration_date_lte": exp_to.isoformat(),
                      "strike_price_gte": f"{strike_lo:.2f}", "strike_price_lte": f"{strike_hi:.2f}", "limit": 1000}
            if token:
                params["page_token"] = token
            d = self._req("GET", f"{self.base}/v2/options/contracts", params=params)
            out.extend(d.get("option_contracts") or [])
            token = d.get("next_page_token")
            if not token:
                return out

    def option_snapshots(self, symbols: list[str]) -> dict[str, dict]:
        """Latest quote, greeks and implied volatility per contract. Uses the
        free 'indicative' feed unless OPTIONS_FEED=opra is set."""
        out = {}
        feed = os.environ.get("OPTIONS_FEED", "indicative")
        for i in range(0, len(symbols), 100):
            d = self._req("GET", f"{self.DATA_URL}/v1beta1/options/snapshots",
                          params={"symbols": ",".join(symbols[i:i + 100]), "feed": feed})
            out.update(d.get("snapshots") or {})
        return out

    # --- price history -------------------------------------------------
    def bars(self, symbols: list[str], timeframe: str, start: dt.datetime, end: dt.datetime | None = None,
             feed: str | None = None) -> dict[str, list[dict]]:
        """Split/dividend-adjusted bars per symbol. Tries the full-market (SIP)
        feed first, which the free plan allows for data older than 15 minutes,
        and falls back to the free IEX feed."""
        iso = lambda d: d.isoformat().replace("+00:00", "Z")
        feeds = [feed] if feed else ["sip", "iex"]
        last_err = None
        for fd in feeds:
            e = end or (now_utc() - dt.timedelta(minutes=16) if fd == "sip" else now_utc())
            out: dict[str, list[dict]] = {s: [] for s in symbols}
            token = None
            try:
                while True:
                    params = {"symbols": ",".join(symbols), "timeframe": timeframe, "start": iso(start),
                              "end": iso(e), "limit": 10000, "adjustment": "all", "feed": fd, "sort": "asc"}
                    if token:
                        params["page_token"] = token
                    d = self._req("GET", f"{self.DATA_URL}/v2/stocks/bars", params=params)
                    for s, bs in (d.get("bars") or {}).items():
                        out.setdefault(s, []).extend(bs)
                    token = d.get("next_page_token")
                    if not token:
                        return out
            except RuntimeError as err:
                last_err = err
        raise last_err

    # --- news ----------------------------------------------------------
    def news(self, symbols: list[str], start: dt.datetime, max_pages: int = 5) -> list[dict]:
        out, token = [], None
        for _ in range(max_pages):
            params = {"symbols": ",".join(symbols), "start": start.isoformat().replace("+00:00", "Z"),
                      "limit": 50, "sort": "desc", "include_content": "false"}
            if token:
                params["page_token"] = token
            d = self._req("GET", f"{self.DATA_URL}/v1beta1/news", params=params)
            out.extend(d.get("news", []))
            token = d.get("next_page_token")
            if not token:
                break
        return out
