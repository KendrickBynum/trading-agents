"""Shared helpers: config, JSON storage, and a small Alpaca REST client."""
from __future__ import annotations

import datetime as dt
import json
import os
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
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


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

    def buy_shares(self, symbol: str, qty: int) -> dict:
        return self.submit_order(symbol=symbol, qty=str(qty), side="buy", type="market", time_in_force="day")

    def sell_shares(self, symbol: str, qty: float) -> dict:
        return self.submit_order(symbol=symbol, qty=f"{qty:.9f}".rstrip("0").rstrip("."), side="sell",
                                 type="market", time_in_force="day")

    def trailing_stop(self, symbol: str, qty: int, trail_percent: float) -> dict:
        """Broker-side trailing stop: sells if the price falls trail_percent
        below its highest point since the order was placed. Runs at Alpaca
        continuously, not just when our hourly job runs."""
        return self.submit_order(symbol=symbol, qty=str(qty), side="sell", type="trailing_stop",
                                 trail_percent=f"{trail_percent:.2f}", time_in_force="gtc")

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
