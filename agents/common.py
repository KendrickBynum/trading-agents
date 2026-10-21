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
