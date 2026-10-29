"""Chart agent: reads price and volume history for the watchlist, scores
trend, support/resistance, order flow and chart patterns, combines them with
the news agent's view, and writes data/analysis.json for the trader.

Re-learns factor weights from history once a week (see learn.py)."""
from __future__ import annotations

import datetime as dt
import sys

import numpy as np

import indicators as ind
import learn
from common import Alpaca, load_config, now_utc, parse_time, read_json, write_json


def news_factor(sig: dict, cfg: dict) -> float | None:
    """News agent score -> factor, or None when there isn't enough news that
    Claude actually read (the factor is then left out, not counted as 0)."""
    if not sig or sig.get("ai_articles", 0) < cfg["risk"]["min_articles"]:
        return None
    return float(np.clip(sig.get("score", 0) * cfg["news"].get("news_scale", 2.0), -1, 1))


def analyse(sym: str, s: dict, intraday: list[dict], lib: dict, sig: dict, weights: dict,
            cfg: dict, now: dt.datetime) -> dict:
    price = intraday[-1]["c"] if intraday else float(s["c"][-1])
    k = cfg.get("analysis", {}).get("pattern_neighbors", 50)
    factors, details = {}, {}
    factors["trend"], details["trend"] = ind.trend_factor(s)
    factors["levels"], details["levels"] = ind.levels_factor(s)
    factors["flow"], details["flow"] = ind.flow_factor(s, intraday, price)
    factors["pattern"], details["pattern"] = ind.pattern_factor(lib, s["c"], len(s["c"]) - 1, now.date(), k)
    nf = news_factor(sig, cfg)
    if nf is not None:
        factors["news"] = nf
    combined = ind.combine(factors, weights)
    return {"price": round(price, 2), "combined": round(combined, 3),
            "factors": {f: round(v, 3) for f, v in factors.items()},
            "agreeing": sum(1 for v in factors.values() if v > 0.1),
            "opposing": sum(1 for v in factors.values() if v < -0.1),
            "news_score": sig.get("score") if sig else None, "news_articles": sig.get("ai_articles", 0) if sig else 0,
            **{k2: v for k2, v in details.items() if v}}


def main() -> int:
    cfg = load_config()
    acfg = cfg.get("analysis", {})
    watch = [s.upper() for s in cfg["watchlist"]]
    now = now_utc()
    alpaca = Alpaca()

    years = acfg.get("history_years", 3)
    daily = alpaca.bars(watch, "1Day", now - dt.timedelta(days=int(365 * years) + 10))
    intraday = alpaca.bars(watch, "15Min", now - dt.timedelta(days=4), feed="iex")
    series = {s: ind.arrays(b) for s, b in daily.items() if len(b) >= 150}
    print(f"History: {', '.join(f'{s} {len(b)}d' for s, b in daily.items())}")
    if not series:
        print("No usable price history", file=sys.stderr)
        return 1

    weights_doc = read_json("weights.json", {})
    factor_log = read_json("factor_log.json", [])
    stale = (not weights_doc or (now - parse_time(weights_doc["learned_at"])).days >= acfg.get("relearn_days", 7))
    if stale:
        print("Re-learning factor weights from history...")
        try:
            weights_doc = learn.run(series, cfg, factor_log, now)
            write_json("weights.json", weights_doc)
            print("Weights:", weights_doc["weights"], "Thresholds:", weights_doc["thresholds"])
        except Exception as e:
            print(f"::warning::Learning failed: {e}", file=sys.stderr)
    prior = acfg.get("prior_weights", {"news": 0.30, "trend": 0.20, "levels": 0.15, "flow": 0.15, "pattern": 0.20})
    weights = weights_doc.get("weights", prior)

    lib = ind.build_library(series)
    signals = read_json("signals.json", {})
    sigs = signals.get("tickers", {})
    out = {}
    for sym in watch:
        if sym not in series:
            continue
        today = [b for b in intraday.get(sym, []) if parse_time(b["t"]).date() == now.date()]
        try:
            out[sym] = analyse(sym, series[sym], today, lib, sigs.get(sym, {}), weights, cfg, now)
        except Exception as e:
            print(f"::warning::{sym} analysis failed: {e}", file=sys.stderr)

    thresholds = weights_doc.get("thresholds", {"buy": 0.25, "sell": -0.25})
    write_json("analysis.json", {"generated_at": now.isoformat(), "weights": weights, "thresholds": thresholds,
                                 "symbols": out})

    # Daily snapshot of every factor, so the learning step can later check
    # which ones (news included) actually predicted moves.
    day = now.date().isoformat()
    if now.weekday() < 5:
        logged = {(e["date"], e["symbol"]) for e in factor_log}
        for sym, a in out.items():
            if (day, sym) not in logged:
                factor_log.append({"date": day, "symbol": sym, "price": a["price"], "factors": a["factors"]})
        write_json("factor_log.json", factor_log[-20000:])

    for sym, a in sorted(out.items(), key=lambda kv: -kv[1]["combined"]):
        f = a["factors"]
        print(f"{sym:5} {a['combined']:+.2f} | " + " ".join(f"{k} {v:+.2f}" for k, v in f.items()))
    print(f"Buy at >= {thresholds['buy']:+.2f}, exit at <= {thresholds['sell']:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
