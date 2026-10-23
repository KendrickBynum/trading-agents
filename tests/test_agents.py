import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agents"))

from news_agent import aggregate, keyword_score  # noqa: E402
from trader_agent import decide  # noqa: E402

NOW = dt.datetime(2026, 10, 20, 15, 0, tzinfo=dt.timezone.utc)

CFG = {
    "kill_switch": False, "liquidate_on_kill": False,
    "watchlist": ["AAPL", "MSFT", "NVDA", "XOM"],
    "risk": {"buy_threshold": 0.45, "sell_threshold": -0.30, "min_articles": 2, "max_positions": 2,
             "max_position_pct": 0.10, "min_cash_pct": 0.20, "stop_loss_pct": 0.05,
             "take_profit_pct": 0.10, "daily_loss_limit_pct": 0.03, "max_market_risk": 0.70,
             "max_signal_age_hours": 3},
}


def sig(**tickers):
    return {"generated_at": NOW.isoformat(), "market_risk": 0.1,
            "scorer": "claude",
            "tickers": {t: {"score": s, "articles": n, "ai_articles": n} for t, (s, n) in tickers.items()}}


def pos(symbol, plpc, value=10000):
    return {"symbol": symbol, "unrealized_plpc": str(plpc), "unrealized_pl": "0", "market_value": str(value)}


def test_buys_strongest_signals_within_limits():
    acts, halts = decide([], sig(AAPL=(0.8, 3), MSFT=(0.6, 2), NVDA=(0.9, 1), XOM=(0.2, 5)),
                         100000, 100000, 0.0, CFG, set(), NOW)
    assert halts == []
    assert [a["symbol"] for a in acts] == ["AAPL", "MSFT"]  # NVDA: too few articles; XOM: too weak
    assert all(a["notional"] == 10000 for a in acts)


def test_cash_reserve_caps_buys():
    acts, _ = decide([], sig(AAPL=(0.8, 3)), 100000, 25000, 0.0, CFG, set(), NOW)
    assert acts[0]["notional"] == 5000  # 25k cash - 20k reserve


def test_stop_loss_take_profit_and_negative_news():
    positions = [pos("AAPL", -0.06), pos("MSFT", 0.12), pos("NVDA", 0.01), pos("TSLA", -0.5)]
    acts, _ = decide(positions, sig(NVDA=(-0.5, 2)), 100000, 50000, 0.0, CFG, set(), NOW)
    sells = {a["symbol"]: a["reason"] for a in acts if a["action"] == "sell"}
    assert set(sells) == {"AAPL", "MSFT", "NVDA"}  # TSLA isn't on the watchlist: untouched
    assert "stop loss" in sells["AAPL"] and "take profit" in sells["MSFT"] and "news" in sells["NVDA"]


def test_daily_loss_limit_blocks_buys_but_not_sells():
    acts, halts = decide([pos("AAPL", -0.07)], sig(MSFT=(0.9, 4)), 100000, 90000, -0.04, CFG, set(), NOW)
    assert any("Daily loss" in h for h in halts)
    assert [a["action"] for a in acts] == ["sell"]


def test_stale_signals_and_market_risk_block_buys():
    s = sig(AAPL=(0.9, 4))
    s["generated_at"] = (NOW - dt.timedelta(hours=5)).isoformat()
    assert decide([], s, 1e5, 1e5, 0, CFG, set(), NOW)[0] == []
    s = sig(AAPL=(0.9, 4))
    s["market_risk"] = 0.9
    assert decide([], s, 1e5, 1e5, 0, CFG, set(), NOW)[0] == []


def test_kill_switch():
    cfg = {**CFG, "kill_switch": True}
    acts, halts = decide([pos("AAPL", 0.0)], sig(MSFT=(0.9, 4)), 1e5, 1e5, 0, cfg, set(), NOW)
    assert acts == [] and halts
    cfg["liquidate_on_kill"] = True
    acts, _ = decide([pos("AAPL", 0.0)], sig(MSFT=(0.9, 4)), 1e5, 1e5, 0, cfg, set(), NOW)
    assert acts == [{"action": "sell", "symbol": "AAPL", "reason": "kill switch liquidation"}]


def test_pending_orders_not_duplicated():
    acts, _ = decide([], sig(AAPL=(0.9, 4)), 1e5, 1e5, 0, CFG, {"AAPL"}, NOW)
    assert acts == []


def test_aggregate_weights_recent_news_more():
    arts = [
        {"created_at": (NOW - dt.timedelta(hours=1)).isoformat(), "headline": "new", "market_risk": 0.2,
         "impacts": [{"ticker": "AAPL", "sentiment": 1.0, "confidence": 1.0}]},
        {"created_at": (NOW - dt.timedelta(hours=20)).isoformat(), "headline": "old", "market_risk": 0.0,
         "impacts": [{"ticker": "AAPL", "sentiment": -1.0, "confidence": 1.0}]},
    ]
    out = aggregate(arts, ["AAPL", "MSFT"], NOW, 8)
    assert out["tickers"]["AAPL"]["score"] > 0.5
    assert out["tickers"]["AAPL"]["articles"] == 2
    assert out["tickers"]["AAPL"]["headlines"][0]["headline"] == "new"
    assert out["tickers"]["MSFT"] == {"score": 0.0, "articles": 0, "ai_articles": 0, "headlines": []}


def test_keyword_fallback():
    r = keyword_score({"id": 1, "headline": "Apple beats estimates, shares surge", "symbols": ["AAPL", "ZZZ"]},
                      {"AAPL"})
    assert r["impacts"][0]["ticker"] == "AAPL" and r["impacts"][0]["sentiment"] > 0


def test_keyword_only_news_never_trades():
    s = sig(AAPL=(0.9, 4), MSFT=(-0.9, 4))
    s["scorer"] = "keywords"
    for v in s["tickers"].values():
        v["ai_articles"] = 0
    acts, halts = decide([pos("MSFT", 0.0)], s, 1e5, 1e5, 0, CFG, set(), NOW)
    assert acts == []  # no buy on AAPL, no news-based sell on MSFT
    assert any("Claude" in h for h in halts)
    # stop-loss still works without Claude
    acts, _ = decide([pos("MSFT", -0.08)], s, 1e5, 1e5, 0, CFG, set(), NOW)
    assert acts[0]["action"] == "sell" and "stop loss" in acts[0]["reason"]


def test_keyword_scores_are_damped():
    r = keyword_score({"id": 1, "headline": "Apple shares surge", "symbols": ["AAPL"]}, {"AAPL"})
    assert 0 < r["impacts"][0]["sentiment"] < 0.5


def test_neutral_news_does_not_dilute():
    mk = lambda h, s: {"created_at": NOW.isoformat(), "headline": h, "market_risk": 0,
                       "impacts": [{"ticker": "AAPL", "sentiment": s, "confidence": 0.8}]}
    arts = [mk("big news", 0.7)] + [mk(f"routine {i}", 0.05) for i in range(8)]
    diluted = aggregate(arts, ["AAPL"], NOW, 8)["tickers"]["AAPL"]
    clean = aggregate(arts, ["AAPL"], NOW, 8, neutral_cutoff=0.15)["tickers"]["AAPL"]
    assert diluted["score"] < 0.15 and clean["score"] == 0.7 and clean["articles"] == 1
