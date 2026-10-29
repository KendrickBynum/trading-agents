import datetime as dt
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agents"))

import indicators as ind  # noqa: E402
import learn  # noqa: E402
from news_agent import aggregate, keyword_score  # noqa: E402
from trader_agent import decide, protection_needed  # noqa: E402

NOW = dt.datetime(2026, 10, 28, 15, 0, tzinfo=dt.timezone.utc)

CFG = {
    "kill_switch": False, "liquidate_on_kill": False,
    "watchlist": ["AAPL", "MSFT", "NVDA", "XOM"],
    "risk": {"buy_threshold": "auto", "min_agreeing_factors": 3, "min_articles": 2, "max_positions": 2,
             "max_position_pct": 0.10, "min_cash_pct": 0.20, "stop_loss_pct": 0.05, "take_profit_pct": None,
             "news_sell_threshold": -0.30, "exit_threshold": "auto", "max_hold_days": 14,
             "reentry_cooldown_hours": 20, "trailing_stop_pct": 0.04, "daily_loss_limit_pct": 0.03,
             "max_market_risk": 0.70, "max_signal_age_hours": 3},
}


def an(**syms):
    """analysis.json stand-in: SYM=(combined, agreeing, price)."""
    return {"generated_at": NOW.isoformat(), "thresholds": {"buy": 0.3, "sell": -0.3},
            "symbols": {s: {"combined": c, "agreeing": a, "price": p,
                            "factors": {"trend": 0.5, "levels": 0.4, "flow": 0.3, "pattern": -0.1}}
                        for s, (c, a, p) in syms.items()}}


def news(**t):
    return {"generated_at": NOW.isoformat(), "market_risk": 0.1, "scorer": "claude",
            "tickers": {s: {"score": v, "articles": n, "ai_articles": n} for s, (v, n) in t.items()}}


def pos(symbol, plpc, value=10000):
    return {"symbol": symbol, "unrealized_plpc": str(plpc), "unrealized_pl": "0",
            "market_value": str(value), "qty": "10"}


def run(positions=(), analysis=None, signals=None, cash=1e5, day=0.0, pending=(), state=None, cfg=CFG):
    return decide(list(positions), analysis or an(), signals or news(), 1e5, cash, day, cfg,
                  set(pending), NOW, state or {})


# ------------------------------------------------------------------ trader
def test_buys_whole_shares_on_strong_combined_score():
    acts, halts = run(analysis=an(AAPL=(0.6, 4, 250.0), MSFT=(0.4, 3, 500.0), NVDA=(0.9, 2, 100.0),
                                  XOM=(0.2, 4, 110.0)))
    assert halts == []
    assert [(a["symbol"], a["qty"]) for a in acts] == [("AAPL", 40), ("MSFT", 20)]  # NVDA: too few agree


def test_bad_news_vetoes_buy():
    acts, _ = run(analysis=an(AAPL=(0.6, 4, 250.0)), signals=news(AAPL=(-0.5, 3)))
    assert acts == []


def test_cooldown_blocks_quick_rebuy():
    state = {"exits": {"AAPL": (NOW - dt.timedelta(hours=5)).isoformat()}}
    assert run(analysis=an(AAPL=(0.6, 4, 250.0)), state=state)[0] == []
    state = {"exits": {"AAPL": (NOW - dt.timedelta(hours=30)).isoformat()}}
    assert run(analysis=an(AAPL=(0.6, 4, 250.0)), state=state)[0][0]["symbol"] == "AAPL"


def test_sell_rules():
    old = (NOW - dt.timedelta(days=15)).isoformat()
    state = {"entries": {"NVDA": {"time": old}, "XOM": {"time": old}}}
    positions = [pos("AAPL", -0.06), pos("MSFT", 0.01), pos("NVDA", 0.02), pos("XOM", 0.02)]
    analysis = an(MSFT=(-0.5, 0, 500.0), NVDA=(0.1, 3, 100.0), XOM=(0.5, 4, 110.0))
    acts, _ = run(positions, analysis, state=state)
    sells = {a["symbol"]: a["reason"] for a in acts if a["action"] == "sell"}
    assert "stop loss" in sells["AAPL"]
    assert "signals turned negative" in sells["MSFT"]
    assert "max hold" in sells["NVDA"]
    assert "XOM" not in sells  # held 15 days but still a strong buy signal: keep


def test_no_take_profit_cap_when_disabled():
    acts, _ = run([pos("AAPL", 0.25)], an(AAPL=(0.1, 3, 250.0)))
    assert acts == []  # big winner stays; the trailing stop protects it


def test_gates_pause_buys_but_not_sells():
    acts, halts = run([pos("AAPL", -0.07)], an(MSFT=(0.9, 5, 500.0)), day=-0.04)
    assert any("Daily loss" in h for h in halts) and [a["action"] for a in acts] == ["sell"]
    stale = an(MSFT=(0.9, 5, 500.0))
    stale["generated_at"] = (NOW - dt.timedelta(hours=5)).isoformat()
    assert run(analysis=stale)[0] == []
    kw = news()
    kw["scorer"] = "keywords"
    assert run(analysis=an(MSFT=(0.9, 5, 500.0)), signals=kw)[0] == []


def test_kill_switch():
    cfg = {**CFG, "kill_switch": True}
    acts, halts = run([pos("AAPL", 0.0)], an(MSFT=(0.9, 5, 500.0)), cfg=cfg)
    assert acts == [] and halts
    cfg["liquidate_on_kill"] = True
    acts, _ = run([pos("AAPL", 0.0)], an(MSFT=(0.9, 5, 500.0)), cfg=cfg)
    assert acts == [{"action": "sell", "symbol": "AAPL", "reason": "kill switch liquidation"}]


def test_protection_needed():
    positions = [{"symbol": "AAPL", "qty": "40.4"}, {"symbol": "MSFT", "qty": "0.5"},
                 {"symbol": "XOM", "qty": "12"}, {"symbol": "TSLA", "qty": "5"}]
    orders = [{"symbol": "XOM", "side": "sell", "type": "trailing_stop"}]
    assert protection_needed(positions, orders, {"AAPL", "MSFT", "XOM"}) == [("AAPL", 40)]


# -------------------------------------------------------------- indicators
def bars_from(close, vol=None, start="2023-01-02"):
    d0 = np.datetime64(start)
    vol = vol if vol is not None else np.full(len(close), 1e6)
    return {"t": d0 + np.arange(len(close)), "o": close, "h": close * 1.01, "l": close * 0.99,
            "c": close, "v": np.asarray(vol, float)}


def test_trend_sign():
    up = bars_from(100 * np.exp(np.linspace(0, 0.3, 120)))
    down = bars_from(100 * np.exp(np.linspace(0, -0.3, 120)))
    assert ind.trend_factor(up)[0] > 0.3 and ind.trend_factor(down)[0] < -0.3


def test_levels_near_support_vs_resistance():
    x = np.arange(160)
    wave = 100 + 10 * np.sin(x / 8)                      # bounces between ~90 and ~110
    near_top = bars_from(np.append(wave, 109.0))
    near_bottom = bars_from(np.append(wave, 91.0))
    f_top, d_top = ind.levels_factor(near_top)
    f_bot, d_bot = ind.levels_factor(near_bottom)
    assert f_top < 0 < f_bot
    assert d_top["resistance"] and d_top["resistance"] > 109 and d_bot["support"] < 91


def test_flow_pressure_sign():
    c = np.full(60, 100.0)
    s = bars_from(c)
    buying = {**s, "h": c + 1, "l": c - 1, "c": c + 0.9}     # closing near the highs
    selling = {**s, "h": c + 1, "l": c - 1, "c": c - 0.9}
    assert ind.flow_factor(buying)[0] > 0.3 and ind.flow_factor(selling)[0] < -0.3


def test_pattern_finds_repeating_setup():
    # A shape that always precedes a rise: a series made of repeated cycles.
    rng = np.random.default_rng(0)
    cycle = np.concatenate([np.linspace(0, -0.08, 20), np.linspace(-0.08, 0.04, 10)])
    logp = np.concatenate([cycle + 0.04 * k for k in range(30)]) + rng.normal(0, 0.002, 900)
    s = bars_from(100 * np.exp(logp))
    lib = ind.build_library({"X": s})
    i = 30 * 25 + 19                                       # bottom of a dip, rise follows
    f, d = ind.pattern_factor(lib, s["c"], i, s["t"][i])
    assert f > 0.3 and d["win_rate"] > 0.7


def test_learning_on_random_walk_finds_no_edge():
    rng = np.random.default_rng(1)
    series = {f"S{k}": bars_from(100 * np.exp(np.cumsum(rng.normal(0, 0.015, 700))),
                                 rng.uniform(5e5, 2e6, 700)) for k in range(4)}
    cfg = {"analysis": {"pattern_neighbors": 30}, "risk": {}}
    w = learn.run(series, cfg, [], NOW)
    assert abs(sum(w["weights"].values()) - 1) < 1e-3
    for f in ind.TECHNICAL:
        assert abs(w["factors"][f]["ic"]) < 0.15        # pure noise: nothing should look predictive
    assert w["thresholds"]["buy"] >= 0.12 and w["thresholds"]["sell"] <= -0.12
    assert not w["news_learned"]


def test_weights_reward_predictive_factors():
    stats = {"trend": {"ic": 0.10, "n": 1000}, "levels": {"ic": -0.05, "n": 1000}, "flow": {"ic": 0.0, "n": 1000},
             "pattern": {"ic": 0.02, "n": 1000}}
    w = learn.weights_from(stats, {"news": 0.3, "trend": 0.2, "levels": 0.15, "flow": 0.15, "pattern": 0.2})
    assert w["trend"] > 0.2 > w["levels"] and abs(sum(w.values()) - 1) < 1e-3


# -------------------------------------------------------------- news agent
def test_aggregate_weights_recent_news_more():
    arts = [
        {"created_at": (NOW - dt.timedelta(hours=1)).isoformat(), "headline": "new", "market_risk": 0.2,
         "impacts": [{"ticker": "AAPL", "sentiment": 1.0, "confidence": 1.0}]},
        {"created_at": (NOW - dt.timedelta(hours=20)).isoformat(), "headline": "old", "market_risk": 0.0,
         "impacts": [{"ticker": "AAPL", "sentiment": -1.0, "confidence": 1.0}]},
    ]
    out = aggregate(arts, ["AAPL", "MSFT"], NOW, 8)
    assert out["tickers"]["AAPL"]["score"] > 0.5 and out["tickers"]["AAPL"]["articles"] == 2


def test_neutral_news_does_not_dilute():
    mk = lambda h, s: {"created_at": NOW.isoformat(), "headline": h, "market_risk": 0,
                       "impacts": [{"ticker": "AAPL", "sentiment": s, "confidence": 0.8}]}
    arts = [mk("big news", 0.7)] + [mk(f"routine {i}", 0.05) for i in range(8)]
    assert aggregate(arts, ["AAPL"], NOW, 8, neutral_cutoff=0.15)["tickers"]["AAPL"]["score"] == 0.7


def test_keyword_fallback():
    r = keyword_score({"id": 1, "headline": "Apple beats estimates, shares surge", "symbols": ["AAPL"]}, {"AAPL"})
    assert 0 < r["impacts"][0]["sentiment"] < 1


def test_noise_level_edges_do_not_move_weights():
    prior = {"news": 0.3, "trend": 0.2, "levels": 0.15, "flow": 0.15, "pattern": 0.2}
    stats = {"trend": {"ic": -0.09, "n": 1250}, "levels": {"ic": -0.02, "n": 1250},
             "flow": {"ic": -0.07, "n": 1250}, "pattern": {"ic": 0.007, "n": 1250}}
    assert learn.weights_from(stats, prior) == prior   # tiny pattern "edge" is luck; keep starting weights
