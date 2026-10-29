"""Learning step: replays the chart factors over years of history to measure
which ones actually predicted the next week's move, and sets each factor's
weight from that. News can't be replayed (we only have the news the agents
have read live), so its weight is learned from the live log once enough
outcomes are in.

Results go to data/weights.json and are shown on the dashboard, including an
honest out-of-sample check: weights are learned on the older two thirds of
history and tested on the most recent third, which they never saw."""
from __future__ import annotations

import datetime as dt

import numpy as np

import indicators as ind

MIN_NEWS_SAMPLES = 40


def rank_ic(a, b) -> float:
    """Spearman correlation: do higher scores go with higher future returns?"""
    if len(a) < 10:
        return 0.0
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def factor_stats(f, z, raw) -> dict:
    f, z, raw = np.asarray(f), np.asarray(z), np.asarray(raw)
    bull, bear = f > 0.2, f < -0.2
    pct = lambda m: round(float((raw[m] > 0).mean()), 3) if m.sum() >= 10 else None
    avg = lambda m: round(float(raw[m].mean()), 4) if m.sum() >= 10 else None
    return {"ic": round(rank_ic(f, z), 4), "n": int(len(f)),
            "bull_n": int(bull.sum()), "bull_up_rate": pct(bull), "bull_avg_return": avg(bull),
            "bear_n": int(bear.sum()), "bear_up_rate": pct(bear), "bear_avg_return": avg(bear),
            "base_up_rate": pct(np.ones(len(f), bool)), "base_avg_return": avg(np.ones(len(f), bool))}


def backtest_samples(series: dict[str, dict], lib: dict, step: int = 5, k: int = 50) -> list[dict]:
    """Factor scores on past dates (using only data available then) plus what
    the stock did over the following PATTERN_HORIZON days."""
    H = ind.PATTERN_HORIZON
    out = []
    for name, s in series.items():
        c = s["c"]
        for i in range(130, len(c) - H, step):
            sub = ind.upto(s, i)
            _, vol = ind.pattern_features(c, i)
            if vol is None:
                continue
            row = {"sym": name, "date": s["t"][i], "i": i}
            row["trend"], _ = ind.trend_factor(sub)
            row["levels"], _ = ind.levels_factor(sub)
            row["flow"], _ = ind.flow_factor(sub)
            row["pattern"], _ = ind.pattern_factor(lib, c, i, s["t"][i], k)
            ret = np.log(c[i + H] / c[i])
            row["z"], row["raw"] = ret / (vol * np.sqrt(H)), float(np.expm1(ret))
            out.append(row)
    out.sort(key=lambda r: r["date"])
    return out


def significant_ic(st: dict) -> float:
    """Only count a factor's edge if it's bigger than luck would produce:
    roughly two standard errors (2/sqrt(n)) above zero."""
    n = st.get("n", 0)
    ic = st.get("ic") or 0.0
    return ic if n >= 30 and ic > 2 / np.sqrt(n) else 0.0


def weights_from(stats: dict, prior: dict) -> dict:
    """Half prior belief, half evidence. Factors without a real, positive
    edge keep only half their starting weight; if no factor shows a real
    edge, the starting weights are kept unchanged."""
    have = {f: significant_ic(stats[f]) for f in prior if f in stats}
    tot = sum(have.values())
    if tot <= 0:
        return {f: round(p / sum(prior.values()), 4) for f, p in prior.items()}
    pool = 0.5 * sum(prior[g] for g in have)
    w = {f: (0.5 * p + pool * have[f] / tot) if f in have else p for f, p in prior.items()}
    s = sum(w.values())
    return {f: round(v / s, 4) for f, v in w.items()}


def news_outcomes(factor_log: list[dict], series: dict[str, dict]) -> list[dict]:
    """Match each logged daily snapshot to what the stock did over the next
    PATTERN_HORIZON trading days."""
    H, out = ind.PATTERN_HORIZON, []
    for e in factor_log:
        s = series.get(e["symbol"])
        if s is None or "news" not in e["factors"]:
            continue
        d = np.datetime64(e["date"], "D")
        i = int(np.searchsorted(s["t"], d))
        if i >= len(s["t"]) or i + H >= len(s["t"]) or s["t"][i] != d:
            continue
        c = s["c"]
        r = np.diff(np.log(c[max(0, i - ind.VOL_LOOKBACK):i + 1]))
        vol = r.std() or 0.02
        ret = np.log(c[i + H] / c[i])
        out.append({"news": e["factors"]["news"], "z": ret / (vol * np.sqrt(H)), "raw": float(np.expm1(ret))})
    return out


def run(series: dict[str, dict], cfg: dict, factor_log: list[dict], now: dt.datetime) -> dict:
    acfg = cfg.get("analysis", {})
    prior = acfg.get("prior_weights", {"news": 0.30, "trend": 0.20, "levels": 0.15, "flow": 0.15, "pattern": 0.20})
    k = acfg.get("pattern_neighbors", 50)
    lib = ind.build_library(series)
    samples = backtest_samples(series, lib, k=k)
    if len(samples) < 200:
        raise RuntimeError(f"Not enough history to learn from ({len(samples)} samples)")

    col = lambda rows, f: np.array([r[f] for r in rows])
    stats = {f: factor_stats(col(samples, f), col(samples, "z"), col(samples, "raw")) for f in ind.TECHNICAL}

    news_rows = news_outcomes(factor_log, series)
    news_stats = (factor_stats(col(news_rows, "news"), col(news_rows, "z"), col(news_rows, "raw"))
                  if news_rows else {"n": 0})
    use = dict(stats)
    if news_stats["n"] >= MIN_NEWS_SAMPLES:
        use["news"] = news_stats
    weights = weights_from(use, prior)

    # Honest check: learn on the older 2/3, test on the newest 1/3.
    cut = int(len(samples) * 2 / 3)
    train, test = samples[:cut], samples[cut:]
    tr_stats = {f: factor_stats(col(train, f), col(train, "z"), col(train, "raw")) for f in ind.TECHNICAL}
    tr_w = weights_from(tr_stats, {f: prior[f] for f in ind.TECHNICAL})
    test_scores = np.array([ind.combine({f: r[f] for f in ind.TECHNICAL}, tr_w) for r in test])
    test_raw = col(test, "raw")
    top = test_scores >= np.percentile(test_scores, 85)
    oos = {"from": str(test[0]["date"]), "to": str(test[-1]["date"]), "n": len(test),
           "top_picks_avg_return": round(float(test_raw[top].mean()), 4),
           "top_picks_up_rate": round(float((test_raw[top] > 0).mean()), 3),
           "all_avg_return": round(float(test_raw.mean()), 4),
           "all_up_rate": round(float((test_raw > 0).mean()), 3),
           "ic": round(rank_ic(test_scores, col(test, "z")), 4)}

    # Buy/sell bars: where the combined score sits historically.
    tech_w = {f: weights[f] for f in ind.TECHNICAL}
    scores = np.array([ind.combine({f: r[f] for f in ind.TECHNICAL}, tech_w) for r in samples])
    rcfg = cfg.get("risk", {})
    buy = max(float(np.percentile(scores, rcfg.get("buy_percentile", 85))), 0.12)
    sell = min(float(np.percentile(scores, rcfg.get("sell_percentile", 10))), -0.12)

    stats["news"] = news_stats
    try:
        frac = learn_frac(series, samples, cfg, {f: prior[f] for f in ind.TECHNICAL})
    except Exception as e:  # fractional settings fall back to defaults
        frac = {"error": str(e)[:200]}
    return {"frac": frac, "learned_at": now.isoformat(), "horizon_days": ind.PATTERN_HORIZON,
            "history_from": str(samples[0]["date"]), "history_to": str(samples[-1]["date"]),
            "samples": len(samples), "weights": weights, "prior_weights": prior,
            "factors": stats, "news_learned": news_stats["n"] >= MIN_NEWS_SAMPLES,
            "thresholds": {"buy": round(buy, 3), "sell": round(sell, 3)}, "out_of_sample": oos}


# ------------------------------------------------- fractional-book learning
def atr_series(h, l, c, n: int = 14):
    prev = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    out = np.convolve(tr, np.ones(n) / n, mode="full")[:len(tr)]
    out[:n] = tr[:n].mean()
    return out


def simulate_frac(s: dict, atr_arr, i: int, k: float, m: float, hold: int = 10, cost: float = 0.001):
    """Signal at the close of day i. Next day: a limit buy k ATRs below that
    close (fills at the open if the stock opens below it). Then a stop m ATRs
    under the highest price since buying, raised as the stock rises. Exits at
    the open if the price gaps through the stop (the agent's market-sell
    fallback). Returns the trade's return, or None if the limit never filled."""
    o, h, l, c = s["o"], s["h"], s["l"], s["c"]
    j = i + 1
    if j + 1 >= len(c):
        return None
    a = atr_arr[i]
    limit = c[i] - k * a
    if o[j] <= limit:
        fill = o[j]
    elif l[j] <= limit:
        fill = limit
    else:
        return None
    hwm = max(fill, c[j])
    end = min(j + hold, len(c) - 1)
    for d in range(j + 1, end + 1):
        stop = max(fill - m * a, hwm - m * a)
        if o[d] <= stop:
            return o[d] / fill - 1 - cost
        if l[d] <= stop:
            return stop / fill - 1 - cost
        hwm = max(hwm, h[d])
    return c[end] / fill - 1 - cost


def learn_frac(series: dict[str, dict], samples: list[dict], cfg: dict, tech_w: dict) -> dict:
    fcfg = cfg.get("books", {}).get("frac", {})
    hold = fcfg.get("max_hold_days", 10)
    default = (fcfg.get("default_limit_atr", 0.25), fcfg.get("default_stop_atr", 2.0))
    grid = [(k, m) for k in (0.0, 0.25, 0.5, 0.75, 1.0) for m in (1.5, 2.0, 2.5, 3.0)]
    scores = np.array([ind.combine({f: r[f] for f in ind.TECHNICAL}, tech_w) for r in samples])
    bar = float(np.percentile(scores, fcfg.get("buy_percentile", 75)))
    sig = [r for r, sc in zip(samples, scores) if sc >= bar and r["trend"] > 0]
    if len(sig) < 60:
        raise RuntimeError(f"only {len(sig)} historical signals")
    atrs = {n: atr_series(s["h"], s["l"], s["c"]) for n, s in series.items()}

    def evaluate(rows, k, m):
        rets = [simulate_frac(series[r["sym"]], atrs[r["sym"]], r["i"], k, m, hold) for r in rows]
        filled = [x for x in rets if x is not None]
        return {"signals": len(rows), "fill_rate": round(len(filled) / len(rows), 3) if rows else 0,
                "avg_trade": round(float(np.mean(filled)), 4) if filled else 0.0,
                "win_rate": round(float(np.mean([x > 0 for x in filled])), 3) if filled else None,
                "per_signal": round(float(np.sum(filled)) / len(rows), 5) if rows else 0.0}

    cut = int(len(sig) * 2 / 3)
    train, test = sig[:cut], sig[cut:]
    tr = {g: evaluate(train, *g) for g in grid}
    best = max(grid, key=lambda g: tr[g]["per_signal"])
    best_test, def_test = evaluate(test, *best), evaluate(test, *default)
    adopt = best != default and best_test["per_signal"] > def_test["per_signal"]
    chosen = best if adopt else default
    return {"limit_atr": chosen[0], "stop_atr": chosen[1], "buy_threshold": round(max(bar, 0.08), 3),
            "adopted_learned": adopt, "signals": len(sig),
            "best_on_train": {"limit_atr": best[0], "stop_atr": best[1], **tr[best]},
            "test_best": best_test, "test_default": def_test,
            "test_period": [str(test[0]["date"]), str(test[-1]["date"])],
            "grid": [{"limit_atr": g[0], "stop_atr": g[1], **tr[g]} for g in grid]}
