"""Chart analysis used by the chart agent. Everything here is a pure function
of price/volume arrays, so it can be unit-tested and replayed on history
(which is how the learning step measures what actually works).

Each factor returns a score from -1 (bearish) to +1 (bullish) plus details:
  trend    - is the stock in an uptrend (moving averages, momentum, RSI)?
  levels   - support/resistance: room to run before resistance, bounce off
             support, breakouts and breakdowns
  flow     - order-flow proxy: buying vs selling pressure inside each bar,
             unusual volume, and price vs today's VWAP
  pattern  - how similar past chart shapes played out over the next days
"""
from __future__ import annotations

import numpy as np

PATTERN_LEN = 20       # days of price shape compared
PATTERN_HORIZON = 5    # trading days ahead we try to predict
VOL_LOOKBACK = 60      # days used to normalise volatility


def arrays(bars: list[dict]) -> dict:
    return {
        "t": np.array([b["t"][:10] for b in bars], dtype="datetime64[D]"),
        "o": np.array([b["o"] for b in bars], float),
        "h": np.array([b["h"] for b in bars], float),
        "l": np.array([b["l"] for b in bars], float),
        "c": np.array([b["c"] for b in bars], float),
        "v": np.array([b["v"] for b in bars], float),
    }


def upto(s: dict, i: int) -> dict:
    """Series truncated to bars 0..i inclusive (no look-ahead)."""
    return {k: v[: i + 1] for k, v in s.items()}


def atr(h, l, c, n: int = 14) -> float:
    prev = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    return float(tr[-n:].mean())


def rsi(c, n: int = 14) -> float:
    d = np.diff(c[-(n * 5 + 1):])
    if len(d) < n + 1:
        return 50.0
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    au, ad = up[:n].mean(), dn[:n].mean()
    for i in range(n, len(d)):
        au = (au * (n - 1) + up[i]) / n
        ad = (ad * (n - 1) + dn[i]) / n
    return 100.0 if ad == 0 else float(100 - 100 / (1 + au / ad))


# ---------------------------------------------------------------- trend
def trend_factor(s: dict) -> tuple[float, dict]:
    c = s["c"]
    if len(c) < 55:
        return 0.0, {}
    p, s20, s50, s20_prev = c[-1], c[-20:].mean(), c[-50:].mean(), c[-25:-5].mean()
    above = np.tanh((p / s50 - 1) / 0.05)          # price vs 50-day average
    slope = np.tanh((s20 / s20_prev - 1) / 0.01)   # is the 20-day average rising?
    stack = np.tanh((s20 / s50 - 1) / 0.03)        # short-term average above long-term?
    r = rsi(c)
    stretch = -0.3 if r > 75 else (0.15 if r < 25 else 0.0)   # overbought / oversold
    f = float(np.clip((above + slope + stack) / 3 + stretch, -1, 1))
    return f, {"sma20": round(float(s20), 2), "sma50": round(float(s50), 2), "rsi": round(r, 1)}


# ---------------------------------------------------- support / resistance
def swing_points(h, l, w: int = 3) -> tuple[list[int], list[int]]:
    highs, lows = [], []
    for i in range(w, len(h) - w):
        if h[i] == h[i - w:i + w + 1].max():
            highs.append(i)
        if l[i] == l[i - w:i + w + 1].min():
            lows.append(i)
    return highs, lows


def sr_levels(h, l, c, lookback: int = 120, w: int = 3) -> tuple[list[dict], float]:
    """Price levels where the stock repeatedly turned. Swing highs and lows
    within half an ATR of each other are merged; more touches = stronger."""
    h, l, c = h[-lookback:], l[-lookback:], c[-lookback:]
    a = atr(h, l, c)
    hi, lo = swing_points(h, l, w)
    pts = sorted([h[i] for i in hi] + [l[i] for i in lo])
    clusters: list[list[float]] = []
    for p in pts:
        if clusters and abs(p - np.mean(clusters[-1])) <= 0.5 * a:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [{"price": float(np.mean(cl)), "touches": len(cl)} for cl in clusters], a


def levels_factor(s: dict) -> tuple[float, dict]:
    h, l, c, v = s["h"], s["l"], s["c"], s["v"]
    if len(c) < 60:
        return 0.0, {}
    levels, a = sr_levels(h, l, c)
    p = c[-1]
    strong = [x for x in levels if x["touches"] >= 2] or levels
    above = [x["price"] for x in strong if x["price"] > p * 1.001]
    below = [x["price"] for x in strong if x["price"] < p * 0.999]
    R = min(above) if above else None
    S = max(below) if below else None
    up = min((R - p) / a, 4.0) if R else 4.0      # ATRs of room before resistance
    down = min((p - S) / a, 4.0) if S else 4.0    # ATRs above support
    f = (up - down) / (up + down) if up + down > 0 else 0.0   # near support +, near resistance -
    vol_ratio = v[-5:].mean() / max(v[-25:-5].mean(), 1.0)
    event = None
    for x in strong:
        lv = x["price"]
        if c[-1] > lv * 1.005 and c[-6] < lv and vol_ratio > 1.15:
            f, event = max(f, 0.6), f"breakout above {lv:.2f}"
        elif c[-1] < lv * 0.995 and c[-6] > lv and vol_ratio > 1.15:
            f, event = min(f, -0.6), f"broke below {lv:.2f}"
    return float(np.clip(f, -1, 1)), {"support": round(S, 2) if S else None,
                                      "resistance": round(R, 2) if R else None,
                                      "atr": round(a, 2), "event": event}


# ------------------------------------------------------ order-flow proxy
def flow_factor(s: dict, intraday: list[dict] | None = None, price: float | None = None) -> tuple[float, dict]:
    """True order flow (the order book, every trade) needs paid data. This is
    a proxy built from bars: where each bar closed within its range, weighted
    by volume (Chaikin money flow), unusual volume in the direction of the
    move, and price vs today's volume-weighted average price (VWAP)."""
    h, l, c, v = s["h"], s["l"], s["c"], s["v"]
    if len(c) < 25:
        return 0.0, {}
    n = 20
    rng = np.maximum(h[-n:] - l[-n:], 1e-9)
    mfm = ((c[-n:] - l[-n:]) - (h[-n:] - c[-n:])) / rng
    cmf = float((mfm * v[-n:]).sum() / max(v[-n:].sum(), 1.0))
    pressure = np.tanh(cmf * 4)
    rv = float(v[-3:].mean() / max(v[-23:-3].mean(), 1.0))
    unusual = float(np.sign(c[-1] - c[-4]) * min(max(rv - 1, 0), 1.5) / 1.5)
    parts = [(0.5, pressure), (0.25, unusual)]
    vwap = None
    if intraday:
        day = intraday[-1]["t"][:10]
        today = [b for b in intraday if b["t"][:10] == day and b.get("v")]
        vol = sum(b["v"] for b in today)
        if vol > 0:
            vwap = sum(b.get("vw", b["c"]) * b["v"] for b in today) / vol
            px = price or today[-1]["c"]
            parts.append((0.25, float(np.tanh((px / vwap - 1) / 0.005))))
    f = sum(w * x for w, x in parts) / sum(w for w, _ in parts)
    return float(np.clip(f, -1, 1)), {"money_flow": round(cmf, 3), "rel_volume": round(rv, 2),
                                      "vwap": round(vwap, 2) if vwap else None}


# ---------------------------------------------------- pattern recognition
def pattern_features(c, i: int):
    """Shape of the last PATTERN_LEN days ending at bar i, scaled by the
    stock's own volatility so different stocks are comparable."""
    if i < PATTERN_LEN + VOL_LOOKBACK:
        return None, None
    r = np.diff(np.log(c[i - PATTERN_LEN - VOL_LOOKBACK:i + 1]))
    vol = r[:VOL_LOOKBACK].std()
    if not vol:
        return None, None
    path = np.cumsum(r[-PATTERN_LEN:] / vol)
    shape = (path - path.mean()) / (path.std() + 1e-9)
    return np.append(shape * 0.5, path[-1] / np.sqrt(PATTERN_LEN)), vol


def build_library(series: dict[str, dict]) -> dict:
    """Every past window in every watchlist stock, with what happened next."""
    X, y, yraw, dend, sym, idx = [], [], [], [], [], []
    for s_name, s in series.items():
        c, t = s["c"], s["t"]
        for i in range(PATTERN_LEN + VOL_LOOKBACK, len(c) - PATTERN_HORIZON):
            x, vol = pattern_features(c, i)
            if x is None:
                continue
            ret = np.log(c[i + PATTERN_HORIZON] / c[i])
            X.append(x); y.append(ret / (vol * np.sqrt(PATTERN_HORIZON))); yraw.append(np.expm1(ret))
            dend.append(t[i + PATTERN_HORIZON]); sym.append(s_name); idx.append(i)
    return {"X": np.array(X), "y": np.array(y), "yraw": np.array(yraw),
            "dend": np.array(dend, dtype="datetime64[D]"), "sym": np.array(sym), "i": np.array(idx)}


def pattern_factor(lib: dict, c, i: int, before, k: int = 50) -> tuple[float, dict]:
    """Find the k most similar past shapes whose outcome was already known
    before `before`, and score by how those played out."""
    x, _ = pattern_features(c, i)
    if x is None or len(lib["X"]) == 0:
        return 0.0, {}
    mask = lib["dend"] < np.datetime64(before, "D")
    if mask.sum() < k * 4:
        return 0.0, {}
    X, y, yraw = lib["X"][mask], lib["y"][mask], lib["yraw"][mask]
    d = ((X - x) ** 2).sum(1)
    nn = np.argpartition(d, k)[:k]
    mean_z = float(y[nn].mean())
    return float(np.tanh(mean_z * 2)), {"win_rate": round(float((yraw[nn] > 0).mean()), 3),
                                        "avg_return": round(float(yraw[nn].mean()), 4), "matches": k}


TECHNICAL = ["trend", "levels", "flow", "pattern"]
FACTORS = ["news"] + TECHNICAL


def combine(factors: dict[str, float], weights: dict[str, float]) -> float:
    tot = sum(weights.get(f, 0) for f in factors)
    if tot <= 0:
        return 0.0
    return float(sum(weights.get(f, 0) * v for f, v in factors.items()) / tot)
