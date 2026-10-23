"""News agent: reads recent news for the watchlist, scores how each story is
likely to move each stock, and writes data/signals.json for the trader.

Uses Claude to read the news when ANTHROPIC_API_KEY is set; otherwise falls
back to a simple keyword scorer so the system still runs."""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys

from common import Alpaca, load_config, now_utc, parse_time, read_json, write_json

POSITIVE = ["beat", "beats", "surge", "soar", "jump", "rally", "upgrade", "record", "raises guidance",
            "raised guidance", "strong demand", "outperform", "buyback", "approval", "partnership",
            "exceeds", "tops estimates", "profit rises", "growth accelerates"]
NEGATIVE = ["miss", "misses", "plunge", "tumble", "slump", "downgrade", "lawsuit", "recall", "probe",
            "investigation", "cuts guidance", "lowered guidance", "layoffs", "antitrust", "fine",
            "underperform", "warns", "weak demand", "bankruptcy", "fraud", "sell-off"]
MACRO_RISK = ["recession", "crash", "rate hike", "inflation surges", "bank failure", "war", "default",
              "shutdown", "tariff", "sell-off", "panic"]

SYSTEM = """You are a financial news analyst feeding an automated paper-trading system.
For each article, judge its likely short-term (1-5 trading days) price impact on each
watchlist ticker it concerns. Be conservative: routine or already-priced-in news is near 0.
Also rate market_risk (0 = calm, 1 = severe market-wide danger such as a crash or crisis).
Respond with ONLY a JSON array, one object per article:
[{"id": "<article id>", "market_risk": 0.0,
  "impacts": [{"ticker": "AAPL", "sentiment": -1.0 to 1.0, "confidence": 0.0 to 1.0,
               "reason": "under 15 words"}]}]
Only include tickers from the watchlist. Use an empty impacts list if none apply."""


def keyword_score(article: dict, watchlist: set[str]) -> dict:
    text = f"{article.get('headline', '')} {article.get('summary', '')}".lower()
    pos = sum(1 for w in POSITIVE if w in text)
    neg = sum(1 for w in NEGATIVE if w in text)
    macro = sum(1 for w in MACRO_RISK if w in text)
    # damped so a single keyword can't produce an extreme score
    s = 0.0 if pos + neg == 0 else (pos - neg) / (pos + neg + 2)
    impacts = [{"ticker": t, "sentiment": round(s, 2), "confidence": 0.3, "reason": "keyword match"}
               for t in article.get("symbols", []) if t in watchlist and s != 0]
    return {"id": str(article["id"]), "market_risk": min(1.0, 0.2 * macro), "impacts": impacts}


def claude_score(batch: list[dict], watchlist: set[str], model: str) -> list[dict]:
    import anthropic
    client = anthropic.Anthropic()
    items = [{"id": str(a["id"]), "headline": a.get("headline", ""), "summary": (a.get("summary") or "")[:600],
              "tagged_symbols": [s for s in a.get("symbols", []) if s in watchlist],
              "published": a.get("created_at")} for a in batch]
    msg = client.messages.create(
        model=model, max_tokens=3000, system=SYSTEM,
        messages=[{"role": "user", "content": f"Watchlist: {sorted(watchlist)}\n\nArticles:\n{json.dumps(items)}"}])
    text = "".join(b.text for b in msg.content if b.type == "text")
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError("Claude returned no JSON array")
    results = json.loads(m.group(0))
    clean = []
    for r in results:
        impacts = []
        for i in r.get("impacts", []):
            t = str(i.get("ticker", "")).upper()
            if t in watchlist:
                impacts.append({"ticker": t,
                                "sentiment": max(-1.0, min(1.0, float(i.get("sentiment", 0)))),
                                "confidence": max(0.0, min(1.0, float(i.get("confidence", 0)))),
                                "reason": str(i.get("reason", ""))[:120]})
        clean.append({"id": str(r.get("id")), "market_risk": max(0.0, min(1.0, float(r.get("market_risk", 0)))),
                      "impacts": impacts})
    return clean


def aggregate(articles: list[dict], watchlist: list[str], now: dt.datetime, half_life_h: float,
              neutral_cutoff: float = 0.0) -> dict:
    """Combine scored articles into one score per ticker, weighting recent and
    high-confidence stories more. Stories Claude rated near-neutral (below
    neutral_cutoff) are left out so routine news doesn't water down real
    signals. Pure function (unit-tested)."""
    acc = {t: {"num": 0.0, "den": 0.0, "count": 0, "ai": 0, "headlines": []} for t in watchlist}
    risk_num = risk_den = 0.0
    for a in articles:
        age_h = max(0.0, (now - parse_time(a["created_at"])).total_seconds() / 3600)
        w = 0.5 ** (age_h / half_life_h)
        risk_num += w * a.get("market_risk", 0)
        risk_den += w
        for i in a.get("impacts", []):
            t = i["ticker"]
            if t not in acc or abs(i["sentiment"]) < neutral_cutoff:
                continue
            acc[t]["num"] += w * i["confidence"] * i["sentiment"]
            acc[t]["den"] += w * i["confidence"]
            acc[t]["count"] += 1
            if a.get("scorer") == "claude":
                acc[t]["ai"] += 1
            acc[t]["headlines"].append({"headline": a["headline"], "url": a.get("url"), "time": a["created_at"],
                                        "sentiment": i["sentiment"], "reason": i.get("reason", "")})
    tickers = {}
    for t, v in acc.items():
        score = v["num"] / v["den"] if v["den"] > 0 else 0.0
        heads = sorted(v["headlines"], key=lambda h: h["time"], reverse=True)[:5]
        tickers[t] = {"score": round(score, 3), "articles": v["count"], "ai_articles": v["ai"], "headlines": heads}
    return {"generated_at": now.isoformat(),
            "market_risk": round(risk_num / risk_den, 3) if risk_den else 0.0,
            "tickers": tickers}


def main() -> int:
    cfg = load_config()
    ncfg = cfg["news"]
    watchlist = [s.upper() for s in cfg["watchlist"]]
    wl = set(watchlist)
    now = now_utc()
    start = now - dt.timedelta(hours=ncfg["lookback_hours"])

    alpaca = Alpaca()
    raw = alpaca.news(watchlist, start)
    print(f"Fetched {len(raw)} articles")

    use_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))
    store = read_json("articles.json", [])
    if use_claude:  # retry articles that earlier fell back to keyword scoring
        store = [a for a in store if a.get("scorer") == "claude"]
    known = {a["id"] for a in store}
    new = [a for a in raw if str(a["id"]) not in known]
    print(f"{len(new)} articles to read")
    scores: dict[str, dict] = {}
    scorer_of: dict[str, str] = {}
    claude_ok = use_claude
    bs = ncfg.get("batch_size", 12)
    for i in range(0, len(new), bs):
        batch = new[i:i + bs]
        res, used = None, "keywords"
        if claude_ok:
            try:
                res, used = claude_score(batch, wl, ncfg["model"]), "claude"
            except Exception as e:  # never let one bad batch stop the run
                print(f"::warning::Claude could not read the news: {e}", file=sys.stderr)
                if "400" in str(e) or "401" in str(e) or "credit" in str(e).lower():
                    claude_ok = False  # account/key problem: don't keep retrying this run
        if res is None:
            res = [keyword_score(a, wl) for a in batch]
        for r in res:
            scores[r["id"]] = r
        for a in batch:
            scorer_of[str(a["id"])] = used

    for a in new:
        r = scores.get(str(a["id"]), {"market_risk": 0, "impacts": []})
        store.append({"id": str(a["id"]), "created_at": a["created_at"], "headline": a.get("headline", ""),
                      "url": a.get("url"), "source": a.get("source"), "market_risk": r["market_risk"],
                      "impacts": r["impacts"], "scorer": scorer_of.get(str(a["id"]), "keywords")})

    store = [a for a in store if parse_time(a["created_at"]) >= start]
    write_json("articles.json", store)

    signals = aggregate(store, watchlist, now, ncfg["half_life_hours"],
                        ncfg.get("neutral_cutoff", 0.0))
    n_ai = sum(1 for a in store if a.get("scorer") == "claude")
    signals["scorer"] = "claude" if n_ai and n_ai == len(store) else ("mixed" if n_ai else "keywords")
    signals["ai_coverage"] = round(n_ai / len(store), 3) if store else 0.0
    write_json("signals.json", signals)
    top = sorted(signals["tickers"].items(), key=lambda kv: kv[1]["score"], reverse=True)
    print("Signals:", ", ".join(f"{t} {v['score']:+.2f} ({v['articles']})" for t, v in top))
    print(f"Market risk: {signals['market_risk']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
