"""One-off read-only check of the options data endpoints (no orders)."""
import datetime as dt
from common import Alpaca
import options_agent, yaml
a = Alpaca(); o = yaml.safe_load(open("../config.yaml"))["books"]["options"]
today = dt.date.today()
for u, px, kind in [("SPY", None, "call"), ("AMD", None, "put")]:
    px = a.bars([u], "1Day", dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=7))[u][-1]["c"]
    cs = a.option_contracts(u, kind, today + dt.timedelta(days=21), today + dt.timedelta(days=45), px * 0.9, px * 1.1)
    cs.sort(key=lambda c: (abs((dt.date.fromisoformat(c["expiration_date"]) - today).days - 30), abs(float(c["strike_price"]) - px)))
    snaps = a.option_snapshots([c["symbol"] for c in cs[:40]])
    s0 = next(iter(snaps.values()), {})
    print(u, kind, "price", px, "contracts", len(cs), "snapshots", len(snaps), "sample keys", list(s0.keys()), "greeks", s0.get("greeks"), "quote", s0.get("latestQuote"))
    print("  first contract fields:", {k: cs[0].get(k) for k in ("symbol", "expiration_date", "strike_price", "open_interest", "type")} if cs else None)
    print("  pick:", options_agent.choose_contract(cs, snaps, kind, px, today, o))
