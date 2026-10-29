# Trading Agents

Three cooperating agents that run on their own, on GitHub's servers, with no laptop needed:

- **News agent** (`agents/news_agent.py`) reads the last 24 hours of news for your watchlist and has Claude judge how each story is likely to move each stock.
- **Chart agent** (`agents/chart_agent.py`) reads 3 years of daily prices plus today's 15-minute bars and scores four things per stock: **trend**, **support/resistance**, an **order-flow proxy** (buying vs selling pressure, unusual volume, price vs VWAP) and **pattern matching** (how the most similar past chart shapes played out). It combines these with the news into one weighted score. Once a week it **re-learns** the weights by replaying history and checking which factors actually predicted the next week's move.
- **Trading agent** (`agents/trader_agent.py`) buys on strong combined scores, keeps a **4% trailing stop at the broker** on every position, and records P&L.
- **Dashboard** (`index.html`) shows P&L, positions with their trailing stops, every stock's factor scores, and what the learning step found, including an honest out-of-sample check.

It runs hourly on weekdays during market hours (cron-job.org triggers it; GitHub's own schedule is a backup). **It uses Alpaca paper trading (simulated money) by default.**

```
 hourly
   ├─ News agent  ──► signals.json ─┐
   ├─ Chart agent ──► analysis.json ├─► Trading agent ──► Alpaca orders + trailing stops
   │   (weekly) learn ► weights.json┘                         │
   └──────────────── status.json, equity.json, trades.json ───┴─► Dashboard
```

## Strategy (swing trading: days to ~2 weeks)

**Buy** when the combined score clears the bar (auto: the top ~15% of historical scores) *and* at least 3 of the 5 factors are positive *and* the news isn't clearly bad. Whole shares, max 10% of the account each, max 6 positions, 20% always in cash.

**Sell** when any of these happen:
- the price falls 4% below its high since buying (trailing stop, enforced by Alpaca continuously, so gains are locked in as the stock rises)
- it's down 5% from the purchase price (backup hard stop)
- the news turns clearly negative, or the combined score falls into the bottom ~10%
- 14 days have passed and there's no fresh buy signal

**Re-entry:** after selling, a stock can be bought again once 20 hours have passed and the combined score clears the bar again.

**Pauses** new buys (sells and stops keep working) when the account is down 3% on the day, Claude flags high market risk, data is stale, Claude can't read the news, or the kill switch is on.

**About "order flow":** true order flow (the order book and every individual trade) needs paid market data. This uses a proxy built from price bars, which captures buying/selling pressure but not hidden orders or the book itself.

**About learning:** the dashboard shows, for each factor, how often stocks rose over the next 5 trading days when that factor was bullish vs on any random day. Weights move toward factors that beat the normal rate. Many technical signals turn out to be close to coin flips on large, heavily traded stocks; the out-of-sample line tells you whether the combined score has had a real edge recently.

## Setup (about 20 minutes, once)

### 1. Alpaca paper account
1. Sign up at alpaca.markets (free).
2. Switch to **Paper Trading** in the dashboard and generate an **API key and secret**. Copy both.

### 2. Anthropic API key (recommended)
Get one at console.anthropic.com. This lets Claude read the news. Without it, the news agent falls back to simple keyword matching, which is much cruder. Expected cost with the default `claude-haiku-5-5` model is small (a few cents to low dollars a month for a 10-symbol watchlist), but check current pricing.

### 3. Put the code on GitHub
1. Create a new repository on github.com.
   - **Public** repo: GitHub Pages (the dashboard) is free. Your API keys stay private as secrets, but anyone with the link could see your paper P&L and trades.
   - **Private** repo: Pages needs a paid GitHub plan.
2. Upload every file in this folder, including the hidden `.github` folder (drag the folder into "Add file → Upload files", or use `git push`).

### 4. Add your keys as secrets
Repo → **Settings → Secrets and variables → Actions → New repository secret**:

| Name | Value |
|---|---|
| `ALPACA_API_KEY` | your Alpaca paper key |
| `ALPACA_SECRET_KEY` | your Alpaca paper secret |
| `ANTHROPIC_API_KEY` | your Anthropic key (optional) |

### 5. Turn on the dashboard
Repo → **Settings → Pages** → Source: *Deploy from a branch* → Branch `main`, folder `/ (root)` → Save.
Your dashboard will be at `https://<your-username>.github.io/<repo-name>/`. Bookmark it on your phone.

### 6. First run
Repo → **Actions** tab → enable workflows if asked → **trading-agents** → **Run workflow**. After it finishes (about a minute), refresh the dashboard. From then on it runs by itself.

## Controlling it from your phone

Edit `config.yaml` on github.com (pencil icon) or in the GitHub mobile app. Changes apply on the next run.

- **Emergency stop:** set `kill_switch: true`. Add `liquidate_on_kill: true` to also sell all watchlist positions.
  Alternatively, set a repository *variable* (Settings → Secrets and variables → Actions → Variables) named `KILL_SWITCH` to `1`.
- **Pause everything:** Actions tab → trading-agents → "…" → Disable workflow.
- **Watchlist:** the example list is just a starting point. The trader only ever touches symbols on this list.
- **Risk rules:** trailing stop size, hard stop, max hold, buy bar, position size, cash reserve and daily loss limit are all in `config.yaml`.

## Before you consider real money

Read this part. News-sentiment trading is hard: by the time a story reaches a feed, the price has usually already moved, and most retail trading bots underperform simply holding an index fund. Treat the paper period as a test:

1. Run on paper for at least 2–3 months.
2. Compare the "Since start" number against SPY over the same period. If it doesn't clearly beat it after costs, the strategy isn't working.
3. Real-money accounts have rules paper doesn't enforce — notably the pattern day trader rule for margin accounts under $25,000, and taxes on every realized gain.

The code only trades live if you deliberately set a repository variable `LIVE_TRADING` to `i-understand-the-risk` **and** swap your secrets to live Alpaca keys. This is not financial advice; you are responsible for every trade it makes.

## Running tests locally

```
pip install -r requirements.txt pytest
pytest tests
```

## Notes

- GitHub's scheduler can run a few minutes late during busy periods; that's normal.
- GitHub disables scheduled workflows in repos with no activity for 60 days. The bot's own commits count as activity, so this shouldn't happen while it runs.
- All history lives in the `data/` folder as plain JSON, so you can download or analyze it any time.
