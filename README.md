# Trading Agents

Two cooperating agents that run on their own, on GitHub's servers, with no laptop needed:

- **News agent** (`agents/news_agent.py`) reads the last 24 hours of news for your watchlist from Alpaca's news feed, has Claude judge how each story is likely to move each stock, and writes a score per symbol (-1 to +1) plus an overall market-risk rating to `data/signals.json`.
- **Trading agent** (`agents/trader_agent.py`) reads those signals, applies your risk rules, places orders through Alpaca, and records P&L.
- **Dashboard** (`index.html`) shows account value, today's and total P&L, positions, trades, and the headlines behind each signal. It updates after every run.

It runs hourly on weekdays during market hours. **It uses Alpaca paper trading (simulated money) by default.**

## How the agents talk to each other

```
 hourly (GitHub Actions)
   │
   ├─ News agent ──► data/signals.json ──► Trading agent ──► Alpaca orders
   │                 (score per symbol,                         │
   │                  market risk, headlines)                   ▼
   └────────────────────────────── data/status.json, equity.json, trades.json ──► Dashboard
```

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
- **Risk rules:** stop loss, take profit, position size, cash reserve, daily loss limit, and max positions are all in `config.yaml`.

## Trading rules (defaults)

Sell a held position if it is down 5% (stop loss), up 10% (take profit), or the news score drops to -0.30 or below.

Buy a symbol when its news score is +0.45 or higher from at least 2 articles, up to 6 positions, with at most 10% of the account per position and at least 20% kept in cash.

New buys pause automatically when: the account is down 3% today, the news agent rates market risk above 0.70, the news signals are more than 3 hours old, or the kill switch is on. Sells keep working while buys are paused.

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
