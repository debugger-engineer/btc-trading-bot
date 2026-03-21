## Overview

Automated BTC perpetuals trading bot. Uses a Bollinger Band mean-reversion strategy with an EMA trend filter. All orders are executed on Hyperliquid, a decentralized perpetuals exchange, so funds stay in a self-custodied wallet at all times. 

## Strategy

Bollinger Bands (20, 2) on 5-minute BTC/USDT candles from Binance. EMA(200) as a trend filter.

Entry conditions (only one position at a time):

- LONG: price touches or crosses below the lower band AND price is above EMA(200)
- SHORT: price touches or crosses above the upper band AND price is below EMA(200)

Entry is placed as a GTC limit order at the band level. The order is cancelled after 60 seconds or if price moves away from the band.

Exit conditions:

- Take profit: price reaches the opposite band (resting limit order, maker fee)
- Stop loss: 2.5% from entry price (stop-market order)
- Break-even: if the take-profit band drifts past the entry price (inverted TP), the exit is moved to entry price

Indicators refresh every 5 minutes. The take-profit limit order is updated on each refresh.

Backtest results (5m, 365 days, BB(20,2) + EMA(200) + 2% SL): ~88.8% annually, 69% win rate, 872 trades.

## Position Sizing

Position size is derived from account balance, leverage, and a target risk per trade:

```
capital_pct = target_risk_pct / (leverage * stop_loss_pct / 100)
position_notional = available_margin * capital_pct * leverage
```

Default config: 8% target risk, 2.5% stop loss, 5x leverage → ~64% of margin per trade.

## Configuration

`config/accounts.json` controls global strategy parameters and named account profiles:

```json
{
  "target_risk_pct": 8.0,
  "stop_loss_pct": 2.5,
  "bb_period": 20,
  "bb_std": 2.0,
  "ema_period": 200,
  "perps_live": true,
  "accounts": [
    { "name": "bot-5x", "leverage": 5, "active": true }
  ]
}
```

Set `perps_live` to `false` to run in dry-run mode (logs signals without placing orders). Set `active` to `false` on an account to skip it on startup.

## Environment Variables

Copy `.env.example` to `.env` and fill in your credentials.

## Database

PostgreSQL is required in live mode. The `perps_trades` table is created automatically and stores one row per trade, including entry/exit prices, BB values at entry and exit, leverage, gross PnL, fees, and net PnL.

## How to Run Locally

1. Create and activate a virtual environment:
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Run:
   ```sh
   python src/main.py
   ```

## How to Run with Docker

1. Build the image:
   ```sh
   docker build -t trading-bot .
   ```

2. Run in the foreground:
   ```sh
   docker run --rm --env-file .env trading-bot
   ```

3. Run in the background:
   ```sh
   docker run -d --name trading-bot-instance --restart always --env-file .env trading-bot
   ```

4. View logs:
   ```sh
   docker logs -f trading-bot-instance
   ```

5. Stop:
   ```sh
   docker stop trading-bot-instance
   ```

## Backtesting

Runs a simulation against historical Binance klines. No exchange credentials or DB required.

```sh
python test/backtest_perps.py
```

The first entry is always the current live strategy (BB(20,2) + EMA(200) + SL=2.5%, 5x leverage) and should not be removed. Results are printed to stdout and saved to `test/backtest_perps_results_btc.txt`.
