"""
bot.py
Main trading loop for the confluence strategy on Angel One (SmartAPI).

Requires: pip install smartapi-python pyotp pandas --break-system-packages

Run:
    python bot.py            # uses dry_run flag from config.json
    python bot.py --live     # forces live order placement (overrides dry_run)
"""

import json
import time
import argparse
import logging
import os
import threading
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd
import pyotp
import numpy as np # import here to support mock fallback
from SmartApi import SmartConnect

from indicators import build_all_indicators

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("algobot")


# ────────────────────────────────────────────────────────
# Health Check Server (Enables Render 100% Free Web Service)
# ────────────────────────────────────────────────────────
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK - Bot is active and running.")
        
    def log_message(self, format, *args):
        return  # Suppress connection logging to keep logs clean

def start_health_server():
    port = int(os.environ.get("PORT", 8000))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    log.info("Starting background health check server on port %d", port)
    server.serve_forever()


class AngelOneClient:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        # Read from environment variables if set (useful for Render), fallback to config.json
        self.client_id = os.environ.get("ANGEL_CLIENT_ID", cfg.get("client_id", "YOUR_CLIENT_ID"))
        self.password = os.environ.get("ANGEL_PASSWORD", cfg.get("password", "YOUR_PASSWORD"))
        self.api_key = os.environ.get("ANGEL_API_KEY", cfg.get("api_key", "YOUR_API_KEY"))
        self.totp_secret = os.environ.get("ANGEL_TOTP_SECRET", cfg.get("totp_secret", "YOUR_TOTP_SECRET"))
        
        self.smart = SmartConnect(api_key=self.api_key)
        self.session = None

    def login(self):
        # Handle cases where credentials are not configured yet
        if self.client_id == "YOUR_CLIENT_ID":
            log.warning("Please configure your config.json or environment variables with your actual Angel One API credentials!")
            return False
            
        try:
            totp = pyotp.TOTP(self.totp_secret).now()
            data = self.smart.generateSession(self.client_id, self.password, totp)
            if not data.get("status"):
                raise RuntimeError(f"Angel One login failed: {data}")
            self.session = data
            log.info("Logged in to Angel One as %s", self.client_id)
            return True
        except Exception as e:
            log.error("Failed to login to Angel One: %s", str(e))
            return False

    def get_equity(self) -> float:
        if not self.session:
            return 100000.0 # Default virtual balance for dry-runs
        try:
            rms = self.smart.rmsLimit()
            if rms.get("status") and "data" in rms and "net" in rms["data"]:
                return float(rms["data"]["net"])
        except Exception as e:
            log.warning("Could not read RMS limit, defaulting to 100,000: %s", str(e))
        return 100000.0

    def get_candles(self, symbol_token: str, exchange: str, interval: str,
                     from_dt: datetime, to_dt: datetime) -> pd.DataFrame:
        if not self.session:
            # Generate mock historical candles for dry-run testing if not connected
            dates = pd.date_range(end=to_dt, periods=100, freq="3min" if interval == "THREE_MINUTE" else "15min")
            prices = np.random.randn(100).cumsum() + 1000.0
            return pd.DataFrame({
                "timestamp": dates,
                "open": prices - 2,
                "high": prices + 5,
                "low": prices - 5,
                "close": prices,
                "volume": 1000
            })

        params = {
            "exchange": exchange,
            "symboltoken": symbol_token,
            "interval": interval,
            "fromdate": from_dt.strftime("%Y-%m-%d %H:%M"),
            "todate": to_dt.strftime("%Y-%m-%d %H:%M"),
        }
        resp = self.smart.getCandleData(params)
        if not resp.get("status"):
            raise RuntimeError(f"Candle fetch failed: {resp.get('message', resp)}")
        
        if resp.get("data") is None:
            return pd.DataFrame()
            
        df = pd.DataFrame(resp["data"], columns=["timestamp", "open", "high", "low", "close", "volume"])
        if df.empty:
            return df
            
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # Ensure values are float
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        return df

    def place_bracket_order(self, symbol: str, symbol_token: str, exchange: str,
                             qty: int, side: str, entry_price: float,
                             stop_loss: float, target: float, dry_run: bool):
        if dry_run or not self.session:
            log.info(
                "[DRY-RUN] %s %s qty=%d entry=%.2f SL=%.2f TP=%.2f",
                side, symbol, qty, entry_price, stop_loss, target,
            )
            return {"status": True, "dry_run": True}

        order = {
            "variety": "ROBO",
            "tradingsymbol": symbol,
            "symboltoken": symbol_token,
            "transactiontype": side,  # "BUY" or "SELL"
            "exchange": exchange,
            "ordertype": "LIMIT",
            "producttype": "BO",
            "duration": "DAY",
            "price": str(entry_price),
            "squareoff": str(round(abs(target - entry_price), 2)),
            "stoploss": str(round(abs(entry_price - stop_loss), 2)),
            "quantity": str(qty),
        }
        try:
            resp = self.smart.placeOrder(order)
            log.info("Order placed: %s -> %s", order, resp)
            return resp
        except Exception as e:
            log.error("Failed to place bracket order: %s", str(e))
            return {"status": False, "message": str(e)}


class ConfluenceStrategy:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def check_confluence(self, ltf_df: pd.DataFrame, htf_df: pd.DataFrame) -> dict:
        """
        Returns dict: {'signal': 'BUY'|'SELL'|None, 'entry':, 'stop_loss':, 'reason':}
        Applies: trend alignment, HMA slope + Supertrend + DTM breakout,
        wick/squeeze/pivot-distance trap filters, base EMA slope.
        """
        ltf = build_all_indicators(ltf_df)
        htf = build_all_indicators(htf_df)

        last = ltf.iloc[-1]
        htf_last = htf.iloc[-1]

        htf_trend = htf_last["dtm_trend"]  # 1 bullish, -1 bearish
        ltf_trend = last["dtm_trend"]

        if htf_trend != ltf_trend:
            return {"signal": None, "reason": "HTF/LTF trend mismatch"}

        st_dir = last["st_direction"]
        hma_slope = last["hma_slope_pct"]

        bullish = (ltf_trend == 1 and st_dir == 1 and hma_slope > 0)
        bearish = (ltf_trend == -1 and st_dir == -1 and hma_slope < 0)

        if not (bullish or bearish):
            return {"signal": None, "reason": "No directional confluence"}

        # Trap filters
        if last["upper_wick_pct"] > self.cfg["wick_ratio_max"] * 100 or \
           last["lower_wick_pct"] > self.cfg["wick_ratio_max"] * 100:
            return {"signal": None, "reason": "Wick trap filter"}

        if last["dtm_band_width_pct"] < self.cfg["band_squeeze_min_pct"]:
            return {"signal": None, "reason": "Band squeeze too tight"}

        dist_pct = last["dist_to_pivot_high_pct"] if bullish else last["dist_to_pivot_low_pct"]
        if pd.isna(dist_pct) or dist_pct < self.cfg["pivot_distance_min_pct"]:
            return {"signal": None, "reason": "Too close to pivot wall"}

        if bullish and last["base_ema_slope_pct"] <= 0:
            return {"signal": None, "reason": "Base EMA slope not confirming"}
        if bearish and last["base_ema_slope_pct"] >= 0:
            return {"signal": None, "reason": "Base EMA slope not confirming"}

        entry = last["close"]
        if bullish:
            stop_loss = last["last_pivot_low"] if not pd.isna(last["last_pivot_low"]) else last["hma_50"]
            signal = "BUY"
        else:
            stop_loss = last["last_pivot_high"] if not pd.isna(last["last_pivot_high"]) else last["hma_50"]
            signal = "SELL"

        return {"signal": signal, "entry": entry, "stop_loss": stop_loss, "reason": "confluence confirmed"}


def position_size(equity: float, risk_pct: float, entry: float, stop_loss: float) -> int:
    risk_amount = equity * (risk_pct / 100)
    per_share_risk = math.abs(entry - stop_loss)
    if per_share_risk <= 0:
        return 0
    return int(risk_amount / per_share_risk)


def load_config(path: str = "config.json") -> dict:
    if not os.path.exists(path):
        # Return fallback configuration dictionary if file doesn't exist (cloud setup)
        return {
            "angel_one": {},
            "trading": {
                "stocks": [
                    "WIPRO",
                    "COALINDIA",
                    "PETRONET",
                    "RAILTEL",
                    "BIOCON",
                    "MOIL",
                    "DLINKINDIA",
                    "JYOTHYLAB",
                    "TCS",
                    "INFY"
                ],
                "ltf": "THREE_MINUTE",
                "htf": "FIFTEEN_MINUTE",
                "wick_ratio_max": 0.38,
                "band_squeeze_min_pct": 0.38,
                "pivot_distance_min_pct": 0.38,
                "risk_per_trade_pct": 0.33,
                "daily_max_loss_pct": 1.0,
                "reward_risk_ratio": 1.5,
                "dry_run": True
            }
        }
    with open(path) as f:
        return json.load(f)


def run(force_live: bool = False):
    # Start the local health web server in a daemon thread
    threading.Thread(target=start_health_server, daemon=True).start()

    cfg = load_config()
    trading_cfg = cfg["trading"]
    dry_run = trading_cfg["dry_run"] and not force_live

    client = AngelOneClient(cfg["angel_one"])
    connected = client.login()
    if not connected:
        log.warning("Running bot in Virtual DRY-RUN mode.")

    strategy = ConfluenceStrategy(trading_cfg)

    equity = client.get_equity()
    log.info("Account equity: %.2f | dry_run=%s", equity, dry_run)

    daily_max_loss = equity * (trading_cfg["daily_max_loss_pct"] / 100)
    daily_pnl = 0.0
    risk_per_trade_pct = trading_cfg["risk_per_trade_pct"]
    rr_ratio = trading_cfg["reward_risk_ratio"]

    # Official Angel One Exchange tokens for NSE Cash segment (Strictly Shariah Compliant List)
    symbol_tokens = {
        "WIPRO": "3721",
        "COALINDIA": "20396",
        "PETRONET": "11359",
        "RAILTEL": "12316",
        "BIOCON": "11373",
        "MOIL": "17937",
        "DLINKINDIA": "16075",
        "JYOTHYLAB": "15012",
        "TCS": "11536",
        "INFY": "1594"
    }

    # Track virtual positions for dry-run simulation
    active_dry_run_trades = []

    while True:
        # Check active dry-run positions to simulate exits and update PnL
        if dry_run and active_dry_run_trades:
            still_active = []
            for trade in active_dry_run_trades:
                # Retrieve current price for the token
                try:
                    now = datetime.now()
                    candles = client.get_candles(trade["token"], "NSE", "THREE_MINUTE", now - timedelta(minutes=15), now)
                    if not candles.empty:
                        current_price = candles.iloc[-1]["close"]
                        
                        exited = False
                        pnl = 0.0
                        if trade["side"] == "BUY":
                            if current_price <= trade["sl"]:
                                exited = True
                                pnl = (trade["sl"] - trade["entry"]) * trade["qty"]
                            elif current_price >= trade["tp"]:
                                exited = True
                                pnl = (trade["tp"] - trade["entry"]) * trade["qty"]
                        else:
                            if current_price >= trade["sl"]:
                                exited = True
                                pnl = (trade["entry"] - trade["sl"]) * trade["qty"]
                            elif current_price <= trade["tp"]:
                                exited = True
                                pnl = (trade["entry"] - trade["tp"]) * trade["qty"]
                                
                        if exited:
                            daily_pnl += pnl
                            log.info("[DRY-RUN EXIT] %s PnL: %.2f (Total Daily PnL: %.2f)", trade["symbol"], pnl, daily_pnl)
                        else:
                            still_active.append(trade)
                    else:
                        still_active.append(trade)
                except Exception as e:
                    still_active.append(trade)
            active_dry_run_trades = still_active

        # Stop trading if Daily Loss Limit is hit
        if daily_pnl <= -daily_max_loss:
            log.warning("Daily max loss limit hit (-%.2f). Halting bot for the session.", daily_max_loss)
            break

        for symbol in trading_cfg["stocks"]:
            token = symbol_tokens.get(symbol)
            if not token:
                log.warning("No symbol token configured for %s, skipping.", symbol)
                continue

            try:
                now = datetime.now()
                
                # Sleep 0.5s before ltf call to stay safely below 3 requests/sec rate limit
                time.sleep(0.5)
                ltf_df = client.get_candles(token, "NSE", trading_cfg["ltf"], now - timedelta(hours=8), now)
                
                # Sleep 0.5s before htf call
                time.sleep(0.5)
                htf_df = client.get_candles(token, "NSE", trading_cfg["htf"], now - timedelta(days=3), now)

                # Validate data exists and contains enough rows for calculations (WMA 50 needs at least 50+ rows)
                if ltf_df.empty or htf_df.empty or len(ltf_df) < 55 or len(htf_df) < 55:
                    log.warning("%s: Not enough candles found (LTF: %d/55, HTF: %d/55). Skipping.", 
                                symbol, len(ltf_df), len(htf_df))
                    continue

                result = strategy.check_confluence(ltf_df, htf_df)
                if not result["signal"]:
                    log.debug("%s: no signal (%s)", symbol, result["reason"])
                    continue

                entry, stop = result["entry"], result["stop_loss"]
                qty = position_size(equity, risk_per_trade_pct, entry, stop)
                if qty <= 0:
                    continue

                target = entry + rr_ratio * (entry - stop) if result["signal"] == "BUY" \
                    else entry - rr_ratio * (stop - entry)

                # Execute
                resp = client.place_bracket_order(
                    symbol=symbol, symbol_token=token, exchange="NSE",
                    qty=qty, side=result["signal"], entry_price=entry,
                    stop_loss=stop, target=target, dry_run=dry_run,
                )
                
                if dry_run and resp.get("status"):
                    active_dry_run_trades.append({
                        "symbol": symbol,
                        "token": token,
                        "side": result["signal"],
                        "entry": entry,
                        "sl": stop,
                        "tp": target,
                        "qty": qty
                    })

            except Exception as e:
                log.error("Error processing %s: %s", symbol, str(e))

        time.sleep(180)  # poll every 3 minutes (matches LTF)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Force live trading, overriding dry_run in config")
    args = parser.parse_args()
    run(force_live=args.live)
import math # import here to support position_size
