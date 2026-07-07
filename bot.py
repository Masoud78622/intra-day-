"""
bot.py
Main trading loop for the confluence strategy on Angel One (SmartAPI).
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
import numpy as np
from SmartApi import SmartConnect

from indicators import build_all_indicators

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("algobot")

# Angel One allows max 1 candle request/second. We use 1.5s to be safe.
API_DELAY_SECONDS = 1.5
# Max retries when rate limited
MAX_RETRIES = 4


# ────────────────────────────────────────────────────────
# Health Check Server (Enables Render Free Web Service)
# ────────────────────────────────────────────────────────
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK - Bot is running.")

    def log_message(self, format, *args):
        return


def start_health_server():
    port = int(os.environ.get("PORT", 8000))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    log.info("Health check server started on port %d", port)
    server.serve_forever()


class AngelOneClient:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.client_id = os.environ.get("ANGEL_CLIENT_ID", cfg.get("client_id", "YOUR_CLIENT_ID"))
        self.password = os.environ.get("ANGEL_PASSWORD", cfg.get("password", "YOUR_PASSWORD"))
        self.api_key = os.environ.get("ANGEL_API_KEY", cfg.get("api_key", "YOUR_API_KEY"))
        self.totp_secret = os.environ.get("ANGEL_TOTP_SECRET", cfg.get("totp_secret", "YOUR_TOTP_SECRET"))

        self.smart = SmartConnect(api_key=self.api_key)
        self.session = None

    def login(self):
        if self.client_id == "YOUR_CLIENT_ID":
            log.warning("Credentials not configured - running in DRY-RUN mode.")
            return False
        try:
            totp = pyotp.TOTP(self.totp_secret).now()
            data = self.smart.generateSession(self.client_id, self.password, totp)
            if not data.get("status"):
                raise RuntimeError(f"Login failed: {data}")
            self.session = data
            log.info("Logged in to Angel One as %s", self.client_id)
            return True
        except Exception as e:
            log.error("Login failed: %s", str(e))
            return False

    def get_equity(self) -> float:
        if not self.session:
            return 100000.0
        try:
            rms = self.smart.rmsLimit()
            if rms.get("status") and "data" in rms and "net" in rms["data"]:
                return float(rms["data"]["net"])
        except Exception as e:
            log.warning("Could not read equity: %s", str(e))
        return 100000.0

    def get_candles_with_retry(self, symbol_token: str, exchange: str, interval: str,
                                from_dt: datetime, to_dt: datetime) -> pd.DataFrame:
        """Fetch candles with exponential backoff retry on rate-limit errors."""
        if not self.session:
            # Mock data for dry-run
            freq = "3min" if "THREE" in interval else "15min"
            dates = pd.date_range(end=to_dt, periods=120, freq=freq)
            prices = np.random.randn(120).cumsum() + 400.0
            return pd.DataFrame({
                "timestamp": dates,
                "open": prices - 2, "high": prices + 4,
                "low": prices - 4, "close": prices, "volume": 50000
            })

        params = {
            "exchange": exchange,
            "symboltoken": symbol_token,
            "interval": interval,
            "fromdate": from_dt.strftime("%Y-%m-%d %H:%M"),
            "todate": to_dt.strftime("%Y-%m-%d %H:%M"),
        }

        for attempt in range(MAX_RETRIES):
            try:
                # Mandatory delay before every API call
                time.sleep(API_DELAY_SECONDS)
                resp = self.smart.getCandleData(params)

                if not resp.get("status"):
                    msg = str(resp.get("message", resp))
                    if "exceeding access rate" in msg.lower() or "access denied" in msg.lower():
                        wait = (2 ** attempt) * 5  # 5s, 10s, 20s, 40s
                        log.warning("Rate limited. Waiting %ds before retry %d/%d...", wait, attempt + 1, MAX_RETRIES)
                        time.sleep(wait)
                        continue
                    raise RuntimeError(f"Candle fetch failed: {msg}")

                data = resp.get("data")
                if not data:
                    return pd.DataFrame()

                df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                for col in ["open", "high", "low", "close"]:
                    df[col] = df[col].astype(float)
                return df

            except RuntimeError:
                raise
            except Exception as e:
                err_msg = str(e)
                if "exceeding access rate" in err_msg.lower() or "access denied" in err_msg.lower():
                    wait = (2 ** attempt) * 5
                    log.warning("Rate limited (exception). Waiting %ds before retry %d/%d...", wait, attempt + 1, MAX_RETRIES)
                    time.sleep(wait)
                    continue
                raise

        log.error("Max retries exceeded for token %s", symbol_token)
        return pd.DataFrame()

    def place_bracket_order(self, symbol: str, symbol_token: str, exchange: str,
                             qty: int, side: str, entry_price: float,
                             stop_loss: float, target: float, dry_run: bool):
        if dry_run or not self.session:
            log.info("[DRY-RUN] %s %s qty=%d entry=%.2f SL=%.2f TP=%.2f",
                     side, symbol, qty, entry_price, stop_loss, target)
            return {"status": True, "dry_run": True}

        order = {
            "variety": "ROBO",
            "tradingsymbol": symbol,
            "symboltoken": symbol_token,
            "transactiontype": side,
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
            log.info("Order placed: %s", resp)
            return resp
        except Exception as e:
            log.error("Order failed: %s", str(e))
            return {"status": False, "message": str(e)}


class ConfluenceStrategy:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def check_confluence(self, ltf_df: pd.DataFrame, htf_df: pd.DataFrame) -> dict:
        ltf = build_all_indicators(ltf_df)
        htf = build_all_indicators(htf_df)

        last = ltf.iloc[-1]
        htf_last = htf.iloc[-1]

        htf_trend = htf_last["dtm_trend"]
        ltf_trend = last["dtm_trend"]

        if htf_trend != ltf_trend:
            return {"signal": None, "reason": "HTF/LTF trend mismatch"}

        st_dir = last["st_direction"]
        hma_slope = last["hma_slope_pct"]
        bullish = (ltf_trend == 1 and st_dir == 1 and hma_slope > 0)
        bearish = (ltf_trend == -1 and st_dir == -1 and hma_slope < 0)

        if not (bullish or bearish):
            return {"signal": None, "reason": "No directional confluence"}

        if last["upper_wick_pct"] > self.cfg["wick_ratio_max"] * 100 or \
           last["lower_wick_pct"] > self.cfg["wick_ratio_max"] * 100:
            return {"signal": None, "reason": "Wick trap filter"}

        if last["dtm_band_width_pct"] < self.cfg["band_squeeze_min_pct"]:
            return {"signal": None, "reason": "Band squeeze too tight"}

        dist_pct = last["dist_to_pivot_high_pct"] if bullish else last["dist_to_pivot_low_pct"]
        if pd.isna(dist_pct) or dist_pct < self.cfg["pivot_distance_min_pct"]:
            return {"signal": None, "reason": "Too close to pivot wall"}

        if bullish and last["base_ema_slope_pct"] <= 0:
            return {"signal": None, "reason": "Base EMA not confirming"}
        if bearish and last["base_ema_slope_pct"] >= 0:
            return {"signal": None, "reason": "Base EMA not confirming"}

        entry = last["close"]
        if bullish:
            stop_loss = last["last_pivot_low"] if not pd.isna(last["last_pivot_low"]) else last["hma_50"]
            signal = "BUY"
        else:
            stop_loss = last["last_pivot_high"] if not pd.isna(last["last_pivot_high"]) else last["hma_50"]
            signal = "SELL"

        return {"signal": signal, "entry": entry, "stop_loss": stop_loss, "reason": "Confluence confirmed"}


def position_size(equity: float, risk_pct: float, entry: float, stop_loss: float) -> int:
    risk_amount = equity * (risk_pct / 100)
    per_share_risk = abs(entry - stop_loss)
    if per_share_risk <= 0:
        return 0
    return int(risk_amount / per_share_risk)


def load_config(path: str = "config.json") -> dict:
    if not os.path.exists(path):
        return {
            "angel_one": {},
            "trading": {
                "stocks": ["WIPRO", "COALINDIA", "PETRONET", "RAILTEL", "BIOCON",
                           "MOIL", "DLINKINDIA", "JYOTHYLAB", "TCS", "INFY"],
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
    threading.Thread(target=start_health_server, daemon=True).start()

    cfg = load_config()
    trading_cfg = cfg["trading"]
    dry_run = trading_cfg["dry_run"] and not force_live

    client = AngelOneClient(cfg["angel_one"])
    connected = client.login()
    if not connected:
        log.warning("Running in Virtual DRY-RUN mode (no live connection).")

    strategy = ConfluenceStrategy(trading_cfg)
    equity = client.get_equity()
    log.info("Account equity: %.2f | dry_run=%s", equity, dry_run)

    daily_max_loss = equity * (trading_cfg["daily_max_loss_pct"] / 100)
    daily_pnl = 0.0
    risk_per_trade_pct = trading_cfg["risk_per_trade_pct"]
    rr_ratio = trading_cfg["reward_risk_ratio"]

    # Official NSE tokens (Shariah-compliant list)
    symbol_tokens = {
        "WIPRO": "3721", "COALINDIA": "20396", "PETRONET": "11359",
        "RAILTEL": "12316", "BIOCON": "11373", "MOIL": "17937",
        "DLINKINDIA": "16075", "JYOTHYLAB": "15012", "TCS": "11536", "INFY": "1594"
    }

    while True:
        if daily_pnl <= -daily_max_loss:
            log.warning("Daily max loss limit hit (-%.2f). Halting for the session.", daily_max_loss)
            break

        for symbol in trading_cfg["stocks"]:
            token = symbol_tokens.get(symbol)
            if not token:
                log.warning("No token for %s, skipping.", symbol)
                continue

            try:
                # Always use IST for Angel One API (Render server is UTC)
                now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)

                # Fetch 3m candles - last 24 hours gives ~160 candles on a trading day
                ltf_df = client.get_candles_with_retry(
                    token, "NSE", trading_cfg["ltf"],
                    now_ist - timedelta(hours=24), now_ist
                )
                # Fetch 15m candles - last 10 days gives plenty of HTF context
                htf_df = client.get_candles_with_retry(
                    token, "NSE", trading_cfg["htf"],
                    now_ist - timedelta(days=10), now_ist
                )

                ltf_count = len(ltf_df)
                htf_count = len(htf_df)

                if ltf_df.empty or htf_df.empty or ltf_count < 40 or htf_count < 40:
                    log.info("%s: Skipping — not enough candles (LTF=%d, HTF=%d).", symbol, ltf_count, htf_count)
                    continue

                log.info("%s: Fetched LTF=%d candles, HTF=%d candles. Checking confluence...", symbol, ltf_count, htf_count)

                result = strategy.check_confluence(ltf_df, htf_df)
                if not result["signal"]:
                    log.info("%s: No signal — %s", symbol, result["reason"])
                    continue

                entry, stop = result["entry"], result["stop_loss"]
                qty = position_size(equity, risk_per_trade_pct, entry, stop)
                if qty <= 0:
                    log.warning("%s: qty=0, skipping (stop too close to entry?)", symbol)
                    continue

                target = entry + rr_ratio * abs(entry - stop) if result["signal"] == "BUY" \
                    else entry - rr_ratio * abs(entry - stop)

                log.info("%s: SIGNAL=%s entry=%.2f SL=%.2f TP=%.2f qty=%d",
                         symbol, result["signal"], entry, stop, target, qty)

                resp = client.place_bracket_order(
                    symbol=symbol, symbol_token=token, exchange="NSE",
                    qty=qty, side=result["signal"], entry_price=entry,
                    stop_loss=stop, target=target, dry_run=dry_run,
                )

            except Exception as e:
                log.error("Error processing %s: %s", symbol, str(e))

        log.info("Cycle complete. Sleeping 3 minutes...")
        time.sleep(180)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Force live trading mode")
    args = parser.parse_args()
    run(force_live=args.live)
