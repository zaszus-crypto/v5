#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XAUUSD BOT v25.4 — PAXG Multi-Source Live Edition
Modes:
  python bot.py                    # live signal
  python bot.py --backtest         # backtest 60d
  python bot.py --monitor          # force daily report
  python bot.py --monitor-check    # send if due
  python bot.py --monitor-dry-run  # preview report
  python bot.py --healthcheck      # watchdog
  python bot.py --audit-features   # feature importance
  python bot.py --test-sources     # test data sources
"""

import os, sys, json, pickle, logging, argparse
import time as py_time
import fcntl
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, time, timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, roc_auc_score, brier_score_loss


# ===========================================================================
# CONFIG
# ===========================================================================
@dataclass(frozen=True)
class Config:
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    gist_token: str = os.getenv("GH_GIST_TOKEN", "")
    gist_id: str = os.getenv("GH_GIST_ID", "")
    backtest_mode: bool = os.getenv("BACKTEST_MODE", "false").lower() == "true"
    monitor_hour_wib: int = int(os.getenv("MONITOR_HOUR_WIB", "8"))

    timeframe_minutes: int = 15
    min_candles: int = 300
    forward_bars: int = 4
    atr_period: int = 14
    atr_sl_multiplier: float = 1.5
    risk_reward: float = 2.5
    spread_usd: float = 0.30
    slippage_usd: float = 0.10
    commission_per_lot: float = 7.0

    base_min_signal_probability: float = 0.60
    max_threshold: float = 0.70
    min_oos_auc: float = 0.55
    min_oos_precision: float = 0.50

    max_entropy_threshold: float = 0.95
    min_hurst_threshold: float = 0.40
    max_wick_ratio_threshold: float = 0.70
    max_trend_age_bars: int = 18

    cooldown_minutes: int = 120
    choppy_cooldown_minutes: int = 30
    max_daily_signals: int = 6
    max_daily_loss_pct: float = 4.0
    session_aware_cooldown: bool = True
    session_aware_risk: bool = True
    friday_cutoff_hour: int = 21

    price_offset: float = 30.0
    account_equity: float = 200.0
    risk_percent: float = 1.0
    min_risk_percent: float = 0.5
    max_risk_percent: float = 1.5
    contract_size: float = 100.0
    max_lot: float = 1.0
    min_lot: float = 0.01

    meta_min_auc: float = 0.55
    meta_max_brier: float = 0.25
    meta_min_probability: float = 0.55
    meta_oof_splits: int = 3

    wf_splits: int = 3
    wf_min_auc: float = 0.55
    wf_min_prec: float = 0.50
    min_train_samples: int = 400

    vol_regime_min_pct: float = 0.20
    vol_regime_max_pct: float = 0.95

    rolling_window: int = 20
    min_rolling_winrate: float = 0.35
    max_consecutive_losses: int = 5
    kill_switch_path: str = "/tmp/xauusd_kill"

    model_cache_path: str = os.getenv("MODEL_CACHE_PATH", "/tmp/xauusd_model_v25.pkl")
    model_cache_ttl_min: int = 60
    max_journal_entries: int = 200
    backtest_retrain_every: int = 192

    healthcheck_interval_hrs: int = 6
    last_sig_alert_hrs: int = 12

    lock_path: str = "/tmp/xauusd_bot.lock"
    wib: ZoneInfo = ZoneInfo("Asia/Jakarta")


CONFIG = Config()

FEATURES = [
    "rsi", "adx", "di_difference", "ema_distance", "ema_slope",
    "trend_strength", "kalman_trend", "kalman_deviation", "atr_pct",
    "atr_regime", "realized_vol", "vol_zscore", "bb_position",
    "keltner_position", "hurst", "shannon_entropy", "body_ratio",
    "upper_wick_ratio", "lower_wick_ratio", "bos_score",
    "liquidity_sweep_score", "return_1", "return_3", "return_5",
    "volume_zscore", "volume_ratio",
    "hour_sin", "hour_cos", "dayofweek", "session_london",
    "session_newyork", "session_overlap",
]

SESSION_EMOJI = {"LONDON": "🇬🇧", "NEWYORK": "🇺🇸", "ASIAN": "🌏", "LATE": "🌙"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stderr,
    force=True,
)
log = logging.getLogger("xauusd_v254")


# ===========================================================================
# UTILITIES
# ===========================================================================
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def now_wib() -> datetime:
    return now_utc().astimezone(CONFIG.wib)

def validate_config():
    if not CONFIG.backtest_mode:
        miss = [k for k, v in [("TELEGRAM_TOKEN", CONFIG.telegram_token),
                                ("TELEGRAM_CHAT_ID", CONFIG.telegram_chat_id)] if not v]
        if miss:
            log.critical("Missing config: %s", ", ".join(miss)); sys.exit(1)
    if not CONFIG.gist_token or not CONFIG.gist_id:
        log.warning("Gist not configured — state won't persist.")

@contextmanager
def acquire_lock():
    lf = open(CONFIG.lock_path, "w")
    locked = False
    try:
        try:
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB); locked = True
        except BlockingIOError:
            log.warning("Another instance running.")
        yield locked
    finally:
        if locked:
            try: fcntl.flock(lf, fcntl.LOCK_UN)
            except Exception: pass
        lf.close()


# ===========================================================================
# SESSION / TIME
# ===========================================================================
def is_weekend() -> bool:
    return now_wib().weekday() >= 5

def is_friday_cutoff() -> bool:
    now = now_wib()
    return now.weekday() == 4 and now.hour >= CONFIG.friday_cutoff_hour

def get_current_session() -> str:
    t = now_wib().time()
    if time(13, 0) <= t <= time(17, 0): return "LONDON"
    if time(19, 0) <= t <= time(23, 0): return "NEWYORK"
    if time(0, 0) <= t < time(13, 0): return "ASIAN"
    return "LATE"

def get_session_cooldown() -> int:
    if not CONFIG.session_aware_cooldown:
        return CONFIG.cooldown_minutes
    return {"LONDON": 90, "NEWYORK": 120}.get(get_current_session(), 60)

def get_session_risk_multiplier() -> float:
    if not CONFIG.session_aware_risk:
        return 1.0
    return 1.0 if get_current_session() in ("LONDON", "NEWYORK") else 0.5

def is_high_impact_news_time() -> bool:
    t = now_wib().time()
    return time(19, 15) <= t <= time(20, 45)


# ===========================================================================
# STATE
# ===========================================================================
DEFAULT_STATE = {
    "version": 254,
    "last_signal_time": None,
    "last_choppy_time": None,
    "last_healthcheck": None,
    "daily_signals_count": 0,
    "daily_pnl_usd": 0.0,
    "last_reset_date": None,
    "last_report_date": None,
    "last_weekly_date": None,
    "active_signals": [],
    "performance_history": {"wins": 0, "losses": 0},
    "adaptive_threshold": 0.60,
    "recent_outcomes": [],
    "trade_journal": [],
    "session_outcomes": {},
    "last_model_metrics": {},
    "last_data_source": None,
}

def _req_sig_keys(s):
    return all(k in s for k in ("type","entry","sl","tp","lot","time","expires_at"))

def _merge_defaults(state):
    for k, v in DEFAULT_STATE.items():
        if k not in state:
            state[k] = json.loads(json.dumps(v))
    state["version"] = 254
    state["active_signals"] = [s for s in state.get("active_signals", [])
                                if isinstance(s, dict) and _req_sig_keys(s)]
    return state

def load_gist_state():
    today = now_wib().strftime("%Y-%m-%d")
    if not CONFIG.gist_token or not CONFIG.gist_id:
        return _merge_defaults({"last_reset_date": today})
    try:
        r = requests.get(f"https://api.github.com/gists/{CONFIG.gist_id}",
                          headers={"Authorization": f"token {CONFIG.gist_token}"},
                          timeout=8)
        if r.status_code == 200:
            for fn, fc in r.json().get("files", {}).items():
                if fn.endswith("bot_state.json"):
                    return _merge_defaults(json.loads(fc["content"]))
        else:
            log.warning("Gist load HTTP %d", r.status_code)
    except Exception as e:
        log.warning("Gist load error: %s", e)
    return _merge_defaults({"last_reset_date": today})

def save_gist_state(state, retries=3):
    if not CONFIG.gist_token or not CONFIG.gist_id: return False
    payload = {"files": {"bot_state.json": {"content": json.dumps(state, indent=2)}}}
    headers = {"Authorization": f"token {CONFIG.gist_token}"}
    for attempt in range(1, retries + 1):
        try:
            r = requests.patch(f"https://api.github.com/gists/{CONFIG.gist_id}",
                                json=payload, headers=headers, timeout=8)
            if r.status_code == 200: return True
        except Exception as e:
            log.warning("Gist save error: %s", e)
        py_time.sleep(2 ** attempt)
    return False

def reset_daily_state_if_needed(state):
    today = now_wib().strftime("%Y-%m-%d")
    if state.get("last_reset_date") != today:
        state["daily_signals_count"] = 0
        state["daily_pnl_usd"] = 0.0
        state["last_signal_time"] = None
        state["last_choppy_time"] = None
        state["last_reset_date"] = today
    return state


# ===========================================================================
# SIGNAL EVALUATION
# ===========================================================================
def cleanup_expired_signals(state):
    now = now_utc()
    remaining, outcomes = [], state.get("recent_outcomes", [])
    for sig in state.get("active_signals", []):
        try:
            if now > datetime.fromisoformat(sig["expires_at"]):
                log.info("Signal expired: %s", sig["type"])
                outcomes.append("timeout")
                sess = sig.get("session", "UNKNOWN")
                so = state.setdefault("session_outcomes", {}).setdefault(
                    sess, {"wins": 0, "losses": 0, "timeouts": 0})
                so["timeouts"] += 1
            else:
                remaining.append(sig)
        except Exception:
            pass
    state["active_signals"] = remaining
    state["recent_outcomes"] = outcomes[-50:]
    return state

def evaluate_active_signals(df, state):
    if not state.get("active_signals"): return state
    wins = state["performance_history"]["wins"]
    losses = state["performance_history"]["losses"]
    pnl = state.get("daily_pnl_usd", 0.0)
    recent = state.get("recent_outcomes", [])
    remaining = []

    for sig in state["active_signals"]:
        try:
            sig_time = pd.Timestamp(sig["time"])
            expires_at = pd.Timestamp(sig["expires_at"])
        except Exception:
            continue
        bars = df.loc[(df["datetime"] > sig_time) & (df["datetime"] <= expires_at)]
        if bars.empty:
            remaining.append(sig); continue

        result = None
        for _, bar in bars.iterrows():
            if sig["type"] == "BUY":
                hit_tp = bar["high"] >= sig["tp"]; hit_sl = bar["low"] <= sig["sl"]
            else:
                hit_tp = bar["low"] <= sig["tp"]; hit_sl = bar["high"] >= sig["sl"]
            if hit_sl: result = "loss"; break
            if hit_tp: result = "win"; break
        if result is None:
            remaining.append(sig); continue

        lot, entry = float(sig["lot"]), float(sig["entry"])
        if result == "win":
            wins += 1
            move = (sig["tp"] - entry) if sig["type"] == "BUY" else (entry - sig["tp"])
            pnl += move * lot * CONFIG.contract_size
            log.info("✅ WIN %s (+%.2f USD)", sig["type"], move*lot*CONFIG.contract_size)
        else:
            losses += 1
            move = (sig["sl"] - entry) if sig["type"] == "BUY" else (entry - sig["sl"])
            pnl += move * lot * CONFIG.contract_size
            log.info("❌ LOSS %s (%.2f USD)", sig["type"], move*lot*CONFIG.contract_size)
        recent.append(result)

        sess = sig.get("session", "UNKNOWN")
        so = state.setdefault("session_outcomes", {}).setdefault(
            sess, {"wins": 0, "losses": 0, "timeouts": 0})
        so["wins" if result == "win" else "losses"] += 1

    state["active_signals"] = remaining
    state["performance_history"]["wins"] = wins
    state["performance_history"]["losses"] = losses
    state["daily_pnl_usd"] = round(pnl, 2)
    state["recent_outcomes"] = recent[-50:]

    total = wins + losses
    if total > 0:
        wr = wins / total
        th = state["adaptive_threshold"]
        if wr < 0.45: th = min(th + 0.02, CONFIG.max_threshold)
        elif wr > 0.65: th = max(th - 0.01, CONFIG.base_min_signal_probability)
        state["adaptive_threshold"] = round(th, 3)
    return state

def check_kill_switch(state):
    if os.path.exists(CONFIG.kill_switch_path):
        log.critical("🛑 KILL SWITCH file found."); return True
    recent = state.get("recent_outcomes", [])
    if len(recent) >= CONFIG.rolling_window:
        window = recent[-CONFIG.rolling_window:]
        decided = [t for t in window if t in ("win", "loss")]
        if len(decided) >= 10:
            wr = sum(1 for t in decided if t == "win") / len(decided)
            if wr < CONFIG.min_rolling_winrate:
                log.critical("🛑 Rolling WR %.1f%%.", wr*100); return True
    consec = 0
    for t in reversed(recent):
        if t == "loss": consec += 1
        elif t == "win": break
    if consec >= CONFIG.max_consecutive_losses:
        log.critical("🛑 %d consecutive losses.", consec); return True
    return False


# ===========================================================================
# TELEGRAM
# ===========================================================================
def send_telegram(message):
    if CONFIG.backtest_mode: return True
    if not CONFIG.telegram_token or not CONFIG.telegram_chat_id: return False
    url = f"https://api.telegram.org/bot{CONFIG.telegram_token}/sendMessage"
    payload = {"chat_id": CONFIG.telegram_chat_id, "text": message,
               "parse_mode": "HTML", "disable_web_page_preview": True}
    for attempt in range(1, 4):
        try:
            r = requests.post(url, json=payload, timeout=8)
            if r.status_code == 200: return True
            if r.status_code == 429:
                py_time.sleep(int(r.json().get("parameters", {}).get("retry_after", 5)) + 1)
                continue
            return False
        except Exception:
            py_time.sleep(2 * attempt)
    return False


# ===========================================================================
# DATA FETCHERS — PAXG MULTI-EXCHANGE (v25.4)
# ===========================================================================
def _fetch_binance_paginated(symbol="PAXGUSDT", interval="15m", total_bars=5000):
    all_data, end_time, remaining = [], None, total_bars
    while remaining > 0:
        limit = min(1000, remaining)
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if end_time: params["endTime"] = end_time
        try:
            r = requests.get("https://api.binance.com/api/v3/klines",
                              params=params, timeout=8,
                              headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                if all_data: break
                raise RuntimeError(f"HTTP {r.status_code}")
            data = r.json()
            if not data or not isinstance(data, list): break
            all_data = data + all_data
            if len(data) < limit: break
            end_time = data[0][0] - 1
            remaining -= len(data)
            py_time.sleep(0.15)
        except Exception as e:
            if all_data: break
            raise RuntimeError(f"Binance: {str(e)[:80]}")
    if not all_data:
        raise RuntimeError("Binance: no data")
    df = pd.DataFrame(all_data, columns=[
        "ot","o","h","l","c","v","ct","qv","t","tb","tq","ig"])
    df["datetime"] = pd.to_datetime(df["ot"], unit="ms", utc=True)
    for a, b in [("o","open"),("h","high"),("l","low"),("c","close"),("v","volume")]:
        df[b] = pd.to_numeric(df[a], errors="coerce")
    df = df[["datetime","open","high","low","close","volume"]]
    df = df.dropna(subset=["datetime","open","high","low","close"])
    return df.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)

def _fetch_kraken(interval_min=15):
    try:
        r = requests.get("https://api.kraken.com/0/public/OHLC",
                          params={"pair": "PAXGUSD", "interval": interval_min},
                          timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200: raise RuntimeError(f"HTTP {r.status_code}")
        data = r.json()
        if data.get("error"): raise RuntimeError(f"API: {data['error']}")
        result = data.get("result", {})
        key = next((k for k in result.keys() if k != "last"), None)
        if not key: raise RuntimeError("no data key")
        rows = result[key]
        if not rows or len(rows) < 100: raise RuntimeError(f"only {len(rows)}")
        df = pd.DataFrame(rows, columns=["ts","open","high","low","close","vwap","volume","count"])
        df["datetime"] = pd.to_datetime(df["ts"].astype("int64"), unit="s", utc=True)
        for col in ("open","high","low","close","volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["datetime","open","high","low","close","volume"]]
        df = df.dropna(subset=["datetime","open","high","low","close"])
        return df.sort_values("datetime").reset_index(drop=True)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network: {e}")

def _fetch_coinbase(interval_sec=900):
    try:
        r = requests.get("https://api.exchange.coinbase.com/products/PAXG-USD/candles",
                          params={"granularity": interval_sec},
                          timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200: raise RuntimeError(f"HTTP {r.status_code}")
        data = r.json()
        if not data or not isinstance(data, list): raise RuntimeError("empty")
        df = pd.DataFrame(data, columns=["ts","low","high","open","close","volume"])
        df["datetime"] = pd.to_datetime(df["ts"].astype("int64"), unit="s", utc=True)
        for col in ("open","high","low","close","volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["datetime","open","high","low","close","volume"]]
        df = df.dropna(subset=["datetime","open","high","low","close"])
        return df.sort_values("datetime").reset_index(drop=True)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network: {e}")

def _fetch_bybit(interval="15", limit=200):
    try:
        r = requests.get("https://api.bybit.com/v5/market/kline",
                          params={"category":"spot","symbol":"PAXGUSDT",
                                  "interval":interval,"limit":limit},
                          timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200: raise RuntimeError(f"HTTP {r.status_code}")
        data = r.json()
        if data.get("retCode") != 0: raise RuntimeError(f"API: {data.get('retMsg')}")
        rows = data.get("result", {}).get("list", [])
        if not rows: raise RuntimeError("empty")
        df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume","turnover"])
        df["datetime"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
        for col in ("open","high","low","close","volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["datetime","open","high","low","close","volume"]]
        df = df.dropna(subset=["datetime","open","high","low","close"])
        return df.sort_values("datetime").reset_index(drop=True)
    except Exception as e:
        raise RuntimeError(f"Bybit: {str(e)[:80]}")

def _fetch_deriv(granularity_sec, count=2500):
    try:
        import websocket
    except ImportError:
        raise RuntimeError("websocket-client missing")
    url = "wss://ws.derivws.com/websockets/v3?app_id=1089"
    for symbol in ("frxXAUUSD", "XAUUSD"):
        payload = {"ticks_history": symbol, "adjust_start_time": 1,
                   "count": count, "end": "latest",
                   "granularity": granularity_sec, "style": "candles"}
        ws = None
        try:
            ws = websocket.create_connection(url, timeout=8,
                                              enable_multithread=True,
                                              header={"User-Agent": "Mozilla/5.0"})
            ws.send(json.dumps(payload))
            data = json.loads(ws.recv())
        except Exception:
            continue
        finally:
            if ws:
                try: ws.close()
                except Exception: pass
        if "error" in data: continue
        candles = data.get("candles", [])
        if not candles or len(candles) < 100: continue
        df = pd.DataFrame(candles)
        df["datetime"] = pd.to_datetime(df["epoch"].astype(int), unit="s", utc=True)
        for col in ("open","high","low","close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = 0.0
        df = df[["datetime","open","high","low","close","volume"]]
        df = df.dropna(subset=["datetime","open","high","low","close"])
        return df.sort_values("datetime").reset_index(drop=True)
    raise RuntimeError("Deriv all failed")

def _fetch_yahoo_gcf(interval="15m", range_period="60d"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval={interval}&range={range_period}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for attempt in range(1, 4):
        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            result = data.get("chart", {}).get("result")
            if not result: raise ValueError("Yahoo empty")
            res = result[0]
            ts = res.get("timestamp", [])
            quote = res.get("indicators", {}).get("quote", [{}])[0]
            if not ts or not quote: raise ValueError("Yahoo missing")
            df = pd.DataFrame({"epoch": ts,
                "open": quote.get("open"), "high": quote.get("high"),
                "low": quote.get("low"), "close": quote.get("close"),
                "volume": quote.get("volume")})
            df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
            for col in ("open","high","low","close"):
                df[col] = pd.to_numeric(df[col], errors="coerce") - CONFIG.price_offset
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
            df = df[["datetime","open","high","low","close","volume"]]
            df = df.dropna(subset=["datetime","open","high","low","close"])
            df = df.sort_values("datetime").reset_index(drop=True)
            if len(df) < CONFIG.min_candles:
                raise ValueError(f"short {len(df)}")
            return df
        except Exception as e:
            log.warning("Yahoo GC=F #%d: %s", attempt, e)
            py_time.sleep(2 * attempt)
    raise RuntimeError("Yahoo GC=F failed")

def fetch_candles(interval="15m", range_period="60d"):
    binance_iv = {"1m":"1m","5m":"5m","15m":"15m","30m":"30m","1h":"1h","4h":"4h"}.get(interval,"15m")
    kraken_iv = {"1m":1,"5m":5,"15m":15,"30m":30,"1h":60,"4h":240}.get(interval,15)
    coinbase_iv = {"1m":60,"5m":300,"15m":900,"30m":1800,"1h":3600,"4h":14400}.get(interval,900)
    bybit_iv = {"1m":"1","5m":"5","15m":"15","30m":"30","1h":"60","4h":"240"}.get(interval,"15")
    gran_map = {"1m":60,"5m":300,"15m":900,"30m":1800,"1h":3600,"4h":14400}
    gran = gran_map.get(interval, 900)

    try:
        df = _fetch_binance_paginated("PAXGUSDT", binance_iv, 5000)
        if len(df) >= CONFIG.min_candles:
            log.info("✅ BINANCE_PAXG: %d bars", len(df))
            return df, "BINANCE_PAXG"
    except Exception as e:
        log.warning("Binance PAXG failed: %s", e)

    try:
        df = _fetch_kraken(kraken_iv)
        if len(df) >= CONFIG.min_candles:
            log.info("✅ KRAKEN_PAXG: %d bars", len(df))
            return df, "KRAKEN_PAXG"
    except Exception as e:
        log.warning("Kraken failed: %s", e)

    try:
        df = _fetch_coinbase(coinbase_iv)
        if len(df) >= CONFIG.min_candles:
            log.info("✅ COINBASE_PAXG: %d bars", len(df))
            return df, "COINBASE_PAXG"
    except Exception as e:
        log.warning("Coinbase failed: %s", e)

    try:
        df = _fetch_bybit(bybit_iv, 200)
        if len(df) >= CONFIG.min_candles:
            log.info("✅ BYBIT_PAXG: %d bars", len(df))
            return df, "BYBIT_PAXG"
    except Exception as e:
        log.warning("Bybit failed: %s", e)

    try:
        df = _fetch_deriv(gran, 2500)
        if len(df) >= CONFIG.min_candles:
            log.info("✅ DERIV: %d bars", len(df))
            return df, "DERIV"
    except Exception as e:
        log.warning("Deriv failed: %s", e)

    try:
        df = _fetch_yahoo_gcf(interval, range_period)
        log.critical("⚠️ YAHOO_FUTURES fallback — offset $%.2f may be off", CONFIG.price_offset)
        return df, "YAHOO_FUTURES"
    except Exception as e:
        log.warning("Yahoo GC=F failed: %s", e)

    raise RuntimeError("ALL data sources failed")


def sanitize_and_trim(df):
    if df is None or len(df) < 2: return None
    df = df.drop_duplicates(subset="datetime")
    df = df[(df["high"] >= df["low"]) & (df["close"] > 0) & (df["open"] > 0)]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["open","high","low","close"])
    df = df.sort_values("datetime").reset_index(drop=True)
    if len(df) < 2: return None
    age_min = (now_utc() - df["datetime"].iloc[-1]).total_seconds() / 60
    if age_min > CONFIG.timeframe_minutes * 3:
        log.warning("Stale data (%.0f min).", age_min); return None
    if age_min < CONFIG.timeframe_minutes:
        df = df.iloc[:-1].reset_index(drop=True)
    if len(df) < CONFIG.min_candles: return None
    return df


def check_h1_trend_confluence():
    try:
        df, source = fetch_candles(interval="1h", range_period="60d")
        if df is None or len(df) < 250: return "UNKNOWN"
        log.info("H1 source: %s", source)
        df["ema50"] = df["close"].ewm(span=50, min_periods=50, adjust=False).mean()
        df["ema200"] = df["close"].ewm(span=200, min_periods=200, adjust=False).mean()
        c = df["close"].iloc[-2]
        e50 = df["ema50"].iloc[-2]
        e200 = df["ema200"].iloc[-2]
        if pd.isna(e200): return "UNKNOWN"
        if c > e50 > e200: return "BUY"
        if c < e50 < e200: return "SELL"
        return "NEUTRAL"
    except Exception as e:
        log.warning("H1 error: %s", e); return "UNKNOWN"


# ===========================================================================
# FEATURES
# ===========================================================================
def kalman_filter_1d(values, q=0.01, r=1.0):
    n = len(values); est, vel, err = np.zeros(n), np.zeros(n), np.zeros(n)
    est[0], vel[0], err[0] = values[0], 0.0, 1.0
    for i in range(1, n):
        pe = est[i-1] + vel[i-1]; per = err[i-1] + q
        kg = per / (per + r + 1e-12)
        est[i] = pe + kg * (values[i] - pe)
        vel[i] = 0.85 * vel[i-1] + 0.15 * (est[i] - est[i-1])
        err[i] = (1 - kg) * per
    return est, vel, err

def hurst_exponent(series, max_lag=20):
    s = np.asarray(series, dtype=float); s = s[np.isfinite(s)]
    if len(s) < max_lag * 2: return 0.5
    lags = np.arange(2, max_lag + 1)
    tau = np.array([np.std(s[lag:] - s[:-lag]) for lag in lags], dtype=float)
    valid = tau > 0
    if valid.sum() < 3: return 0.5
    try:
        return float(np.clip(np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)[0], 0.0, 1.0))
    except Exception: return 0.5

def shannon_entropy(returns, bins=10):
    r = np.asarray(returns, dtype=float); r = r[np.isfinite(r)]
    if len(r) < 20: return 1.0
    hist, _ = np.histogram(r, bins=bins)
    p = hist / (hist.sum() + 1e-12); p = p[p > 0]
    mx = np.log2(bins)
    return float(-np.sum(p * np.log2(p)) / mx) if mx > 0 else 1.0

def wilder_atr(df, period=14):
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(),
                     (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def calculate_adx(df, period=14):
    df = df.copy()
    up, dn = df["high"].diff(), -df["low"].diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = wilder_atr(df, period)
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-12)
    mdi = 100 * pd.Series(mdm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-12)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-12)
    df["plus_di"], df["minus_di"] = pdi, mdi
    df["di_difference"] = pdi - mdi
    df["adx"] = dx.ewm(alpha=1/period, adjust=False).mean()
    return df

def calculate_market_structure(df):
    df = df.copy(); sw = 5
    df["prior_high"] = df["high"].rolling(sw).max().shift(1)
    df["prior_low"] = df["low"].rolling(sw).min().shift(1)
    bos_up = (df["high"] > df["prior_high"]) & (df["close"] > df["prior_high"])
    bos_dn = (df["low"] < df["prior_low"]) & (df["close"] < df["prior_low"])
    df["bos_score"] = np.where(bos_up, 1.0, np.where(bos_dn, -1.0, 0.0))
    sweep_up = (df["high"] > df["prior_high"]) & (df["close"] < df["prior_high"])
    sweep_dn = (df["low"] < df["prior_low"]) & (df["close"] > df["prior_low"])
    df["liquidity_sweep_score"] = np.where(sweep_dn, 1.0, np.where(sweep_up, -1.0, 0.0))
    return df

def calculate_features(df):
    df = df.copy()
    d = df["close"].diff()
    g = d.clip(lower=0).rolling(14).mean()
    l = (-d.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + g / (l + 1e-12))
    df["atr"] = wilder_atr(df, CONFIG.atr_period)
    df["atr_pct"] = df["atr"] / (df["close"] + 1e-12)
    df["atr_regime"] = df["atr"] / (df["atr"].rolling(100).mean() + 1e-12)
    df["ema_fast"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_trend"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_distance"] = (df["ema_fast"] - df["ema_slow"]) / (df["atr"] + 1e-12)
    df["ema_slope"] = (df["ema_fast"] - df["ema_fast"].shift(3)) / (df["atr"] + 1e-12)
    df["trend_strength"] = (df["close"] - df["ema_trend"]) / (df["atr"] + 1e-12)
    kp, kv, _ = kalman_filter_1d(df["close"].to_numpy())
    df["kalman_trend"] = kv / (df["atr"].to_numpy() + 1e-12)
    df["kalman_deviation"] = (df["close"].to_numpy() - kp) / (df["atr"].to_numpy() + 1e-12)
    df = calculate_adx(df, CONFIG.atr_period)
    m = df["close"].rolling(20).mean(); s = df["close"].rolling(20).std()
    df["bb_position"] = (df["close"] - m) / (2 * s + 1e-12)
    km = df["close"].ewm(span=20, adjust=False).mean()
    kb = df["atr"].ewm(span=20, adjust=False).mean() * 2
    df["keltner_position"] = (df["close"] - km) / (kb + 1e-12)
    ret = df["close"].pct_change()
    df["realized_vol"] = ret.rolling(20).std() * np.sqrt(24 * 60 / CONFIG.timeframe_minutes)
    df["vol_zscore"] = (df["realized_vol"] - df["realized_vol"].rolling(100).mean()) / (
        df["realized_vol"].rolling(100).std() + 1e-12)
    logp = np.log(df["close"].to_numpy()); rarr = ret.to_numpy(); n = len(df)
    h = np.full(n, 0.5); e = np.full(n, 1.0)
    for i in range(100, n):
        h[i] = hurst_exponent(logp[i-99:i+1])
        e[i] = shannon_entropy(rarr[i-99:i+1])
    df["hurst"] = h; df["shannon_entropy"] = e
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = (df["close"] - df["open"]).abs() / (rng + 1e-12)
    df["upper_wick_ratio"] = (df["high"] - df[["open","close"]].max(axis=1)) / (rng + 1e-12)
    df["lower_wick_ratio"] = (df[["open","close"]].min(axis=1) - df["low"]) / (rng + 1e-12)
    df = calculate_market_structure(df)
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)
    df["volume_zscore"] = 0.0; df["volume_ratio"] = 1.0
    if "volume" in df.columns and df["volume"].notna().sum() > 20:
        v = df["volume"].fillna(0); sd = v.rolling(20).std()
        df["volume_zscore"] = ((v - v.rolling(20).mean()) / (sd + 1e-12)).clip(-5, 5)
        df["volume_ratio"] = (v / (v.rolling(20).mean() + 1e-12)).clip(0, 10)
    hr = df["datetime"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["session_london"] = hr.between(7, 15).astype(int)
    df["session_newyork"] = hr.between(12, 20).astype(int)
    df["session_overlap"] = hr.between(12, 15).astype(int)
    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

def triple_barrier_labels(df, fwd, atr_mult):
    df = df.copy()
    c, h, l, a = (df["close"].to_numpy(), df["high"].to_numpy(),
                   df["low"].to_numpy(), df["atr"].to_numpy())
    n = len(df); labels = np.zeros(n, dtype=int)
    for i in range(n):
        if i + fwd >= n: continue
        up = c[i] + a[i] * atr_mult; dn = c[i] - a[i] * atr_mult
        for j in range(i + 1, i + fwd + 1):
            if h[j] >= up: labels[i] = 1; break
            if l[j] <= dn: labels[i] = -1; break
    df["label"] = labels; return df


# ===========================================================================
# MODELS
# ===========================================================================
def build_base_model():
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=12,
                                  max_features="sqrt", class_weight="balanced_subsample",
                                  random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.025,
                                      subsample=0.8, random_state=42)
    return VotingClassifier([("rf", rf), ("gb", gb)], voting="soft")

def build_light_model():
    rf = RandomForestClassifier(n_estimators=80, max_depth=6, min_samples_leaf=20,
                                  max_features="sqrt", class_weight="balanced_subsample",
                                  random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.03,
                                      subsample=0.8, random_state=42)
    return VotingClassifier([("rf", rf), ("gb", gb)], voting="soft")

def purged_wf_validation(X, y):
    tscv = TimeSeriesSplit(n_splits=CONFIG.wf_splits)
    aucs, precs = [], []
    for tr_idx, va_idx in tscv.split(X):
        tr_end = tr_idx[-1] - CONFIG.forward_bars
        if tr_end < 100: continue
        tr_slice = np.arange(tr_idx[0], tr_end + 1)
        if len(np.unique(y[tr_slice])) < 2: continue
        try:
            m = build_light_model(); m.fit(X[tr_slice], y[tr_slice])
            proba = m.predict_proba(X[va_idx]); preds = m.predict(X[va_idx])
            dmask = y[va_idx] != 0
            if dmask.sum() < 10: continue
            y_dir, p_dir = y[va_idx][dmask], preds[dmask]
            if len(np.unique(y_dir)) >= 2:
                precs.append(precision_score(y_dir, p_dir, average="macro", zero_division=0))
            classes = list(m.classes_)
            if 1 in classes and -1 in classes:
                bi, si = classes.index(1), classes.index(-1)
                y_bin = (y_dir == 1).astype(int)
                if len(np.unique(y_bin)) >= 2:
                    aucs.append(roc_auc_score(y_bin, proba[dmask][:, [si, bi]][:, 1]))
        except Exception as e:
            log.warning("WF fold: %s", e)
    if not aucs:
        return {"valid": False, "auc_mean": 0.5, "prec_mean": 0.0}
    return {"valid": True, "auc_mean": float(np.mean(aucs)),
            "prec_mean": float(np.mean(precs)) if precs else 0.0}

def generate_oof(X, y, n_splits=3, purge=4):
    oof = np.full(len(y), 0, dtype=int)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for tr_idx, va_idx in tscv.split(X):
        tr_end = tr_idx[-1] - purge
        if tr_end < 100: continue
        tr_slice = np.arange(tr_idx[0], tr_end + 1)
        if len(np.unique(y[tr_slice])) < 2: continue
        try:
            m = build_base_model(); m.fit(X[tr_slice], y[tr_slice])
            oof[va_idx] = m.predict(X[va_idx])
        except Exception: pass
    return oof

def _train_meta_models(df_train, run_wf_gate=True):
    df_lab = triple_barrier_labels(df_train, CONFIG.forward_bars, CONFIG.atr_sl_multiplier)
    train_df = df_lab.iloc[:-CONFIG.forward_bars]
    X = train_df[FEATURES].to_numpy(); y = train_df["label"].to_numpy()
    if len(X) < CONFIG.min_train_samples or len(np.unique(y)) < 2:
        return {"valid": False, "reason": "insufficient_data"}
    wf = None
    if run_wf_gate:
        wf = purged_wf_validation(X, y)
        if (not wf["valid"] or wf["auc_mean"] < CONFIG.wf_min_auc
                or wf["prec_mean"] < CONFIG.wf_min_prec):
            return {"valid": False, "reason": f"wf_gate_{wf.get('auc_mean',0):.2f}"}
    oof = generate_oof(X, y, CONFIG.meta_oof_splits, CONFIG.forward_bars)
    meta_mask = (oof != 0) & (y != 0)
    if meta_mask.sum() < 150:
        return {"valid": False, "reason": "insufficient_meta"}
    X_meta = X[meta_mask]; y_meta = (y[meta_mask] == oof[meta_mask]).astype(int)
    if len(np.unique(y_meta)) < 2 or not (0.30 <= y_meta.mean() <= 0.70):
        return {"valid": False, "reason": "meta_unbalanced"}
    split = int(len(X_meta) * 0.8); tr_end = split - CONFIG.forward_bars
    if tr_end < 100:
        return {"valid": False, "reason": "insufficient_meta_train"}
    meta_val = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=20,
                                        class_weight="balanced", random_state=42, n_jobs=-1)
    meta_val.fit(X_meta[:tr_end], y_meta[:tr_end])
    meta_prob_te = meta_val.predict_proba(X_meta[split:])[:, 1]
    try: meta_auc = roc_auc_score(y_meta[split:], meta_prob_te)
    except Exception: meta_auc = 0.5
    meta_brier = brier_score_loss(y_meta[split:], meta_prob_te)
    if meta_auc < CONFIG.meta_min_auc or meta_brier > CONFIG.meta_max_brier:
        return {"valid": False, "reason": f"meta_gate_auc{meta_auc:.2f}"}
    weights = np.exp(-np.arange(len(y))[::-1] / 500.0)
    primary = build_base_model(); primary.fit(X, y, sample_weight=weights)
    meta = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=20,
                                    class_weight="balanced", random_state=42, n_jobs=-1)
    meta.fit(X_meta, y_meta, sample_weight=weights[meta_mask])
    return {"valid": True, "primary": primary, "meta": meta,
            "meta_auc": meta_auc, "meta_brier": meta_brier,
            "wf_auc": wf["auc_mean"] if wf else 0.5,
            "wf_prec": wf["prec_mean"] if wf else 0.0}

def predict_from_models(df_row, models):
    if not models.get("valid") or "primary" not in models: return models
    primary, meta = models["primary"], models["meta"]
    X = df_row[FEATURES].to_numpy()
    primary_pred = int(primary.predict(X)[0])
    primary_proba = primary.predict_proba(X)[0]
    classes = list(primary.classes_)
    proba_dict = {int(c): float(primary_proba[i]) for i, c in enumerate(classes)}
    if primary_pred == 0: return {**models, "side": 0}
    meta_prob = float(meta.predict_proba(X)[0, 1])
    return {**models, "side": primary_pred,
            "primary_buy_prob": proba_dict.get(1, 0.0),
            "primary_sell_prob": proba_dict.get(-1, 0.0),
            "meta_prob": meta_prob}


# ===========================================================================
# CACHE
# ===========================================================================
def load_cached_models(fp):
    try:
        if not os.path.exists(CONFIG.model_cache_path): return None
        with open(CONFIG.model_cache_path, "rb") as f: cache = pickle.load(f)
        if cache.get("fingerprint") != fp: return None
        if (py_time.time() - cache["ts"]) / 60 > CONFIG.model_cache_ttl_min: return None
        log.info("Cache HIT"); return cache["payload"]
    except Exception as e:
        log.warning("Cache load: %s", e); return None

def save_cached_models(fp, payload):
    try:
        with open(CONFIG.model_cache_path, "wb") as f:
            pickle.dump({"fingerprint": fp, "payload": payload, "ts": py_time.time()}, f)
    except Exception as e:
        log.warning("Cache save: %s", e)

def model_fingerprint(df_feat):
    return f"v254_{df_feat['datetime'].iloc[-1].floor('h').isoformat()}"

def get_or_train_models(df_feat, use_cache=True):
    fp = model_fingerprint(df_feat)
    if use_cache:
        cached = load_cached_models(fp)
        if cached is not None: return cached
    models = _train_meta_models(df_feat, run_wf_gate=True)
    if models.get("valid"): save_cached_models(fp, models)
    return models


# ===========================================================================
# VOL REGIME
# ===========================================================================
def check_vol_regime(df_feat):
    rv = df_feat["realized_vol"].dropna()
    if len(rv) < 200: return True, 0.5
    pct = float((rv.tail(200) <= rv.iloc[-1]).mean())
    return (CONFIG.vol_regime_min_pct <= pct <= CONFIG.vol_regime_max_pct, pct)


# ===========================================================================
# TRADE COMPUTE
# ===========================================================================
def compute_trade(sig_type, df_feat, prob, meta_prob, state, source):
    last = df_feat.iloc[-1]
    price, atr = float(last["close"]), float(last["atr"])
    sl_dist = atr * CONFIG.atr_sl_multiplier
    tp_dist = sl_dist * CONFIG.risk_reward
    if sig_type == "BUY":
        sl, tp = price - sl_dist - CONFIG.spread_usd, price + tp_dist
    else:
        sl, tp = price + sl_dist + CONFIG.spread_usd, price - tp_dist
    th = state["adaptive_threshold"]
    ratio = prob / max(th, 1e-9)
    meta_boost = max(0.5, min(meta_prob / max(CONFIG.meta_min_probability, 1e-9), 1.3))
    session_mult = get_session_risk_multiplier()
    risk_pct = max(min(CONFIG.risk_percent * ratio * meta_boost * session_mult,
                        CONFIG.max_risk_percent), CONFIG.min_risk_percent * session_mult)
    risk_usd = CONFIG.account_equity * risk_pct / 100.0
    sl_abs = abs(price - sl)
    lot = (CONFIG.min_lot if sl_abs <= 0 else
           max(min(round(risk_usd / (sl_abs * CONFIG.contract_size), 2),
                   CONFIG.max_lot), CONFIG.min_lot))
    expires = now_utc() + timedelta(minutes=CONFIG.timeframe_minutes * CONFIG.forward_bars * 2)
    return {"type": sig_type, "entry": round(price, 2), "sl": round(sl, 2),
            "tp": round(tp, 2), "lot": lot, "risk_pct": round(risk_pct, 2),
            "risk_usd": round(risk_usd, 2), "prob": prob, "meta_prob": meta_prob,
            "session": get_current_session(), "source": source,
            "time": now_utc().isoformat(), "expires_at": expires.isoformat()}

def has_conflicting_signal(state, new_type):
    now = now_utc()
    for s in state.get("active_signals", []):
        if not s.get("type"): continue
        if s["type"] != new_type: return True
        try:
            if (now - datetime.fromisoformat(s["time"])).total_seconds() < 3600:
                return True
        except Exception: pass
    return False

def append_trade_journal(trade, df_feat, state):
    last = df_feat.iloc[-1]
    entry = {"ts": now_utc().isoformat(), "candle_ts": last["datetime"].isoformat(),
             "type": trade["type"], "session": trade.get("session", "UNKNOWN"),
             "source": trade.get("source", "UNKNOWN"),
             "entry": trade["entry"], "sl": trade["sl"], "tp": trade["tp"],
             "lot": trade["lot"], "prob": trade["prob"],
             "meta_prob": trade.get("meta_prob", 0),
             "features": {f: float(last[f]) for f in FEATURES if pd.notna(last.get(f))}}
    j = state.get("trade_journal", []); j.append(entry)
    state["trade_journal"] = j[-CONFIG.max_journal_entries:]

def format_signal_message(trade, h1, hurst, entropy, wick, time_str, meta, source):
    emoji = "🟢" if trade["type"] == "BUY" else "🔴"
    src_emoji = {"BINANCE_PAXG":"🟢","KRAKEN_PAXG":"🟡","COINBASE_PAXG":"🟡",
                 "BYBIT_PAXG":"🟡","DERIV":"🟢","YAHOO_FUTURES":"🔴"}.get(source,"❓")
    return (
        f"<b>{emoji} XAUUSD {trade['type']} [{trade.get('session','?')}]</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕒 {time_str}\n"
        f"📡 Source: {src_emoji} {source}\n"
        f"🎯 Primary: {trade['prob']*100:.1f}% | 🧠 Meta: {trade['meta_prob']*100:.1f}%\n"
        f"📊 H1: {h1} | 🌊 H {hurst:.2f} E {entropy:.2f}\n"
        f"🕯️ Wick: {wick*100:.0f}%\n"
        f"📈 OOS: AUC {meta.get('meta_auc',0):.3f} Brier {meta.get('meta_brier',0):.3f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📍 Entry: {trade['entry']:.2f}\n"
        f"🛑 SL: {trade['sl']:.2f}\n"
        f"🎯 TP: {trade['tp']:.2f}\n"
        f"⚖️ Lot: {trade['lot']} (Risk {trade['risk_pct']:.1f}% = ${trade['risk_usd']:.2f})\n"
        f"<i>Expires: {trade['expires_at'][:16].replace('T',' ')} UTC</i>"
    )


# ===========================================================================
# MONITOR
# ===========================================================================
def is_monitor_due(state):
    now = now_wib()
    if now.weekday() >= 5: return False
    if state.get("last_report_date") == now.strftime("%Y-%m-%d"): return False
    return now.hour >= CONFIG.monitor_hour_wib

def _wr_badge(wr):
    return "✅" if wr >= 0.60 else "🟡" if wr >= 0.50 else "🟠" if wr >= 0.40 else "🔴"

def _compute_monitor_metrics(state):
    perf = state.get("performance_history", {})
    wins, losses = perf.get("wins", 0), perf.get("losses", 0)
    total = wins + losses
    wr = wins / total if total > 0 else 0.0
    recent = state.get("recent_outcomes", [])
    roll = [t for t in recent[-20:] if t in ("win", "loss")]
    roll_wr = sum(1 for t in roll if t == "win") / len(roll) if roll else 0.0
    consec = 0
    for t in reversed(recent):
        if t == "loss": consec += 1
        elif t == "win": break
    sess_out = state.get("session_outcomes", {})
    sess_breakdown = []
    for sess in ("LONDON","NEWYORK","ASIAN","LATE"):
        d = sess_out.get(sess, {})
        w, l, t = d.get("wins",0), d.get("losses",0), d.get("timeouts",0)
        n = w + l
        sess_breakdown.append({"session": sess, "wins": w, "losses": l,
                                "timeouts": t, "trades": n,
                                "win_rate": w/n if n > 0 else 0.0})
    return {"wins": wins, "losses": losses, "total": total, "win_rate": wr,
            "roll_wr": roll_wr, "roll_n": len(roll), "consec_losses": consec,
            "active_n": len(state.get("active_signals", [])),
            "recent_10": recent[-10:], "sess_breakdown": sess_breakdown,
            "model_metrics": state.get("last_model_metrics", {}),
            "daily_pnl": state.get("daily_pnl_usd", 0.0),
            "daily_signals": state.get("daily_signals_count", 0),
            "adaptive_threshold": state.get("adaptive_threshold", 0.60),
            "last_signal_time": state.get("last_signal_time"),
            "last_data_source": state.get("last_data_source")}

def _compute_monitor_alerts(m, state):
    alerts = []
    if os.path.exists(CONFIG.kill_switch_path):
        alerts.append("🔴 <b>KILL SWITCH</b> aktif")
    if m["consec_losses"] >= CONFIG.max_consecutive_losses:
        alerts.append(f"🔴 {m['consec_losses']} consecutive losses")
    elif m["consec_losses"] >= 3:
        alerts.append(f"🟠 {m['consec_losses']} consecutive losses")
    if m["roll_n"] >= 10 and m["roll_wr"] < CONFIG.min_rolling_winrate:
        alerts.append(f"🔴 Rolling WR {m['roll_wr']*100:.1f}%")
    if m["adaptive_threshold"] >= CONFIG.max_threshold:
        alerts.append(f"🟠 Threshold stuck {m['adaptive_threshold']:.2f}")
    src = m.get("last_data_source", "") or ""
    if src == "YAHOO_FUTURES":
        alerts.append("🔴 YAHOO FUTURES — all PAXG failed!")
    elif src.startswith("BINANCE_"):
        alerts.append(f"🟢 {src}")
    elif src.endswith("_PAXG"):
        alerts.append(f"🟡 {src}")
    if not alerts:
        alerts.append("🟢 Sistem normal")
    return alerts

def _format_monitor_daily(m, alerts):
    now = now_wib()
    day = ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"][now.weekday()]
    parts = [f"<b>📊 XAUUSD — DAILY REPORT</b>",
             f"🗓️ {now.strftime('%Y-%m-%d')} ({day}) {now.strftime('%H:%M')} WIB",
             "━━━━━━━━━━━━━━━━━━━━━━", "", "<b>⚠️ ALERTS</b>"]
    parts.extend(alerts); parts.append("")
    parts.append("<b>📈 PERFORMANCE</b>")
    if m["total"] > 0:
        parts.append(f"Trades: <b>{m['total']}</b> (W:{m['wins']} L:{m['losses']}) "
                      f"{_wr_badge(m['win_rate'])} <b>{m['win_rate']*100:.1f}%</b>")
        if m["roll_n"] >= 5:
            parts.append(f"Rolling {m['roll_n']}: <b>{m['roll_wr']*100:.1f}%</b> "
                          f"{_wr_badge(m['roll_wr'])}")
    else:
        parts.append("Belum ada trade.")
    parts.append("")
    pe = "🟢" if m["daily_pnl"] >= 0 else "🔴"
    parts.append(f"<b>💰 HARI INI</b>\nPnL: {pe} <b>${m['daily_pnl']:+.2f}</b> "
                  f"| Signals: <b>{m['daily_signals']}</b>\n")
    if any(s["trades"] > 0 or s["timeouts"] > 0 for s in m["sess_breakdown"]):
        parts.append("<b>🎯 SESSION</b>")
        for s in m["sess_breakdown"]:
            if s["trades"] == 0 and s["timeouts"] == 0: continue
            em = SESSION_EMOJI.get(s["session"], "•")
            wb = _wr_badge(s["win_rate"]) if s["trades"] > 0 else "—"
            parts.append(f"{em} {s['session']:<8} {s['trades']:>2} {wb} "
                          f"{s['win_rate']*100:>5.1f}%"
                          + (f" ({s['timeouts']}T)" if s["timeouts"] else ""))
        parts.append("")
    if m["recent_10"]:
        icons = "".join({"win":"✅","loss":"❌","timeout":"⏱️"}.get(t,"?")
                         for t in m["recent_10"])
        w10 = sum(1 for t in m["recent_10"] if t == "win")
        parts.append(f"<b>🕐 RECENT</b>\n{icons} — {w10}/{len(m['recent_10'])}\n")
    mm = m["model_metrics"]
    if mm:
        parts.append("<b>🧠 MODEL</b>")
        parts.append(f"Meta AUC: <b>{mm.get('meta_auc',0):.3f}</b> "
                      f"| Brier: <b>{mm.get('meta_brier',0):.3f}</b>")
        parts.append(f"WF AUC: {mm.get('wf_auc',0):.3f} "
                      f"| WF Prec: {mm.get('wf_prec',0):.3f}\n")
    parts.append("<b>⚙️ SYSTEM</b>")
    parts.append(f"Threshold: <b>{m['adaptive_threshold']:.2f}</b>")
    parts.append(f"Source: <b>{m.get('last_data_source', 'N/A')}</b>")
    parts.append(f"Active: {m['active_n']}")
    parts.append("")
    parts.append(f"<i>💵 Modal: ${CONFIG.account_equity:.2f}</i>")
    return "\n".join(parts)

def _format_monitor_weekly(m):
    now = now_wib()
    parts = ["<b>📅 XAUUSD — WEEKLY</b>",
             f"🗓️ Week ending {now.strftime('%d/%m/%Y')}",
             "━━━━━━━━━━━━━━━━━━━━━━", "",
             f"Closed: <b>{m['total']}</b>",
             f"WR: <b>{m['win_rate']*100:.1f}%</b> {_wr_badge(m['win_rate'])}"]
    if m["sess_breakdown"] and any(s["trades"] >= 3 for s in m["sess_breakdown"]):
        best = max(m["sess_breakdown"], key=lambda s: s["win_rate"] if s["trades"] >= 3 else 0)
        parts.append(f"Best: <b>{best['session']}</b> ({best['win_rate']*100:.0f}%)")
    parts.append("\n<b>💡 REKOMENDASI</b>")
    if m["total"] < 10: parts.append("⏳ Sample sedikit.")
    elif m["win_rate"] >= 0.55 and m["roll_n"] >= 10 and m["roll_wr"] >= 0.50:
        parts.append("✅ Sehat. Naikkan modal pelan.")
    elif m["win_rate"] >= 0.45: parts.append("🟡 Marginal. Pertahankan modal.")
    elif m["win_rate"] < 0.40 and m["total"] >= 20:
        parts.append("🔴 STOP, backtest ulang.")
    else: parts.append("🟠 Waspada.")
    return "\n".join(parts)

def run_monitor(state=None, force=False, dry_run=False):
    if state is None: state = load_gist_state()
    if not force and not is_monitor_due(state):
        log.info("Monitor not due."); return
    m = _compute_monitor_metrics(state)
    alerts = _compute_monitor_alerts(m, state)
    daily = _format_monitor_daily(m, alerts)
    is_monday = now_wib().weekday() == 0
    if dry_run:
        print("\n" + "="*60); print(daily); print("="*60)
        if is_monday:
            print("\n" + "="*60); print(_format_monitor_weekly(m)); print("="*60)
        return
    if not send_telegram(daily):
        log.error("Daily report FAILED."); return
    state["last_report_date"] = now_wib().strftime("%Y-%m-%d")
    log.info("✅ Daily sent.")
    if is_monday:
        if send_telegram(_format_monitor_weekly(m)):
            state["last_weekly_date"] = now_wib().strftime("%Y-%m-%d")
            log.info("✅ Weekly sent.")
    save_gist_state(state)


# ===========================================================================
# HEALTHCHECK
# ===========================================================================
def run_healthcheck():
    log.info("=== HEALTHCHECK ===")
    state = load_gist_state()
    now = now_utc()
    last_hc = state.get("last_healthcheck")
    if last_hc:
        try:
            if (now - datetime.fromisoformat(last_hc)).total_seconds() < CONFIG.healthcheck_interval_hrs * 3600:
                log.info("Not due."); return
        except Exception: pass
    problems = []
    if os.path.exists(CONFIG.kill_switch_path):
        problems.append("🔴 Kill switch active")
    if state.get("last_signal_time") and not is_weekend() and not is_friday_cutoff():
        try:
            hrs = (now - datetime.fromisoformat(state["last_signal_time"])).total_seconds() / 3600
            if hrs > CONFIG.last_sig_alert_hrs:
                problems.append(f"🟠 No signal {hrs:.0f}h")
        except Exception: pass
    if now_wib().hour >= 10 and state.get("last_report_date") != now_wib().strftime("%Y-%m-%d") and now_wib().weekday() < 5:
        problems.append("🟠 Daily report missing")
    if state.get("last_data_source") == "YAHOO_FUTURES":
        problems.append("🔴 FUTURES fallback")
    if not problems:
        log.info("OK."); state["last_healthcheck"] = now.isoformat()
        save_gist_state(state); return
    send_telegram("<b>⚠️ HEALTHCHECK</b>\n" + "\n".join(problems))
    state["last_healthcheck"] = now.isoformat()
    save_gist_state(state)


# ===========================================================================
# AUDIT
# ===========================================================================
def run_audit_features():
    log.info("=== FEATURE AUDIT ===")
    try:
        df, source = fetch_candles()
    except Exception as e:
        log.error("Fetch: %s", e); return
    log.info("Source: %s | %d bars", source, len(df))
    df = sanitize_and_trim(df)
    if df is None: return
    df_feat = calculate_features(df)
    df_lab = triple_barrier_labels(df_feat, CONFIG.forward_bars, CONFIG.atr_sl_multiplier)
    train = df_lab.iloc[:-CONFIG.forward_bars]
    X, y = train[FEATURES].to_numpy(), train["label"].to_numpy()
    if len(np.unique(y)) < 2: log.error("One class."); return
    split = int(len(X) * 0.8); tr_end = split - CONFIG.forward_bars
    if tr_end < 100: log.error("Insufficient."); return
    model = build_light_model(); model.fit(X[:tr_end], y[:tr_end])
    try:
        from sklearn.inspection import permutation_importance
        imp = permutation_importance(model, X[split:], y[split:], n_repeats=5,
                                       random_state=42, n_jobs=-1)
    except Exception as e:
        log.error("Importance: %s", e); return
    results = sorted(zip(FEATURES, imp.importances_mean, imp.importances_std),
                      key=lambda x: -x[1])
    print("\n" + "="*70 + "\nFEATURE IMPORTANCE\n" + "="*70)
    print(f"{'Feature':<24} {'Mean':>10} {'Std':>8}  Bar")
    print("-"*70)
    for name, mean, std in results:
        bar = "█" * int(min(50, max(0, mean * 300)))
        print(f"{name:<24} {mean:>+10.4f} {std:>8.4f}  {bar}")
    to_drop = [n for n, m, s in results if m < 0.001]
    keep = [n for n, m, s in results if m >= 0.001]
    print("\n" + "="*70)
    print(f"DROP ({len(to_drop)}): {to_drop}")
    print(f"KEEP ({len(keep)}): {keep}")
    print("="*70)


# ===========================================================================
# TEST SOURCES
# ===========================================================================
def run_test_sources():
    print("Testing all sources (15m, ~500 bars)...\n")
    tests = [
        ("BINANCE PAXG (paged)", lambda: _fetch_binance_paginated("PAXGUSDT", "15m", 500)),
        ("KRAKEN PAXG", lambda: _fetch_kraken(15)),
        ("COINBASE PAXG", lambda: _fetch_coinbase(900)),
        ("BYBIT PAXG", lambda: _fetch_bybit("15", 200)),
        ("YAHOO GC=F", lambda: _fetch_yahoo_gcf("15m", "5d")),
    ]
    results = {}
    for name, fn in tests:
        t0 = py_time.time()
        try:
            df = fn()
            el = py_time.time() - t0
            last = df.iloc[-1]
            close = float(last["close"])
            results[name] = close
            print(f"✅ {name:<22} {len(df):>5} bars | {el:>5.2f}s | close=${close:.2f}")
        except Exception as e:
            print(f"❌ {name:<22} FAILED: {str(e)[:80]}")
    if len(results) >= 2:
        print("\n=== DIFFERENCES ===")
        base = results.get("BINANCE PAXG (paged)") or next(iter(results.values()))
        for name, close in results.items():
            diff = close - base
            pct = diff / base * 100
            print(f"{name:<22} ${diff:+.2f} ({pct:+.2f}%)")
    print("\nPAXG sources should agree within $5. GC=F ~$20-40 above.")


# ===========================================================================
# BACKTEST
# ===========================================================================
@dataclass
class BacktestConfig:
    initial_equity: float = 10_000.0
    risk_percent: float = 1.0

class WalkForwardBacktest:
    def __init__(self, df_feat, cfg):
        self.df = df_feat.reset_index(drop=True); self.cfg = cfg
        self.trades, self.equity_curve, self.equity = [], [], cfg.initial_equity
    def run(self):
        n = len(self.df); start = CONFIG.min_train_samples + CONFIG.forward_bars
        models, last_retrain, skipped = None, -10**9, 0
        for i in range(start, n - CONFIG.forward_bars):
            if i - last_retrain >= CONFIG.backtest_retrain_every or models is None:
                models = _train_meta_models(self.df.iloc[:i], run_wf_gate=False)
                last_retrain = i
                if not models.get("valid"): skipped += 1
            if not models.get("valid"): self._rec(i); continue
            pred = predict_from_models(self.df.iloc[[i]], models)
            side = pred.get("side", 0)
            if side == 0 or pred.get("meta_prob", 0) < CONFIG.meta_min_probability:
                self._rec(i); continue
            t = self._sim(i, side); self.trades.append(t)
            self.equity += t["pnl_usd"]; self._rec(i)
        log.info("Backtest: %d trades, %d skips", len(self.trades), skipped)
        return self._report()
    def _sim(self, i, direction):
        row = self.df.iloc[i]; entry = float(row["close"])
        atr = float(row["atr"]) if pd.notna(row["atr"]) else 10.0
        sl_dist = atr * CONFIG.atr_sl_multiplier; tp_dist = sl_dist * CONFIG.risk_reward
        if direction == 1:
            entry_eff = entry + CONFIG.spread_usd/2 + CONFIG.slippage_usd
            sl, tp = entry - sl_dist, entry + tp_dist
        else:
            entry_eff = entry - CONFIG.spread_usd/2 - CONFIG.slippage_usd
            sl, tp = entry + sl_dist, entry - tp_dist
        outcome, exit_price, exit_idx = "timeout", entry, i + CONFIG.forward_bars
        for j in range(i+1, min(i+CONFIG.forward_bars+1, len(self.df))):
            bar = self.df.iloc[j]
            if direction == 1: hit_sl, hit_tp = bar["low"] <= sl, bar["high"] >= tp
            else: hit_sl, hit_tp = bar["high"] >= sl, bar["low"] <= tp
            if hit_sl: outcome, exit_price, exit_idx = "loss", sl, j; break
            if hit_tp: outcome, exit_price, exit_idx = "win", tp, j; break
        else:
            exit_price = float(self.df.iloc[min(i+CONFIG.forward_bars, len(self.df)-1)]["close"])
        move = ((exit_price - entry_eff) if direction == 1 else (entry_eff - exit_price))
        risk_usd = self.equity * self.cfg.risk_percent / 100.0
        lot = max(CONFIG.min_lot, min(round(risk_usd/(sl_dist*CONFIG.contract_size), 2), CONFIG.max_lot))
        pnl = move * lot * CONFIG.contract_size - lot * CONFIG.commission_per_lot * 2
        return {"entry_idx": i, "exit_idx": exit_idx, "direction": direction,
                "entry": round(entry_eff,2), "exit": round(exit_price,2),
                "sl": round(sl,2), "tp": round(tp,2), "lot": lot,
                "outcome": outcome, "pnl_usd": round(pnl,2)}
    def _rec(self, i): self.equity_curve.append({"bar": i, "equity": self.equity})
    def _report(self):
        if not self.trades: return {"error": "no_trades"}
        df = pd.DataFrame(self.trades); eq = pd.Series([e["equity"] for e in self.equity_curve])
        wins, losses = df[df["outcome"]=="win"], df[df["outcome"]=="loss"]
        n = len(df); wr = len(wins)/n if n else 0
        pf = (wins["pnl_usd"].sum()/abs(losses["pnl_usd"].sum())
              if len(losses) and losses["pnl_usd"].sum() != 0 else 999)
        ret = eq.pct_change().dropna()
        sharpe = (ret.mean()/ret.std()*np.sqrt(96*252)) if len(ret)>1 and ret.std()>0 else 0
        dd = (eq - eq.cummax())/eq.cummax(); max_dd = dd.min()
        rng = np.random.default_rng(42); pnls = df["pnl_usd"].to_numpy()
        boot = [self.cfg.initial_equity + rng.choice(pnls, len(pnls), replace=True).sum()
                for _ in range(1000)]
        mc5, mc50, mc95 = np.percentile(boot, [5,50,95])
        return {"trades": n, "wins": len(wins), "losses": len(losses),
                "win_rate_pct": round(wr*100,2), "profit_factor": round(pf,3),
                "sharpe": round(sharpe,2), "max_dd_pct": round(max_dd*100,2),
                "final_equity": round(eq.iloc[-1],2),
                "total_return_pct": round((eq.iloc[-1]/eq.iloc[0]-1)*100,2),
                "mc_p5": round(mc5,2), "mc_p50": round(mc50,2), "mc_p95": round(mc95,2),
                "verdict": self._verdict(sharpe, pf, wr, max_dd)}
    @staticmethod
    def _verdict(sharpe, pf, wr, dd):
        if sharpe >= 1.0 and pf >= 1.5 and wr >= 0.50 and dd > -0.15:
            return "✅ LAYAK LIVE"
        if sharpe >= 0.5 and pf >= 1.2 and wr >= 0.45:
            return "⚠️ MARGINAL"
        return "❌ JANGAN LIVE"


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    log.info("=" * 60)
    log.info("XAUUSD v25.4 — PAXG Live")
    log.info("=" * 60)
    validate_config()
    with acquire_lock() as locked:
        if not locked: return
        state = load_gist_state()
        state = reset_daily_state_if_needed(state)
        state = cleanup_expired_signals(state)

        if is_weekend():
            log.info("Weekend."); save_gist_state(state); return
        if is_friday_cutoff():
            log.info("Friday cutoff."); save_gist_state(state); return
        if check_kill_switch(state):
            save_gist_state(state); return
        if state["daily_signals_count"] >= CONFIG.max_daily_signals:
            log.info("Daily limit."); return
        if state.get("daily_pnl_usd", 0) <= -CONFIG.max_daily_loss_pct * CONFIG.account_equity / 100.0:
            log.info("Daily loss limit."); return
        if state.get("last_signal_time"):
            try:
                mins = (now_utc() - datetime.fromisoformat(state["last_signal_time"])).total_seconds()/60
                if mins < get_session_cooldown():
                    log.info("Cooldown (%.0f).", mins); return
            except Exception: pass
        
        if state.get("last_choppy_time"):
            try:
                mins = (now_utc() - datetime.fromisoformat(state["last_choppy_time"])).total_seconds()/60
                if mins < CONFIG.choppy_cooldown_minutes:
                    log.info("Choppy cooldown (%.0f/%d min).", mins, CONFIG.choppy_cooldown_minutes)
                    return
            except Exception: pass

        if is_high_impact_news_time():
            log.info("News window."); return

        try:
            df, source = fetch_candles()
            df = sanitize_and_trim(df)
            if df is None: return
        except Exception as e:
            log.error("Fetch: %s", e); return
        log.info("✅ Data: %s (%d)", source, len(df))
        state["last_data_source"] = source

        state = evaluate_active_signals(df, state)

        h1 = check_h1_trend_confluence()
        if h1 == "UNKNOWN":
            log.info("H1 unavailable."); save_gist_state(state); return
        log.info("H1: %s | Session: %s", h1, get_current_session())

        try:
            df_feat = calculate_features(df)
        except Exception as e:
            log.error("Features: %s", e); return
        if len(df_feat) < 100: return

        last = df_feat.iloc[-1]
        analyzed_time = last["datetime"].astimezone(CONFIG.wib).strftime("%Y-%m-%d %H:%M WIB")
        log.info("Analyzed: %s | Close: %.2f", analyzed_time, float(last["close"]))

        hurst = float(last["hurst"]); entropy = float(last["shannon_entropy"])
        if entropy > CONFIG.max_entropy_threshold or hurst < CONFIG.min_hurst_threshold:
            state["last_choppy_time"] = now_utc().isoformat()
            save_gist_state(state); return

        ok, vp = check_vol_regime(df_feat)
        if not ok:
            state["last_choppy_time"] = now_utc().isoformat()
            save_gist_state(state); return

        rng_ = float(last["high"]-last["low"]); body = abs(float(last["close"]-last["open"]))
        wick = (rng_-body)/(rng_+1e-12) if rng_ > 0 else 1.0
        if wick > CONFIG.max_wick_ratio_threshold:
            state["last_choppy_time"] = now_utc().isoformat()
            save_gist_state(state); return

        carr = df_feat["close"].to_numpy(); earr = df_feat["ema_trend"].to_numpy()
        above = carr[-1] > earr[-1]; age = 0
        for i in range(len(carr)-1, -1, -1):
            if (carr[i] > earr[i]) == above: age += 1
            else: break
        if age > CONFIG.max_trend_age_bars:
            state["last_choppy_time"] = now_utc().isoformat()
            save_gist_state(state); return

        models = get_or_train_models(df_feat)
        if not models.get("valid"):
            log.info("Model invalid: %s", models.get("reason"))
            save_gist_state(state); return

        state["last_model_metrics"] = {
            "meta_auc": round(models.get("meta_auc",0),3),
            "meta_brier": round(models.get("meta_brier",0),3),
            "wf_auc": round(models.get("wf_auc",0),3),
            "wf_prec": round(models.get("wf_prec",0),3),
            "trained_at": now_utc().isoformat()}

        pred = predict_from_models(df_feat.iloc[[-1]], models)
        side = pred.get("side", 0)
        if side == 0: save_gist_state(state); return
        meta_prob = pred.get("meta_prob", 0.0)
        pb, ps = pred.get("primary_buy_prob",0.0), pred.get("primary_sell_prob",0.0)
        th = state["adaptive_threshold"]
        log.info("Prob: B=%.1f%% S=%.1f%% M=%.1f%% T=%.1f%%",
                 pb*100, ps*100, meta_prob*100, th*100)
        if meta_prob < CONFIG.meta_min_probability: save_gist_state(state); return
        sig = None
        if side == 1 and pb >= th and h1 in ("BUY","NEUTRAL"): sig = "BUY"
        elif side == -1 and ps >= th and h1 in ("SELL","NEUTRAL"): sig = "SELL"
        if not sig: save_gist_state(state); return
        if has_conflicting_signal(state, sig): save_gist_state(state); return

        prob = pb if sig == "BUY" else ps
        trade = compute_trade(sig, df_feat, prob, meta_prob, state, source)
        msg = format_signal_message(trade, h1, hurst, entropy, wick,
                                     analyzed_time, models, source)
        if not send_telegram(msg):
            log.error("Telegram failed."); save_gist_state(state); return
        state["active_signals"].append(trade)
        state["daily_signals_count"] += 1
        state["last_signal_time"] = now_utc().isoformat()
        state["last_choppy_time"] = None
        append_trade_journal(trade, df_feat, state)
        save_gist_state(state)
        log.info("✅ %s sent (source=%s).", sig, source)


def run_backtest_cli(days, equity):
    log.info("Backtest %d days, $%.0f", days, equity)
    df, source = fetch_candles(interval="15m", range_period=f"{days}d")
    log.info("Source: %s", source)
    if df is None or len(df) < CONFIG.min_train_samples + 500:
        log.error("Insufficient."); return
    df_feat = calculate_features(df)
    log.info("Bars: %d", len(df_feat))
    bt = WalkForwardBacktest(df_feat, BacktestConfig(initial_equity=equity))
    report = bt.run()
    print("\n" + "="*60 + "\nBACKTEST (" + source + ")\n" + "="*60)
    print(json.dumps(report, indent=2, default=str))
    print("="*60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="XAUUSD v25.4")
    p.add_argument("--backtest", action="store_true")
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--equity", type=float, default=10000)
    p.add_argument("--monitor", action="store_true")
    p.add_argument("--monitor-check", action="store_true")
    p.add_argument("--monitor-dry-run", action="store_true")
    p.add_argument("--healthcheck", action="store_true")
    p.add_argument("--audit-features", action="store_true")
    p.add_argument("--test-sources", action="store_true")
    a = p.parse_args()
    try:
        if a.test_sources: run_test_sources()
        elif a.audit_features: run_audit_features()
        elif a.backtest: run_backtest_cli(a.days, a.equity)
        elif a.monitor: run_monitor(force=True)
        elif a.monitor_dry_run: run_monitor(force=True, dry_run=True)
        elif a.monitor_check: run_monitor(force=False)
        elif a.healthcheck: run_healthcheck()
        else: main()
    except KeyboardInterrupt: log.warning("Interrupted.")
    except Exception as e:
        log.critical("Fatal: %s", e, exc_info=True); sys.exit(1)
