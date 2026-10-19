#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
XAUUSD MODERN QUANT SIGNAL BOT - v16.0 (Production Ready)
=============================================================================
Fitur:
- Kalman Filter price/trend estimate
- Hurst Exponent: trending vs mean-reverting
- Shannon Entropy: kekacauan pasar
- Realized volatility & volatility Z-score
- ADX / +DI / -DI
- Bollinger Band & Keltner Channel position
- Market structure: Higher High / Lower Low / BOS
- Liquidity sweep: sweep prior high/low lalu rejection
- Triple-barrier labeling & Purged walk-forward validation
- OOS expectancy setelah spread + slippage
- Risk-based lot sizing
- Telegram + GitHub Gist state
- BACKTEST_MODE & Robust WebSocket Retry (Anti Cloudflare 520)
=============================================================================
"""

import os
import json
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import websocket

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
)


# ===========================================================================
# KONFIGURASI
# ===========================================================================

@dataclass(frozen=True)
class Config:
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")

    gist_token: str = os.getenv("GH_GIST_TOKEN", "")
    gist_id: str = os.getenv("GH_GIST_ID", "")

    deriv_app_id: str = os.getenv("DERIV_APP_ID", "1089")
    deriv_symbol: str = os.getenv("DERIV_SYMBOL", "XAUUSD")
    backtest_mode: bool = os.getenv("BACKTEST_MODE", "false").lower() == "true"

    timeframe_minutes: int = 15
    history_candles: int = 2500
    min_candles: int = 800

    forward_bars: int = 4
    atr_period: int = 14
    atr_sl_multiplier: float = 1.5
    risk_reward: float = 2.5

    min_signal_probability: float = 0.60
    min_oos_auc: float = 0.55
    min_oos_precision: float = 0.50
    min_expectancy_r: float = 0.05

    atr_regime_min: float = 0.70
    atr_regime_max: float = 2.20
    min_adx: float = 18.0
    max_entropy: float = 0.95

    spread_pips: float = 2.0
    slippage_pips: float = 0.5
    pip_value_usd_per_lot: float = 10.0

    account_equity: float = 1000.0
    risk_percent: float = 0.5
    contract_size: float = 100.0

    cooldown_minutes: int = 120
    max_daily_signals: int = 3
    max_daily_loss_usd: float = 25.0

    purge_bars: int = 4
    embargo_bars: int = 4
    walk_forward_folds: int = 4

    wib: ZoneInfo = ZoneInfo("Asia/Jakarta")


CONFIG = Config()

DERIV_WS_URL = (
    f"wss://ws.derivws.com/websockets/v3?app_id={CONFIG.deriv_app_id}"
)

FEATURES = [
    # Momentum & trend
    "rsi",
    "adx",
    "di_difference",
    "ema_distance",
    "ema_slope",
    "trend_strength",
    "kalman_trend",
    "kalman_error",

    # Volatility & regime
    "atr_pct",
    "atr_regime",
    "realized_vol",
    "vol_zscore",
    "bb_position",
    "keltner_position",

    # Market character
    "hurst",
    "shannon_entropy",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",

    # Structure & liquidity
    "bos_score",
    "liquidity_sweep_score",

    # Return & time
    "return_1",
    "return_3",
    "return_5",
    "hour_sin",
    "hour_cos",
    "dayofweek",
    "session_london",
    "session_newyork",
    "session_overlap",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("xauusd_modern_quant")


# ===========================================================================
# WAKTU & SESI
# ===========================================================================

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_wib() -> datetime:
    return now_utc().astimezone(CONFIG.wib)


def session_flags(hour_utc: int) -> tuple[bool, bool, bool]:
    london = 7 <= hour_utc < 16
    new_york = 12 <= hour_utc < 21
    overlap = 12 <= hour_utc < 16
    return london, new_york, overlap


def is_valid_session() -> bool:
    current = now_wib().time()
    sessions = [
        (time(13, 0), time(17, 0)),  # London, WIB
        (time(19, 0), time(23, 0)),  # New York, WIB
    ]
    return any(start <= current <= end for start, end in sessions)


# ===========================================================================
# GIST STATE
# ===========================================================================

def get_gist_state() -> dict:
    if not CONFIG.gist_token or not CONFIG.gist_id:
        return {}

    url = f"https://api.github.com/gists/{CONFIG.gist_id}"
    headers = {
        "Authorization": f"token {CONFIG.gist_token}",
        "Accept": "application/vnd.github+json",
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        for file_data in response.json().get("files", {}).values():
            content = file_data.get("content")
            if content:
                return json.loads(content)

    except Exception as exc:
        log.warning("Gagal membaca Gist state: %s", exc)

    return {}


def update_gist_state(state: dict) -> None:
    if not CONFIG.gist_token or not CONFIG.gist_id:
        return

    url = f"https://api.github.com/gists/{CONFIG.gist_id}"
    headers = {
        "Authorization": f"token {CONFIG.gist_token}",
        "Accept": "application/vnd.github+json",
    }

    payload = {
        "files": {
            "bot_state.json": {
                "content": json.dumps(state, indent=2),
            }
        }
    }

    try:
        response = requests.patch(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
    except Exception as exc:
        log.warning("Gagal memperbarui Gist state: %s", exc)


# ===========================================================================
# TELEGRAM
# ===========================================================================

def send_telegram(message: str) -> bool:
    if CONFIG.backtest_mode:
        log.info("BACKTEST_MODE aktif. Telegram tidak dikirim.")
        return True

    if not CONFIG.telegram_token or not CONFIG.telegram_chat_id:
        log.error("TELEGRAM_TOKEN atau TELEGRAM_CHAT_ID belum diset.")
        return False

    url = f"https://api.telegram.org/bot{CONFIG.telegram_token}/sendMessage"
    payload = {
        "chat_id": CONFIG.telegram_chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    for attempt in range(1, 4):
        try:
            response = requests.post(url, json=payload, timeout=10)

            if response.status_code == 200:
                return True

            if response.status_code == 429:
                retry_after = response.json().get("parameters", {}).get("retry_after", 5)
                log.warning("Telegram rate limit. Tunggu %s detik.", retry_after)
                time.sleep(int(retry_after) + 1)
                continue

            log.error("Telegram gagal HTTP %s: %s", response.status_code, response.text[:300])
            return False

        except Exception as exc:
            log.warning("Telegram attempt %s gagal: %s", attempt, exc)
            time.sleep(2 * attempt)

    return False


# ===========================================================================
# DERIV PUBLIC DATA (DENGAN RETRY ANTI CLOUDFLARE 520)
# ===========================================================================

def deriv_request(payload: dict, timeout: int = 20, max_retries: int = 4) -> dict:
    for attempt in range(1, max_retries + 1):
        try:
            ws = websocket.create_connection(
                DERIV_WS_URL, 
                timeout=timeout,
                header={"User-Agent": "Mozilla/5.0 (Compatible; XAUUSDQuantBot/16.0)"}
            )
            ws.send(json.dumps(payload))
            response_raw = ws.recv()
            ws.close()

            response = json.loads(response_raw)

            if "error" in response:
                error = response["error"]
                raise RuntimeError(
                    f"Deriv error [{error.get('code')}]: {error.get('message')}"
                )

            return response

        except Exception as exc:
            log.warning("Koneksi Deriv gagal (Attempt %s/%s): %s", attempt, max_retries, exc)
            if attempt == max_retries:
                raise RuntimeError(f"Gagal terhubung ke Deriv API setelah {max_retries} percobaan: {exc}")
            time.sleep(3 * attempt)


def fetch_candles(symbol: str, granularity: int, count: int) -> pd.DataFrame:
    response = deriv_request(
        {
            "ticks_history": symbol,
            "style": "candles",
            "granularity": granularity,
            "count": count,
            "end": "latest",
            "adjust_start_time": 1,
            "req_id": 2,
        }
    )

    candles = response.get("candles", [])

    if len(candles) < CONFIG.min_candles:
        raise ValueError(
            f"Candle tidak cukup: {len(candles)} < {CONFIG.min_candles}"
        )

    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)

    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df[["datetime", "open", "high", "low", "close"]]
    df = df.dropna().sort_values("datetime").reset_index(drop=True)
    return df


# ===========================================================================
# INDIKATOR MODERN
# ===========================================================================

def kalman_filter_1d(values: np.ndarray, q: float = 0.01, r: float = 1.0):
    n = len(values)
    estimate = np.zeros(n)
    velocity = np.zeros(n)
    error = np.zeros(n)

    estimate[0] = values[0]
    velocity[0] = 0.0
    error[0] = 1.0

    for i in range(1, n):
        predicted_estimate = estimate[i - 1] + velocity[i - 1]
        predicted_error = error[i - 1] + q
        kalman_gain = predicted_error / (predicted_error + r)

        estimate[i] = predicted_estimate + kalman_gain * (values[i] - predicted_estimate)
        velocity[i] = 0.85 * velocity[i - 1] + 0.15 * (estimate[i] - estimate[i - 1])
        error[i] = (1 - kalman_gain) * predicted_error

    return estimate, velocity, error


def hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]

    if len(series) < max_lag * 2:
        return 0.5

    lags = range(2, max_lag + 1)
    tau = []

    for lag in lags:
        diff = series[lag:] - series[:-lag]
        tau.append(np.std(diff))

    tau = np.array(tau, dtype=float)
    lags = np.array(list(lags), dtype=float)
    valid = tau > 0

    if valid.sum() < 3:
        return 0.5

    poly = np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)
    return float(np.clip(poly[0], 0.0, 1.0))


def shannon_entropy(returns: np.ndarray, bins: int = 10) -> float:
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]

    if len(returns) < 20:
        return 1.0

    hist, _ = np.histogram(returns, bins=bins)
    probabilities = hist / hist.sum()
    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(probabilities * np.log2(probabilities))
    max_entropy = np.log2(bins)

    return float(entropy / max_entropy) if max_entropy > 0 else 1.0


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    high, low, close = df["high"], df["low"], df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    prev_close = close.shift(1)

    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    atr_wilder = true_range.ewm(alpha=1 / period, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * plus_dm_smooth / (atr_wilder + 1e-12)
    minus_di = 100 * minus_dm_smooth / (atr_wilder + 1e-12)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["di_difference"] = plus_di - minus_di
    df["adx"] = adx
    return df


def calculate_market_structure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    swing_window = 5

    df["prior_high"] = df["high"].rolling(swing_window).max().shift(1)
    df["prior_low"] = df["low"].rolling(swing_window).min().shift(1)

    higher_high = df["high"] > df["prior_high"]
    lower_low = df["low"] < df["prior_low"]

    close_above_prior_high = df["close"] > df["prior_high"]
    close_below_prior_low = df["close"] < df["prior_low"]

    bos_up = higher_high & close_above_prior_high
    bos_down = lower_low & close_below_prior_low

    df["bos_score"] = np.where(bos_up, 1.0, np.where(bos_down, -1.0, 0.0))

    sweep_up = (df["high"] > df["prior_high"]) & (df["close"] < df["prior_high"])
    sweep_down = (df["low"] < df["prior_low"]) & (df["close"] > df["prior_low"])

    df["liquidity_sweep_score"] = np.where(sweep_up, 1.0, np.where(sweep_down, -1.0, 0.0))
    return df


def calculate_features(df: pd.DataFrame) ->
