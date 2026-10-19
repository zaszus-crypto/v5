#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
XAUUSD MODERN QUANT SIGNAL BOT - v16.0
=============================================================================
Upgrade indikator:
- Kalman Filter price/trend estimate
- Hurst Exponent: trending vs mean-reverting
- Shannon Entropy: kekacauan pasar
- Realized volatility & volatility Z-score
- ADX / +DI / -DI
- Bollinger Band position
- Keltner Channel position
- Market structure: Higher High / Lower Low / BOS
- Liquidity sweep: sweep prior high/low lalu rejection
- Triple-barrier labeling
- Purged walk-forward validation
- OOS expectancy after spread + slippage
- Risk-based lot sizing
- Telegram + GitHub Gist state
- BACKTEST_MODE

INSTALASI:
pip install requests websocket-client numpy pandas scikit-learn

ENVIRONMENT:
export TELEGRAM_TOKEN="..."
export TELEGRAM_CHAT_ID="..."
export GH_GIST_TOKEN="..."       # opsional
export GH_GIST_ID="..."          # opsional
export DERIV_APP_ID="1089"
export DERIV_SYMBOL="XAUUSD"
export BACKTEST_MODE="false"

CATATAN:
- Alat riset/edukasi, bukan financial advice.
- Wajib backtest sebelum live.
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
        response = requests.patch(
            url,
            headers=headers,
            json=payload,
            timeout=10,
        )
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
                retry_after = response.json().get(
                    "parameters", {}
                ).get("retry_after", 5)

                log.warning("Telegram rate limit. Tunggu %s detik.", retry_after)
                time.sleep(int(retry_after) + 1)
                continue

            log.error(
                "Telegram gagal HTTP %s: %s",
                response.status_code,
                response.text[:300],
            )
            return False

        except Exception as exc:
            log.warning("Telegram attempt %s gagal: %s", attempt, exc)
            time.sleep(2 * attempt)

    return False


# ===========================================================================
# DERIV PUBLIC DATA
# ===========================================================================

def deriv_request(payload: dict, timeout: int = 15) -> dict:
    ws = websocket.create_connection(DERIV_WS_URL, timeout=timeout)

    try:
        ws.send(json.dumps(payload))
        response = json.loads(ws.recv())

        if "error" in response:
            error = response["error"]
            raise RuntimeError(
                f"Deriv error [{error.get('code')}]: {error.get('message')}"
            )

        return response

    finally:
        ws.close()


def verify_symbol(symbol: str) -> dict:
    response = deriv_request(
        {
            "active_symbols": "brief",
            "product_type": "basic",
            "req_id": 1,
        }
    )

    for item in response.get("active_symbols", []):
        if item.get("symbol", "").upper() == symbol.upper():
            return {
                "symbol": item.get("symbol"),
                "display_name": item.get("display_name"),
                "market": item.get("market_display_name"),
            }

    raise ValueError(
        f"Simbol '{symbol}' tidak ditemukan pada Deriv active_symbols."
    )


def fetch_candles(
    symbol: str,
    granularity: int,
    count: int,
) -> pd.DataFrame:
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

    required = {"epoch", "open", "high", "low", "close"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Kolom candle Deriv tidak lengkap: {missing}")

    df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)

    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df[["datetime", "open", "high", "low", "close"]]
    df = df.dropna().sort_values("datetime").reset_index(drop=True)

    validate_ohlc(df)

    return df


def validate_ohlc(df: pd.DataFrame) -> None:
    if df["datetime"].duplicated().any():
        raise ValueError("Terdapat candle duplikat.")

    invalid = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )

    if invalid.any():
        raise ValueError("Terdapat candle OHLC tidak valid.")

    time_diff = df["datetime"].diff().dropna()
    expected = pd.Timedelta(minutes=CONFIG.timeframe_minutes)

    irregular = int((time_diff != expected).sum())

    if irregular > 0:
        log.warning(
            "Terdapat %s gap/irregularitas interval candle.",
            irregular,
        )


# ===========================================================================
# INDIKATOR MODERN
# ===========================================================================

def kalman_filter_1d(values: np.ndarray, q: float = 0.01, r: float = 1.0):
    """
    Kalman Filter sederhana untuk smoothing harga dan estimasi tren.
    q = process noise, r = measurement noise.
    """
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

        estimate[i] = (
            predicted_estimate
            + kalman_gain * (values[i] - predicted_estimate)
        )

        velocity[i] = 0.85 * velocity[i - 1] + 0.15 * (
            estimate[i] - estimate[i - 1]
        )

        error[i] = (1 - kalman_gain) * predicted_error

    return estimate, velocity, error


def hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    """
    Estimasi Hurst exponent memakai rescaled range sederhana.
    H > 0.5 cenderung trending, H < 0.5 cenderung mean-reverting.
    """
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
    """
    Shannon entropy dari distribusi return.
    Nilai tinggi = pasar lebih acak/tidak terstruktur.
    """
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


def calculate_adx(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """
    ADX / +DI / -DI dengan Wilder smoothing.
    """
    df = df.copy()

    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where(
        (up_move > down_move) & (up_move > 0),
        up_move,
        0.0,
    )

    minus_dm = np.where(
        (down_move > up_move) & (down_move > 0),
        down_move,
        0.0,
    )

    prev_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
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
    """
    Deteksi sederhana:
    - Higher High / Lower Low
    - Break of Structure (BOS)
    - Liquidity sweep pada prior high/low
    """
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

    df["bos_score"] = np.where(
        bos_up,
        1.0,
        np.where(bos_down, -1.0, 0.0),
    )

    # Liquidity sweep:
    # high menembus prior high tetapi close kembali di bawahnya = sweep bullish trap
    # low menembus prior low tetapi close kembali di atasnya = sweep bearish trap
    sweep_up = (
        (df["high"] > df["prior_high"])
        & (df["close"] < df["prior_high"])
    )

    sweep_down = (
        (df["low"] < df["prior_low"])
        & (df["close"] > df["prior_low"])
    )

    df["liquidity_sweep_score"] = np.where(
        sweep_up,
        1.0,
        np.where(sweep_down, -1.0, 0.0),
    )

    return df


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # === RSI ===
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / (loss + 1e-12)
    df["rsi"] = 100 - (100 / (1 + rs))

    # === ATR ===
    prev_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = true_range.rolling(CONFIG.atr_period).mean()
    df["atr_pct"] = df["atr"] / df["close"]
    df["atr_regime"] = df["atr"] / df["atr"].rolling(100).mean()

    # === EMA ===
    df["ema_fast"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_trend"] = df["close"].ewm(span=50, adjust=False).mean()

    df["ema_distance"] = (
        (df["ema_fast"] - df["ema_slow"]) / (df["atr"] + 1e-12)
    )

    df["ema_slope"] = (
        (df["ema_fast"] - df["ema_fast"].shift(3)) / (df["atr"] + 1e-12)
    )

    df["trend_strength"] = (
        (df["close"] - df["ema_trend"]) / (df["atr"] + 1e-12)
    )

    # === Kalman Filter ===
    kalman_price, kalman_velocity, kalman_error = kalman_filter_1d(
        df["close"].to_numpy(),
        q=0.01,
        r=1.0,
    )

    df["kalman_price"] = kalman_price
    df["kalman_trend"] = kalman_velocity / (df["atr"] + 1e-12)
    df["kalman_error"] = kalman_error

    # === ADX ===
    df = calculate_adx(df, period=14)

    # === Bollinger Bands ===
    bb_window = 20
    bb_mean = df["close"].rolling(bb_window).mean()
    bb_std = df["close"].rolling(bb_window).std()

    df["bb_upper"] = bb_mean + 2 * bb_std
    df["bb_lower"] = bb_mean - 2 * bb_std

    df["bb_position"] = (
        (df["close"] - bb_mean) / (2 * bb_std + 1e-12)
    )

    # === Keltner Channel ===
    keltner_mid = df["close"].ewm(span=20, adjust=False).mean()
    keltner_band = df["atr"].ewm(span=20, adjust=False).mean() * 2

    df["keltner_position"] = (
        (df["close"] - keltner_mid) / (keltner_band + 1e-12)
    )

    # === Realized volatility & Z-score ===
    returns = df["close"].pct_change()

    df["realized_vol"] = returns.rolling(20).std() * np.sqrt(
        24 * 60 / CONFIG.timeframe_minutes
    )

    vol_mean = df["realized_vol"].rolling(100).mean()
    vol_std = df["realized_vol"].rolling(100).std()

    df["vol_zscore"] = (
        (df["realized_vol"] - vol_mean) / (vol_std + 1e-12)
    )

    # === Hurst & Entropy rolling ===
    hurst_values = []
    entropy_values = []

    close_values = df["close"].to_numpy()
    return_values = returns.to_numpy()

    window = 100

    for i in range(len(df)):
        start = max(0, i - window + 1)

        price_window = close_values[start:i + 1]
        return_window = return_values[start:i + 1]

        hurst_values.append(hurst_exponent(price_window))
        entropy_values.append(shannon_entropy(return_window))

    df["hurst"] = hurst_values
    df["shannon_entropy"] = entropy_values

    # === Candle structure ===
    df["range"] = df["high"] - df["low"]
    df["body"] = (df["close"] - df["open"]).abs()

    df["body_ratio"] = df["body"] / (df["range"] + 1e-12)

    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    df["upper_wick_ratio"] = df["upper_wick"] / (df["range"] + 1e-12)
    df["lower_wick_ratio"] = df["lower_wick"] / (df["range"] + 1e-12)

    # === Market structure & liquidity ===
    df = calculate_market_structure(df)

    # === Returns ===
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)

    # === Time ===
    hour = df["datetime"].dt.hour

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dayofweek"] = df["datetime"].dt.dayofweek

    session_values = df["datetime"].dt.hour.apply(session_flags)

    df["session_london"] = session_values.apply(lambda x: int(x[0]))
    df["session_newyork"] = session_values.apply(lambda x: int(x[1]))
    df["session_overlap"] = session_values.apply(lambda x: int(x[2]))

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    return df


# ===========================================================================
# TRIPLE-BARRIER LABELING
# ===========================================================================

def triple_barrier_labels(
    df: pd.DataFrame,
    forward_bars: int,
    atr_multiplier: float,
) -> pd.DataFrame:
    """
    +1 = BUY event
    -1 = SELL event
     0 = timeout / ambiguous double-barrier
    """
    df = df.copy()

    labels = []
    exit_bars = []

    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    atr = df["atr"].to_numpy()

    n = len(df)

    for i in range(n):
        if i + forward_bars >= n:
            labels.append(0)
            exit_bars.append(0)
            continue

        entry = close[i]
        atr_value = atr[i]

        upper_barrier = entry + atr_value * atr_multiplier
        lower_barrier = entry - atr_value * atr_multiplier

        label = 0
        bars_to_exit = forward_bars

        for j in range(i + 1, min(i + forward_bars + 1, n)):
            hit_upper = high[j] >= upper_barrier
            hit_lower = low[j] <= lower_barrier

            if hit_upper and hit_lower:
                label = 0
                bars_to_exit = j - i
                break

            if hit_upper:
                label = 1
                bars_to_exit = j - i
                break

            if hit_lower:
                label = -1
                bars_to_exit = j - i
                break

        labels.append(label)
        exit_bars.append(bars_to_exit)

    df["label"] = labels
    df["exit_bars"] = exit_bars

    return df


# ===========================================================================
# MODEL & PURGED WALK-FORWARD
# ===========================================================================

def build_model():
    rf = RandomForestClassifier(
        n_estimators=350,
        max_depth=8,
        min_samples_leaf=12,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    gb = GradientBoostingClassifier(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.025,
        subsample=0.8,
        random_state=42,
    )

    ensemble = VotingClassifier(
        estimators=[
            ("random_forest", rf),
            ("gradient_boosting", gb),
        ],
        voting="soft",
    )

    return CalibratedClassifierCV(
        ensemble,
        method="isotonic",
        cv=3,
    )


def purged_walk_forward_splits(
    n_samples: int,
    n_splits: int,
    purge_bars: int,
    embargo_bars: int,
):
    fold_size = n_samples // (n_splits + 1)

    for fold in range(1, n_splits + 1):
        train_end = fold * fold_size - purge_bars
        test_start = fold * fold_size + embargo_bars
        test_end = min((fold + 1) * fold_size, n_samples)

        if train_end <= CONFIG.min_candles // 2:
            continue

        if test_start >= test_end:
            continue

        yield (
            np.arange(0, train_end),
            np.arange(test_start, test_end),
        )


def evaluate_fold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_proba)),
    }

    if len(np.unique(y_true)) > 1:
        metrics["precision"] = float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        )

        metrics["recall"] = float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        )

        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["auc"] = 0.5
    else:
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["auc"] = 0.5

    return metrics


def train_and_validate(df: pd.DataFrame) -> dict:
    X = df[FEATURES].values
    y = df["label"].values

    if len(np.unique(y)) < 2:
        raise ValueError(
            "Label hanya memiliki satu kelas. "
            "Perbanyak history atau sesuaikan barrier."
        )

    fold_metrics = []
    oos_true = []
    oos_proba = []
    oos_pred = []

    for train_idx, test_idx in purged_walk_forward_splits(
        n_samples=len(df),
        n_splits=CONFIG.walk_forward_folds,
        purge_bars=CONFIG.purge_bars,
        embargo_bars=CONFIG.embargo_bars,
    ):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_train)) < 2:
            continue

        model = build_model()
        model.fit(X_train, y_train)

        y_pred_fold = model.predict(X_test)
        y_proba_fold = model.predict_proba(X_test)[:, 1]

        metrics = evaluate_fold(y_test, y_pred_fold, y_proba_fold)
        fold_metrics.append(metrics)

        oos_true.extend(y_test.tolist())
        oos_proba.extend(y_proba_fold.tolist())
        oos_pred.extend(y_pred_fold.tolist())

    if not fold_metrics:
        raise ValueError("Tidak ada fold walk-forward yang valid.")

    aggregate = {
        key: float(np.mean([fold[key] for fold in fold_metrics]))
        for key in fold_metrics[0]
    }

    final_model = build_model()
    final_model.fit(X, y)

    latest_proba = final_model.predict_proba(X[-1:])[0]

    # Class order:
    # 0 = SELL (-1)
    # 1 = neutral / timeout (0)
    # 2 = BUY (+1)
    sell_probability = float(latest_proba[0])
    buy_probability = float(latest_proba[2])

    return {
        "model": final_model,
        "buy_probability": buy_probability,
        "sell_probability": sell_probability,
        "oos_accuracy": aggregate["accuracy"],
        "oos_precision": aggregate["precision"],
        "oos_recall": aggregate["recall"],
        "oos_auc": aggregate["auc"],
        "oos_brier": aggregate["brier"],
        "oos_true": np.array(oos_true),
        "oos_proba": np.array(oos_proba),
        "oos_pred": np.array(oos_pred),
    }


# ===========================================================================
# OOS EXPECTANCY SETELAH BIAYA
# ===========================================================================

def calculate_cost_r() -> float:
    total_cost_pips = CONFIG.spread_pips + CONFIG.slippage_pips
    cost_usd_per_lot = total_cost_pips * CONFIG.pip_value_usd_per_lot

    risk_usd = CONFIG.account_equity * CONFIG.risk_percent / 100

    if risk_usd <= 0:
        return 999.0

    return cost_usd_per_lot / risk_usd


def calculate_oos_expectancy(result: dict) -> float:
    y_true = result["oos_true"]
    y_proba = result["oos_proba"]

    if len(y_true) == 0:
        return -999.0

    cost_r = calculate_cost_r()
    tp_r = CONFIG.risk_reward
    sl_r = 1.0

    buy_mask = y_proba >= CONFIG.min_signal_probability
    sell_mask = y_proba <= (1 - CONFIG.min_signal_probability)

    buy_expectancy = -999.0
    sell_expectancy = -999.0

    if buy_mask.sum() >= 20:
        buy_win_rate = float(np.mean(y_true[buy_mask] == 1))
        buy_expectancy = (
            buy_win_rate * tp_r
            - (1 - buy_win_rate) * sl_r
            - cost_r
        )

    if sell_mask.sum() >= 20:
        sell_win_rate = float(np.mean(y_true[sell_mask] == -1))
        sell_expectancy = (
            sell_win_rate * tp_r
            - (1 - sell_win_rate) * sl_r
            - cost_r
        )

    return max(buy_expectancy, sell_expectancy)


# ===========================================================================
# RISK & POSITION SIZING
# ===========================================================================

def calculate_lot_size(entry: float, stop_loss: float) -> float:
    risk_amount = CONFIG.account_equity * CONFIG.risk_percent / 100
    stop_distance = abs(entry - stop_loss)

    if stop_distance <= 0:
        return 0.0

    dollar_risk_per_lot = stop_distance * CONFIG.contract_size

    if dollar_risk_per_lot <= 0:
        return 0.0

    lot = risk_amount / dollar_risk_per_lot

    return max(0.0, round(lot, 2))


def cooldown_active(state: dict) -> bool:
    last_signal_time = state.get("last_signal_time")

    if not last_signal_time:
        return False

    try:
        last_time = datetime.fromisoformat(last_signal_time)
    except ValueError:
        return False

    return now_utc() - last_time < timedelta(
        minutes=CONFIG.cooldown_minutes
    )


def daily_signal_limit_reached(state: dict) -> bool:
    today = now_utc().date().isoformat()
    daily_signals = state.get("daily_signals", {})

    return int(daily_signals.get(today, 0)) >= CONFIG.max_daily_signals


def daily_loss_limit_reached(state: dict) -> bool:
    today = now_utc().date().isoformat()
    daily_pnl = float(state.get("daily_pnl", {}).get(today, 0.0))

    return daily_pnl <= -abs(CONFIG.max_daily_loss_usd)


# ===========================================================================
# SINYAL MODERN
# ===========================================================================

def determine_signal(
    result: dict,
    latest: pd.Series,
) -> tuple[str | None, float]:
    buy_probability = result["buy_probability"]
    sell_probability = result["sell_probability"]

    trend_up = (
        latest["ema_fast"] > latest["ema_slow"]
        and latest["ema_slow"] > latest["ema_trend"]
        and latest["trend_strength"] > 0
        and latest["kalman_trend"] > 0
        and latest["plus_di"] > latest["minus_di"]
    )

    trend_down = (
        latest["ema_fast"] < latest["ema_slow"]
        and latest["ema_slow"] < latest["ema_trend"]
        and latest["trend_strength"] < 0
        and latest["kalman_trend"] < 0
        and latest["minus_di"] > latest["plus_di"]
    )

    trend_strength_ok = latest["adx"] >= CONFIG.min_adx

    volatility_normal = (
        CONFIG.atr_regime_min
        <= latest["atr_regime"]
        <= CONFIG.atr_regime_max
    )

    market_not_too_random = (
        latest["shannon_entropy"] <= CONFIG.max_entropy
    )

    model_quality_ok = (
        result["oos_auc"] >= CONFIG.min_oos_auc
        and result["oos_precision"] >= CONFIG.min_oos_precision
        and result["expectancy_r"] >= CONFIG.min_expectancy_r
    )

    if not model_quality_ok:
        return None, 0.0

    if not volatility_normal:
        return None, 0.0

    if not market_not_too_random:
        return None, 0.0

    if not trend_strength_ok:
        return None, 0.0

    if (
        buy_probability >= CONFIG.min_signal_probability
        and trend_up
        and buy_probability > sell_probability
    ):
        return "BUY", buy_probability

    if (
        sell_probability >= CONFIG.min_signal_probability
        and trend_down
        and sell_probability > buy_probability
    ):
        return "SELL", sell_probability

    return None, 0.0


# ===========================================================================
# PESAN TELEGRAM
# ===========================================================================

def build_signal_message(
    direction: str,
    probability: float,
    result: dict,
    entry: float,
    sl: float,
    tp: float,
    lot: float,
) -> str:
    latest = result["latest"]
    current_time = now_wib()

    return (
        "<b>XAUUSD MODERN QUANT SIGNAL</b>\n"
        "<i>Deriv Public Market Data</i>\n\n"
        f"<b>Direction:</b> {direction}\n"
        f"<b>Entry:</b> <code>{entry:.2f}</code>\n"
        f"<b>Stop Loss:</b> <code>{sl:.2f}</code>\n"
        f"<b>Take Profit:</b> <code>{tp:.2f}</code>\n"
        f"<b>Suggested lot:</b> <code>{lot:.2f}</code>\n\n"
        f"<b>Signal probability:</b> <code>{probability:.1%}</code>\n"
        f"<b>BUY probability:</b> <code>{result['buy_probability']:.1%}</code>\n"
        f"<b>SELL probability:</b> <code>{result['sell_probability']:.1%}</code>\n\n"
        f"<b>OOS AUC:</b> <code>{result['oos_auc']:.3f}</code>\n"
        f"<b>OOS Precision:</b> <code>{result['oos_precision']:.1%}</code>\n"
        f"<b>OOS Accuracy:</b> <code>{result['oos_accuracy']:.1%}</code>\n"
        f"<b>OOS Brier:</b> <code>{result['oos_brier']:.3f}</code>\n"
        f"<b>OOS Expectancy:</b> <code>{result['expectancy_r']:.2f}R</code>\n\n"
        f"<b>ADX:</b> <code>{latest['adx']:.2f}</code>\n"
        f"<b>+DI:</b> <code>{latest['plus_di']:.2f}</code>\n"
        f"<b>-DI:</b> <code>{latest['minus_di']:.2f}</code>\n"
        f"<b>Hurst:</b> <code>{latest['hurst']:.3f}</code>\n"
        f"<b>Entropy:</b> <code>{latest['shannon_entropy']:.3f}</code>\n"
        f"<b>Kalman trend:</b> <code>{latest['kalman_trend']:.3f}</code>\n"
        f"<b>ATR regime:</b> <code>{latest['atr_regime']:.2f}</code>\n"
        f"<b>BOS score:</b> <code>{latest['bos_score']:.1f}</code>\n"
        f"<b>Liquidity sweep:</b> <code>{latest['liquidity_sweep_score']:.1f}</code>\n\n"
        f"<b>Timeframe:</b> M{CONFIG.timeframe_minutes}\n"
        f"<b>Time:</b> {current_time.strftime('%Y-%m-%d %H:%M:%S')} WIB\n\n"
        "<i>Educational signal. Not financial advice.</i>"
    )


# ===========================================================================
# BACKTEST REPORT
# ===========================================================================

def run_backtest_report(result: dict) -> str:
    y_true = result["oos_true"]
    y_proba = result["oos_proba"]

    buy_mask = y_proba >= CONFIG.min_signal_probability
    sell_mask = y_proba <= (1 - CONFIG.min_signal_probability)

    buy_signals = int(buy_mask.sum())
    sell_signals = int(sell_mask.sum())

    buy_wins = int(np.sum(y_true[buy_mask] == 1)) if buy_signals else 0
    sell_wins = int(np.sum(y_true[sell_mask] == -1)) if sell_signals else 0

    total_signals = buy_signals + sell_signals
    total_wins = buy_wins + sell_wins

    win_rate = total_wins / total_signals if total_signals else 0.0

    return (
        "<b>HASIL BACKTEST MODERN QUANT</b>\n\n"
        f"Total OOS samples: <code>{len(y_true)}</code>\n"
        f"BUY signals: <code>{buy_signals}</code>\n"
        f"SELL signals: <code>{sell_signals}</code>\n"
        f"Qualified signals: <code>{total_signals}</code>\n"
        f"Total wins: <code>{total_wins}</code>\n"
        f"Win rate: <code>{win_rate:.1%}</code>\n"
        f"OOS AUC: <code>{result['oos_auc']:.3f}</code>\n"
        f"OOS Precision: <code>{result['oos_precision']:.1%}</code>\n"
        f"OOS Brier: <code>{result['oos_brier']:.3f}</code>\n"
        f"Estimated expectancy: <code>{result['expectancy_r']:.2f}R</code>\n\n"
        "<i>Backtest awal. Belum memodelkan partial fill, news gap, "
        "atau spread ekstrem.</i>"
    )


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    log.info("Menjalankan XAUUSD Modern Quant Signal Bot v16.0")

    if CONFIG.backtest_mode:
        log.info("BACKTEST_MODE aktif. Sesi waktu diabaikan.")
    elif not is_valid_session():
        log.info("Di luar sesi London/New York. Standby.")
        return

    state = get_gist_state()

    if not CONFIG.backtest_mode:
        if cooldown_active(state):
            log.info("Cooldown aktif.")
            return

        if daily_signal_limit_reached(state):
            log.info("Limit sinyal harian tercapai.")
            return

        if daily_loss_limit_reached(state):
            log.info("Daily loss limit tercapai.")
            return

    try:
        symbol_info = verify_symbol(CONFIG.deriv_symbol)
        log.info(
            "Simbol valid: %s | %s | %s",
            symbol_info["symbol"],
            symbol_info["display_name"],
            symbol_info["market"],
        )

        raw_df = fetch_candles(
            symbol=CONFIG.deriv_symbol,
            granularity=CONFIG.timeframe_minutes * 60,
            count=CONFIG.history_candles,
        )

        featured_df = calculate_features(raw_df)

        labeled_df = triple_barrier_labels(
            df=featured_df,
            forward_bars=CONFIG.forward_bars,
            atr_multiplier=CONFIG.atr_sl_multiplier,
        )

        result = train_and_validate(labeled_df)

        result["df"] = labeled_df
        result["latest"] = labeled_df.iloc[-1]
        result["expectancy_r"] = calculate_oos_expectancy(result)

    except Exception as exc:
        log.error("Pipeline gagal: %s", exc)
        return

    if CONFIG.backtest_mode:
        report = run_backtest_report(result)
        log.info("\n%s", report)
        send_telegram(report)
        return

    latest = result["latest"]

    direction, probability = determine_signal(result, latest)

    if direction is None:
        log.info(
            "Tidak ada sinyal valid. BUY=%.3f SELL=%.3f ADX=%.2f "
            "Hurst=%.3f Entropy=%.3f AUC=%.3f Exp=%.2fR",
            result["buy_probability"],
            result["sell_probability"],
            latest["adx"],
            latest["hurst"],
            latest["shannon_entropy"],
            result["oos_auc"],
            result["expectancy_r"],
        )
        return

    entry = float(latest["close"])
    atr = float(latest["atr"])

    sl_distance = atr * CONFIG.atr_sl_multiplier
    tp_distance = sl_distance * CONFIG.risk_reward

    if direction == "BUY":
        sl = entry - sl_distance
        tp = entry + tp_distance
    else:
        sl = entry + sl_distance
        tp = entry - tp_distance

    lot = calculate_lot_size(entry=entry, stop_loss=sl)

    if lot <= 0:
        log.warning("Lot size tidak valid. Sinyal dibatalkan.")
        return

    message = build_signal_message(
        direction=direction,
        probability=probability,
        result=result,
        entry=entry,
        sl=sl,
        tp=tp,
        lot=lot,
    )

    if not send_telegram(message):
        log.error("Sinyal gagal dikirim.")
        return

    today = now_utc().date().isoformat()

    daily_signals = state.get("daily_signals", {})
    daily_signals[today] = int(daily_signals.get(today, 0)) + 1

    state.update(
        {
            "last_signal_time": now_utc().isoformat(),
            "last_direction": direction,
            "last_entry": entry,
            "last_sl": sl,
            "last_tp": tp,
            "last_lot": lot,
            "last_probability": probability,
            "last_expectancy_r": result["expectancy_r"],
            "daily_signals": daily_signals,
        }
    )

    update_gist_state(state)

    log.info(
        "Sinyal terkirim: %s | Entry=%.2f SL=%.2f TP=%.2f "
        "Lot=%.2f Prob=%.1f%% Exp=%.2fR",
        direction,
        entry,
        sl,
        tp,
        lot,
        probability * 100,
        result["expectancy_r"],
    )


if __name__ == "__main__":
    main()
