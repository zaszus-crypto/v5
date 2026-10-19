#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
XAUUSD MODERN QUANT SIGNAL BOT - v17.0 (Production Ready with Gist State & Offset)
=============================================================================
"""

import os
import json
import time as py_time
import logging
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

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

    deriv_symbol: str = os.getenv("DERIV_SYMBOL", "XAUUSD")
    backtest_mode: bool = os.getenv("BACKTEST_MODE", "false").lower() == "true"

    timeframe_minutes: int = 15
    history_candles: int = 2500
    min_candles: int = 100

    forward_bars: int = 4
    atr_period: int = 14
    atr_sl_multiplier: float = 1.5
    risk_reward: float = 2.5

    min_signal_probability: float = 0.60
    min_oos_auc: float = 0.52
    min_oos_precision: float = 0.45

    cooldown_minutes: int = 120
    max_daily_signals: int = 3

    price_offset: float = 30.0  # Penyesuaian basis risk Yahoo Finance vs MT5 Spot

    wib: ZoneInfo = ZoneInfo("Asia/Jakarta")


CONFIG = Config()

FEATURES = [
    "rsi",
    "adx",
    "di_difference",
    "ema_distance",
    "ema_slope",
    "trend_strength",
    "kalman_trend",
    "kalman_error",
    "atr_pct",
    "atr_regime",
    "realized_vol",
    "vol_zscore",
    "bb_position",
    "keltner_position",
    "hurst",
    "shannon_entropy",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "bos_score",
    "liquidity_sweep_score",
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


def is_valid_session() -> bool:
    current = now_wib().time()
    sessions = [
        (time(13, 0), time(17, 0)),  # London, WIB
        (time(19, 0), time(23, 0)),  # New York, WIB
    ]
    return any(start <= current <= end for start, end in sessions)


# ===========================================================================
# GIST STATE MANAGEMENT (BLIND SPOT #2)
# ===========================================================================

def load_gist_state() -> dict:
    today_str = now_wib().strftime("%Y-%m-%d")
    default_state = {
        "last_signal_time": None,
        "daily_signals_count": 0,
        "last_reset_date": today_str
    }
    if not CONFIG.gist_token or not CONFIG.gist_id:
        return default_state

    try:
        headers = {"Authorization": f"token {CONFIG.gist_token}"}
        res = requests.get(f"https://api.github.com/gists/{CONFIG.gist_id}", headers=headers, timeout=10)
        if res.status_code == 200:
            files = res.json().get("files", {})
            for fname, fcontent in files.items():
                if "bot_state.json" in fname or fname == "bot_state.json":
                    state = json.loads(fcontent["content"])
                    if state.get("last_reset_date") != today_str:
                        state["daily_signals_count"] = 0
                        state["last_reset_date"] = today_str
                    return state
    except Exception as e:
        log.warning("Gagal memuat state dari Gist: %s", e)
    return default_state


def save_gist_state(state: dict):
    if not CONFIG.gist_token or not CONFIG.gist_id:
        return
    try:
        headers = {"Authorization": f"token {CONFIG.gist_token}"}
        payload = {
            "files": {
                "bot_state.json": {
                    "content": json.dumps(state, indent=2)
                }
            }
        }
        requests.patch(f"https://api.github.com/gists/{CONFIG.gist_id}", json=payload, headers=headers, timeout=10)
    except Exception as e:
        log.warning("Gagal menyimpan state ke Gist: %s", e)


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
                py_time.sleep(int(retry_after) + 1)
                continue
            return False
        except Exception:
            py_time.sleep(2 * attempt)
    return False


# ===========================================================================
# FETCH CANDLES DENGAN OFFSET (BLIND SPOT #3)
# ===========================================================================

def fetch_candles(symbol: str, count: int) -> pd.DataFrame:
    log.info("Mengambil data market XAUUSD via Yahoo Finance API (dengan offset -%.1f)...", CONFIG.price_offset)
    url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval=15m&range=60d"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    data = response.json()
    
    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    
    df = pd.DataFrame({
        "epoch": timestamps,
        "open": quote["open"],
        "high": quote["high"],
        "low": quote["low"],
        "close": quote["close"]
    })
    
    df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
    
    # Terapkan nilai offset 30 ke bawah agar sesuai dengan harga spot MT5
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce") - CONFIG.price_offset

    df = df[["datetime", "open", "high", "low", "close"]].dropna().sort_values("datetime").reset_index(drop=True)
    return df


# ===========================================================================
# INDIKATOR & MACHINE LEARNING (BLIND SPOT #4 - OOS VALIDATION)
# ===========================================================================

def kalman_filter_1d(values: np.ndarray, q: float = 0.01, r: float = 1.0):
    n = len(values)
    estimate, velocity, error = np.zeros(n), np.zeros(n), np.zeros(n)
    estimate[0], velocity[0], error[0] = values[0], 0.0, 1.0
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
    tau = [np.std(series[lag:] - series[:-lag]) for lag in lags]
    tau, lags_arr = np.array(tau, dtype=float), np.array(list(lags), dtype=float)
    valid = tau > 0
    if valid.sum() < 3:
        return 0.5
    poly = np.polyfit(np.log(lags_arr[valid]), np.log(tau[valid]), 1)
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
    up_move, down_move = high.diff(), -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    prev_close = close.shift(1)
    true_range = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_wilder = true_range.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean() / (atr_wilder + 1e-12)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean() / (atr_wilder + 1e-12)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["di_difference"] = plus_di - minus_di
    df["adx"] = dx.ewm(alpha=1 / period, adjust=False).mean()
    return df


def calculate_market_structure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    swing_window = 5
    df["prior_high"] = df["high"].rolling(swing_window).max().shift(1)
    df["prior_low"] = df["low"].rolling(swing_window).min().shift(1)
    bos_up = (df["high"] > df["prior_high"]) & (df["close"] > df["prior_high"])
    bos_down = (df["low"] < df["prior_low"]) & (df["close"] < df["prior_low"])
    df["bos_score"] = np.where(bos_up, 1.0, np.where(bos_down, -1.0, 0.0))
    sweep_up = (df["high"] > df["prior_high"]) & (df["close"] < df["prior_high"])
    sweep_down = (df["low"] < df["prior_low"]) & (df["close"] > df["prior_low"])
    df["liquidity_sweep_score"] = np.where(sweep_up, 1.0, np.where(sweep_down, -1.0, 0.0))
    return df


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + (gain / (loss + 1e-12))))

    prev_close = df["close"].shift(1)
    true_range = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(CONFIG.atr_period).mean()
    df["atr_pct"] = df["atr"] / df["close"]
    df["atr_regime"] = df["atr"] / df["atr"].rolling(100).mean()

    df["ema_fast"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_trend"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_distance"] = (df["ema_fast"] - df["ema_slow"]) / (df["atr"] + 1e-12)
    df["ema_slope"] = (df["ema_fast"] - df["ema_fast"].shift(3)) / (df["atr"] + 1e-12)
    df["trend_strength"] = (df["close"] - df["ema_trend"]) / (df["atr"] + 1e-12)

    kalman_price, kalman_velocity, kalman_error = kalman_filter_1d(df["close"].to_numpy())
    df["kalman_price"] = kalman_price
    df["kalman_trend"] = kalman_velocity / (df["atr"] + 1e-12)
    df["kalman_error"] = kalman_error

    df = calculate_adx(df)
    bb_mean = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_position"] = (df["close"] - bb_mean) / (2 * bb_std + 1e-12)

    keltner_mid = df["close"].ewm(span=20, adjust=False).mean()
    keltner_band = df["atr"].ewm(span=20, adjust=False).mean() * 2
    df["keltner_position"] = (df["close"] - keltner_mid) / (keltner_band + 1e-12)

    returns = df["close"].pct_change()
    df["realized_vol"] = returns.rolling(20).std() * np.sqrt(24 * 60 / CONFIG.timeframe_minutes)
    df["vol_zscore"] = (df["realized_vol"] - df["realized_vol"].rolling(100).mean()) / (df["realized_vol"].rolling(100).std() + 1e-12)

    close_arr, return_arr = df["close"].to_numpy(), returns.to_numpy()
    df["hurst"] = [hurst_exponent(close_arr[max(0, i - 99):i + 1]) for i in range(len(df))]
    df["shannon_entropy"] = [shannon_entropy(return_arr[max(0, i - 99):i + 1]) for i in range(len(df))]

    df["range"] = df["high"] - df["low"]
    df["body_ratio"] = (df["close"] - df["open"]).abs() / (df["range"] + 1e-12)
    df["upper_wick_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (df["range"] + 1e-12)
    df["lower_wick_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (df["range"] + 1e-12)

    df = calculate_market_structure(df)
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)

    hour = df["datetime"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["session_london"] = hour.apply(lambda h: int(7 <= h < 16))
    df["session_newyork"] = hour.apply(lambda h: int(12 <= h < 21))
    df["session_overlap"] = hour.apply(lambda h: int(12 <= h < 16))

    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def triple_barrier_labels(df: pd.DataFrame, forward_bars: int, atr_multiplier: float) -> pd.DataFrame:
    df = df.copy()
    labels = []
    close, high, low, atr = df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(), df["atr"].to_numpy()
    n = len(df)

    for i in range(n):
        if i + forward_bars >= n:
            labels.append(0)
            continue
        entry, atr_val = close[i], atr[i]
        upper, lower = entry + atr_val * atr_multiplier, entry - atr_val * atr_multiplier
        label = 0
        for j in range(i + 1, min(i + forward_bars + 1, n)):
            if high[j] >= upper:
                label = 1
                break
            if low[j] <= lower:
                label = -1
                break
        labels.append(label)

    df["label"] = labels
    return df


def build_model():
    rf = RandomForestClassifier(n_estimators=350, max_depth=8, min_samples_leaf=12, max_features="sqrt", class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=250, max_depth=3, learning_rate=0.025, subsample=0.8, random_state=42)
    ensemble = VotingClassifier(estimators=[("rf", rf), ("gb", gb)], voting="soft")
    return CalibratedClassifierCV(ensemble, method="isotonic", cv=3)


def train_and_validate(df: pd.DataFrame) -> dict:
    X, y = df[FEATURES].values, df["label"].values
    if len(np.unique(y)) < 2:
        raise ValueError("Label hanya memiliki satu kelas.")

    # Validasi OOS (Out-of-Sample) Check
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    temp_model = build_model()
    temp_model.fit(X_train, y_train)

    oos_auc, oos_prec = 0.5, 0.0
    if len(np.unique(y_test)) >= 2:
        test_proba = temp_model.predict_proba(X_test)
        test_preds = temp_model.predict(X_test)
        try:
            oos_auc = roc_auc_score(y_test, test_proba, multi_class="ovr")
        except Exception:
            pass
        try:
            oos_prec = precision_score(y_test, test_preds, average="macro", zero_division=0)
        except Exception:
            pass

    log.info("Validasi Model OOS - AUC: %.3f | Precision: %.3f", oos_auc, oos_prec)

    if oos_auc < CONFIG.min_oos_auc or oos_prec < CONFIG.min_oos_precision:
        log.warning("Model gagal melewati ambang batas OOS. Sinyal akan diblokir.")
        return {"buy_probability": 0.0, "sell_probability": 0.0, "valid": False}

    # Fit final model dengan seluruh data
    model = build_model()
    model.fit(X, y)
    proba = model.predict_proba(X[-1:])[0]

    return {
        "buy_probability": float(proba[2] if len(proba) > 2 else proba[-1]),
        "sell_probability": float(proba[0]),
        "valid": True
    }


def main():
    log.info("Menjalankan XAUUSD Modern Quant Signal Bot v17.0")
    if not CONFIG.backtest_mode and not is_valid_session():
        log.info("Diluar sesi trading aktif London/New York. Bot standby.")
        return

    # Load State Gist
    state = load_gist_state()
    if state["daily_signals_count"] >= CONFIG.max_daily_signals:
        log.info("Batas maksimal sinyal harian (%d) telah tercapai. Bot standby.", CONFIG.max_daily_signals)
        return

    if state["last_signal_time"]:
        last_dt = datetime.fromisoformat(state["last_signal_time"])
        diff_mins = (now_utc() - last_dt).total_seconds() / 60.0
        if diff_mins < CONFIG.cooldown_minutes:
            log.info("Dalam masa cooldown (%.1f menit berlalu dari %d menit yang ditentukan). Bot standby.", diff_mins, CONFIG.cooldown_minutes)
            return

    df = fetch_candles(CONFIG.deriv_symbol, CONFIG.history_candles)
    df = calculate_features(df)
    df = triple_barrier_labels(df, CONFIG.forward_bars, CONFIG.atr_sl_multiplier)

    res = train_and_validate(df)
    if not res["valid"]:
        log.info("Sinyal dibatalkan karena validasi OOS tidak terpenuhi.")
        return

    buy_p, sell_p = res["buy_probability"], res["sell_probability"]
    log.info("Probabilitas - BUY: %.2f%% | SELL: %.2f%%", buy_p * 100, sell_p * 100)

    signal_sent = False
    if buy_p >= CONFIG.min_signal_probability:
        send_telegram(f"<b>XAUUSD SIGNAL: BUY</b>\nProbabilitas: {buy_p * 100:.1f}%\nOffset Spot: -{CONFIG.price_offset}")
        log.info("Sinyal BUY terkirim!")
        signal_sent = True
    elif sell_p >= CONFIG.min_signal_probability:
        send_telegram(f"<b>XAUUSD SIGNAL: SELL</b>\nProbabilitas: {sell_p * 100:.1f}%\nOffset Spot: -{CONFIG.price_offset}")
        log.info("Sinyal SELL terkirim!")
        signal_sent = True
    else:
        log.info("Tidak ada sinyal kuat yang memenuhi ambang batas probabilitas.")

    if signal_sent:
        state["daily_signals_count"] += 1
        state["last_signal_time"] = now_utc().isoformat()
        save_gist_state(state)


if __name__ == "__main__":
    main()
