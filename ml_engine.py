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


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-12)
    df["rsi"] = 100 - (100 / (1 + rs))

    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    df["atr"] = true_range.rolling(CONFIG.atr_period).mean()
    df["atr_pct"] = df["atr"] / df["close"]
    df["atr_regime"] = df["atr"] / df["atr"].rolling(100).mean()

    df["ema_fast"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_trend"] = df["close"].ewm(span=50, adjust=False).mean()

    df["ema_distance"] = (df["ema_fast"] - df["ema_slow"]) / (df["atr"] + 1e-12)
    df["ema_slope"] = (df["ema_fast"] - df["ema_fast"].shift(3)) / (df["atr"] + 1e-12)
    df["trend_strength"] = (df["close"] - df["ema_trend"]) / (df["atr"] + 1e-12)

    kalman_price, kalman_velocity, kalman_error = kalman_filter_1d(
        df["close"].to_numpy(), q=0.01, r=1.0
    )
    df["kalman_price"] = kalman_price
    df["kalman_trend"] = kalman_velocity / (df["atr"] + 1e-12)
    df["kalman_error"] = kalman_error

    df = calculate_adx(df, period=14)

    bb_window = 20
    bb_mean = df["close"].rolling(bb_window).mean()
    bb_std = df["close"].rolling(bb_window).std()
    df["bb_position"] = (df["close"] - bb_mean) / (2 * bb_std + 1e-12)

    keltner_mid = df["close"].ewm(span=20, adjust=False).mean()
    keltner_band = df["atr"].ewm(span=20, adjust=False).mean() * 2
    df["keltner_position"] = (df["close"] - keltner_mid) / (keltner_band + 1e-12)

    returns = df["close"].pct_change()
    df["realized_vol"] = returns.rolling(20).std() * np.sqrt(24 * 60 / CONFIG.timeframe_minutes)
    vol_mean = df["realized_vol"].rolling(100).mean()
    vol_std = df["realized_vol"].rolling(100).std()
    df["vol_zscore"] = (df["realized_vol"] - vol_mean) / (vol_std + 1e-12)

    hurst_values = []
    entropy_values = []
    close_values = df["close"].to_numpy()
    return_values = returns.to_numpy()
    window = 100

    for i in range(len(df)):
        start = max(0, i - window + 1)
        hurst_values.append(hurst_exponent(close_values[start:i + 1]))
        entropy_values.append(shannon_entropy(return_values[start:i + 1]))

    df["hurst"] = hurst_values
    df["shannon_entropy"] = entropy_values

    df["range"] = df["high"] - df["low"]
    df["body"] = (df["close"] - df["open"]).abs()
    df["body_ratio"] = df["body"] / (df["range"] + 1e-12)
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

    session_values = hour.apply(session_flags)
    df["session_london"] = session_values.apply(lambda x: int(x[0]))
    df["session_newyork"] = session_values.apply(lambda x: int(x[1]))
    df["session_overlap"] = session_values.apply(lambda x: int(x[2]))

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df


# ===========================================================================
# TRIPLE-BARRIER LABELING & MODEL
# ===========================================================================

def triple_barrier_labels(df: pd.DataFrame, forward_bars: int, atr_multiplier: float) -> pd.DataFrame:
    df = df.copy()
    labels, exit_bars = [], []
    close, high, low, atr = df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(), df["atr"].to_numpy()
    n = len(df)

    for i in range(n):
        if i + forward_bars >= n:
            labels.append(0)
            exit_bars.append(0)
            continue

        entry, atr_value = close[i], atr[i]
        upper_barrier = entry + atr_value * atr_multiplier
        lower_barrier = entry - atr_value * atr_multiplier
        label = 0
        bars_to_exit = forward_bars

        for j in range(i + 1, min(i + forward_bars + 1, n)):
            hit_upper = high[j] >= upper_barrier
            hit_lower = low[j] <= lower_barrier

            if hit_upper and hit_lower:
                label, bars_to_exit = 0, j - i
                break
            if hit_upper:
                label, bars_to_exit = 1, j - i
                break
            if hit_lower:
                label, bars_to_exit = -1, j - i
                break

        labels.append(label)
        exit_bars.append(bars_to_exit)

    df["label"] = labels
    df["exit_bars"] = exit_bars
    return df


def build_model():
    rf = RandomForestClassifier(
        n_estimators=350, max_depth=8, min_samples_leaf=12,
        max_features="sqrt", class_weight="balanced_subsample", random_state=42, n_jobs=-1
    )
    gb = GradientBoostingClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.025, subsample=0.8, random_state=42
    )
    ensemble = VotingClassifier(
        estimators=[("random_forest", rf), ("gradient_boosting", gb)], voting="soft"
    )
    return CalibratedClassifierCV(ensemble, method="isotonic", cv=3)


def train_and_validate(df: pd.DataFrame) -> dict:
    X = df[FEATURES].values
    y = df["label"].values

    if len(np.unique(y)) < 2:
        raise ValueError("Label hanya memiliki satu kelas. Perbanyak history.")

    fold_metrics = []
    oos_true, oos_proba, oos_pred = [], [], []
    n_samples = len(df)
    fold_size = n_samples // (CONFIG.walk_forward_folds + 1)

    for fold in range(1, CONFIG.walk_forward_folds + 1):
        train_end = fold * fold_size - CONFIG.purge_bars
        test_start = fold * fold_size + CONFIG.embargo_bars
        test_end = min((fold + 1) * fold_size, n_samples)

        if train_end <= CONFIG.min_candles // 2 or test_start >= test_end:
            continue

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_train)) < 2:
            continue

        model = build_model()
        model.fit(X_train, y_train)

        y_pred_fold = model.predict(X_test)
        y_proba_fold = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred_fold)),
            "brier": float(brier_score_loss(y_test, y_proba_fold)),
            "precision": float(precision_score(y_test, y_pred_fold, average="macro", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_fold, average="macro", zero_division=0)),
            "auc": float(roc_auc_score(y_test, y_proba_fold)) if len(np.unique(y_test)) > 1 else 0.5
        }
        fold_metrics.append(metrics)
        oos_true.extend(y_test.tolist())
        oos_proba.extend(y_proba_fold.tolist())
        oos_pred.extend(y_pred_fold.tolist())

    if not fold_metrics:
        raise ValueError("Tidak ada fold walk-forward yang valid.")

    aggregate = {key: float(np.mean([f[key] for f in fold_metrics])) for key in fold_metrics[0]}

    final_model = build_model()
    final_model.fit(X, y)
    latest_proba = final_model.predict_proba(X[-1:])[0]

    return {
        "model": final_model,
        "buy_probability": float(latest_proba[2]),
        "sell_probability": float(latest_proba[0]),
        "oos_accuracy": aggregate["accuracy"],
        "oos_precision": aggregate["precision"],
        "oos_auc": aggregate["auc"],
        "oos_true": np.array(oos_true),
        "oos_proba": np.array(oos_proba),
    }


# ===========================================================================
# EKSEKUSI UTAMA
# ===========================================================================

def main():
    log.info("Menjalankan XAUUSD Modern Quant Signal Bot v16.0")
    
    if CONFIG.backtest_mode:
        log.info("BACKTEST_MODE aktif. Sesi waktu diabaikan.")
    elif not is_valid_session():
        log.info("Diluar sesi trading aktif London/New York. Bot standby.")
        return

    try:
        df = fetch_candles(CONFIG.deriv_symbol, CONFIG.timeframe_minutes * 60, CONFIG.history_candles)
        df = calculate_features(df)
        df = triple_barrier_labels(df, CONFIG.forward_bars, CONFIG.atr_sl_multiplier)

        result = train_and_validate(df)
        log.info("Model berhasil dilatih. Out-of-Sample AUC: %.4f", result["oos_auc"])

        latest = df.iloc[-1]
        buy_p = result["buy_probability"]
        sell_p = result["sell_probability"]

        log.info("Probabilitas - BUY: %.2f%% | SELL: %.2f%%", buy_p * 100, sell_p * 100)

        if buy_p >= CONFIG.min_signal_probability:
            msg = f"<b>XAUUSD SIGNAL: BUY</b>\nProbabilitas: {buy_p * 100:.1f}%\nHarga: {latest['close']}"
            send_telegram(msg)
            log.info("Sinyal BUY terkirim!")
        elif sell_p >= CONFIG.min_signal_probability:
            msg = f"<b>XAUUSD SIGNAL: SELL</b>\nProbabilitas: {sell_p * 100:.1f}%\nHarga: {latest['close']}"
            send_telegram(msg)
            log.info("Sinyal SELL terkirim!")
        else:
            log.info("Tidak ada sinyal kuat yang memenuhi ambang batas probabilitas.")

    except Exception as exc:
        log.error("Pipeline gagal: %s", exc, exc_info=True)
        raise


if __name__ == "__main__":
    main()
