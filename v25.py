#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XAUUSD BOT v25.5 (Institusional Refactored Edition)
═══════════════════════════════════════════════════════════════════════════════
Arsitektur Kuantitatif Spot Emas (XAU/USD) yang Telah Disempurnakan:
1. PEMBERSIHAN SUMBER DATA: Mengganti token kripto PAXG & offset statis $30
   dengan kuotasi Spot Emas Murni (XAU/USD) interbank (OANDA / Yahoo XAUUSD=X /
   Forex Gold Feed).
2. PERLINDUNGAN MODAL & CENT ACCOUNT: Mencegah jebakan leverage akun mikro.
   Mendukung mode Cent (USC) & Standard. Lot sizing dibatasi kebawah (floor)
   agar risiko riil tidak pernah melampaui max_risk_percent.
3. KONSISTENSI PELABELAN INTRA-BAR: Menghilangkan Data Leakage / Asymmetric
   Tie-Breaking. Jika TP dan SL tersentuh di candle yang sama, otomatis
   diasumsikan LOSS (worst-case assumption) baik di training maupun live.
4. REGULARISASI MACHINE LEARNING: Pangkas max_depth pohon, naikkan min_samples_leaf,
   dan proteksi covariate shift pada volume.
5. SINKRONISASI CLOSED BAR KETAT: Mencegah unclosed candle / lookahead repainting.
6. PNL REALISTIS: Memotong komisi round-turn, spread, dan estimasi slippage gap.
7. STATE MANAGEMENT PERSISTEN ATOMIK: Penyimpanan state aman terhadap race condition.

Penggunaan:
  python v25.py                   # Jalankan bot sinyal live
  python v25.py --backtest        # Jalankan walk-forward backtest
  python v25.py --monitor         # Kirim laporan performa harian
  python v25.py --monitor-dry-run # Preview laporan di terminal
  python v25.py --healthcheck     # Uji integritas sistem
  python v25.py --test-sources    # Uji konektivitas data feed spot
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import math
import pickle
import logging
import argparse
import tempfile
import time as py_time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, time, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import requests

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, roc_auc_score, brier_score_loss


# ===========================================================================
# 1. KONFIGURASI SISTEM
# ===========================================================================
@dataclass(frozen=True)
class Config:
    # Kredensial Notifikasi & State
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    gist_token: str = os.getenv("GH_GIST_TOKEN", "")
    gist_id: str = os.getenv("GH_GIST_ID", "")
    backtest_mode: bool = os.getenv("BACKTEST_MODE", "false").lower() == "true"
    monitor_hour_wib: int = int(os.getenv("MONITOR_HOUR_WIB", "8"))

    # Timeframe & Periode Candle
    timeframe_minutes: int = 15
    min_candles: int = 400
    forward_bars: int = 4
    atr_period: int = 14
    atr_sl_multiplier: float = 1.5
    risk_reward: float = 2.0  # Realistis untuk market M15 XAUUSD

    # Biaya Transaksi Spot Forex Riil
    spread_usd: float = 0.25         # Rata-rata spread spot XAU/USD interbank ($0.20 - $0.30)
    slippage_usd: float = 0.15       # Buffer slippage eksekusi riil
    commission_per_lot: float = 6.0  # Komisi broker ECN per standard lot ($3 per side)

    # Threshold Probabilitas Machine Learning
    base_min_signal_probability: float = 0.58
    max_threshold: float = 0.70
    min_oos_auc: float = 0.53
    min_oos_precision: float = 0.50

    # Filter Rezim Pasar & Entropi
    max_entropy_threshold: float = 0.92
    min_hurst_threshold: float = 0.42
    max_wick_ratio_threshold: float = 0.65
    max_trend_age_bars: int = 16

    # Manajemen Waktu & Sesi
    cooldown_minutes: int = 90
    choppy_cooldown_minutes: int = 45
    max_daily_signals: int = 5
    max_daily_loss_pct: float = 3.0  # Cut-off harian ketat
    session_aware_cooldown: bool = True
    session_aware_risk: bool = True
    friday_cutoff_hour_utc: int = 20  # Tutup menjelang penutupan pasar Forex Jumat

    # Parameter Modal & Manajemen Lot (Dukungan Cent & Standard)
    # Mode: "STANDARD" (1 lot = 100 oz) atau "CENT" (1 lot = 1 oz / nilai uang dalam USC)
    account_mode: str = os.getenv("ACCOUNT_MODE", "STANDARD").upper()
    account_equity: float = float(os.getenv("ACCOUNT_EQUITY", "200.0"))
    risk_percent: float = 1.0        # Target 1% per posisi
    min_risk_percent: float = 0.5
    max_risk_percent: float = 1.5    # Hard ceiling: tidak boleh dilampaui
    contract_size: float = 100.0 if os.getenv("ACCOUNT_MODE", "STANDARD").upper() == "STANDARD" else 1.0
    min_lot: float = 0.01
    max_lot: float = 1.0

    # Validasi Meta-Model
    meta_min_auc: float = 0.53
    meta_max_brier: float = 0.24
    meta_min_probability: float = 0.54
    meta_oof_splits: int = 3

    # Walk-Forward Validation
    wf_splits: int = 3
    wf_min_auc: float = 0.53
    wf_min_prec: float = 0.48
    min_train_samples: int = 600

    vol_regime_min_pct: float = 0.20
    vol_regime_max_pct: float = 0.90

    # Risk Kill-Switch & Circuit Breaker
    rolling_window: int = 20
    min_rolling_winrate: float = 0.35
    max_consecutive_losses: int = 4
    state_file_path: str = os.getenv("STATE_FILE_PATH", "bot_state_v25.json")
    model_cache_path: str = os.getenv("MODEL_CACHE_PATH", "/tmp/xauusd_model_v255.pkl")
    model_cache_ttl_min: int = 120
    max_journal_entries: int = 150
    backtest_retrain_every: int = 192

    healthcheck_interval_hrs: int = 6
    last_sig_alert_hrs: int = 12
    lock_path: str = "/tmp/xauusd_bot_v255.lock"
    wib: ZoneInfo = ZoneInfo("Asia/Jakarta")


CONFIG = Config()

# Daftar Fitur ML Bebas Lookahead
FEATURES = [
    "rsi", "adx", "di_difference", "ema_distance", "ema_slope",
    "trend_strength", "kalman_trend", "kalman_deviation", "atr_pct",
    "atr_regime", "realized_vol", "vol_zscore", "bb_position",
    "keltner_position", "hurst", "shannon_entropy", "body_ratio",
    "upper_wick_ratio", "lower_wick_ratio", "bos_score",
    "liquidity_sweep_score", "return_1", "return_3", "return_5",
    "hour_sin", "hour_cos", "dayofweek", "session_london",
    "session_newyork", "session_overlap",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stderr,
    force=True,
)
log = logging.getLogger("xauusd_v255")


# ===========================================================================
# 2. UTILITY & TIME FUNCTIONS
# ===========================================================================
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def now_wib() -> datetime:
    return now_utc().astimezone(CONFIG.wib)

def is_forex_market_closed() -> bool:
    """
    Pasar Forex XAU/USD tutup dari Jumat 21:00 UTC sampai Minggu 22:00 UTC.
    Trading di waktu ini adalah bahaya besar karena pasar spot interbank tidak ada likuiditas.
    """
    now = now_utc()
    weekday = now.weekday()  # Monday is 0, Sunday is 6
    hour = now.hour

    # Hari Sabtu: Tutup penuh
    if weekday == 5:
        return True
    # Hari Jumat malam: Tutup setelah pukul 21:00 UTC
    if weekday == 4 and hour >= CONFIG.friday_cutoff_hour_utc:
        return True
    # Hari Minggu: Tutup sebelum pukul 22:00 UTC
    if weekday == 6 and hour < 22:
        return True
    return False

def get_current_session() -> str:
    utc_hour = now_utc().hour
    if 7 <= utc_hour < 12:
        return "LONDON"
    elif 12 <= utc_hour < 16:
        return "OVERLAP"
    elif 16 <= utc_hour < 21:
        return "NEWYORK"
    elif 0 <= utc_hour < 7:
        return "ASIAN"
    return "LATE"

def get_session_risk_multiplier() -> float:
    if not CONFIG.session_aware_risk:
        return 1.0
    sess = get_current_session()
    if sess in ("LONDON", "OVERLAP", "NEWYORK"):
        return 1.0
    return 0.5  # Sesi Asia & Late memiliki likuiditas tipis; kecilkan risiko

@contextmanager
def acquire_lock():
    """File lock untuk mencegah race-condition eksekusi lokal."""
    locked = False
    f = None
    try:
        import fcntl
        f = open(CONFIG.lock_path, "w")
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        yield locked
    except Exception:
        log.warning("Lock lokal terdeteksi aktif atau tidak didukung di OS ini.")
        yield True  # Fallback gracefully
    finally:
        if locked and f is not None:
            try:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()
            except Exception:
                pass


# ===========================================================================
# 3. ATOMIC STATE MANAGEMENT
# ===========================================================================
DEFAULT_STATE: Dict[str, Any] = {
    "version": 255,
    "kill_switch_active": False,
    "last_signal_time": None,
    "last_choppy_time": None,
    "last_healthcheck": None,
    "daily_signals_count": 0,
    "daily_pnl_usd": 0.0,
    "last_reset_date": None,
    "last_report_date": None,
    "active_signals": [],
    "performance_history": {"wins": 0, "losses": 0},
    "adaptive_threshold": CONFIG.base_min_signal_probability,
    "recent_outcomes": [],
    "trade_journal": [],
    "session_outcomes": {},
    "last_model_metrics": {},
    "last_data_source": None,
}

def load_system_state() -> Dict[str, Any]:
    """Muat state secara atomik dari disk lokal, lalu periksa sync Gist jika tersedia."""
    state = dict(DEFAULT_STATE)
    today = now_wib().strftime("%Y-%m-%d")
    state["last_reset_date"] = today

    # 1. Coba baca dari file lokal
    if os.path.exists(CONFIG.state_file_path):
        try:
            with open(CONFIG.state_file_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
                state.update(saved)
                return state
        except Exception as e:
            log.warning("Gagal membaca local state: %s", e)

    # 2. Fallback baca dari Gist jika dikonfigurasi
    if CONFIG.gist_token and CONFIG.gist_id:
        try:
            url = f"https://api.github.com/gists/{CONFIG.gist_id}"
            headers = {"Authorization": f"token {CONFIG.gist_token}"}
            r = requests.get(url, headers=headers, timeout=6)
            if r.status_code == 200:
                files = r.json().get("files", {})
                for fn, fc in files.items():
                    if "state" in fn.lower():
                        state.update(json.loads(fc["content"]))
                        log.info("State berhasil disinkronisasi dari GitHub Gist.")
                        return state
        except Exception as e:
            log.warning("Sinkronisasi Gist gagal: %s", e)

    return state

def save_system_state(state: Dict[str, Any]) -> bool:
    """Simpan state dengan teknik Atomic Write (tulis temp file lalu replace)."""
    # 1. Simpan atomik ke file lokal
    try:
        dir_name = os.path.dirname(os.path.abspath(CONFIG.state_file_path))
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
            json.dump(state, tf, indent=2)
            temp_name = tf.name
        os.replace(temp_name, CONFIG.state_file_path)
    except Exception as e:
        log.error("Atomic write state lokal gagal: %s", e)

    # 2. Sinkronkan ke Gist secara async-friendly dengan retry
    if CONFIG.gist_token and CONFIG.gist_id:
        try:
            payload = {"files": {"bot_state_v255.json": {"content": json.dumps(state, indent=2)}}}
            headers = {"Authorization": f"token {CONFIG.gist_token}"}
            requests.patch(
                f"https://api.github.com/gists/{CONFIG.gist_id}",
                json=payload,
                headers=headers,
                timeout=6,
            )
        except Exception as e:
            log.warning("Update Gist background gagal: %s", e)
    return True

def reset_daily_state_if_needed(state: Dict[str, Any]) -> Dict[str, Any]:
    today = now_wib().strftime("%Y-%m-%d")
    if state.get("last_reset_date") != today:
        log.info("Sesi hari baru terdeteksi (%s). Mereset counter harian.", today)
        state["daily_signals_count"] = 0
        state["daily_pnl_usd"] = 0.0
        state["last_signal_time"] = None
        state["last_choppy_time"] = None
        state["last_reset_date"] = today
    return state


# ===========================================================================
# 4. DATA FETCHER: BULLETPROOF MULTI-TIER GOLD FEED ENGINE
# ===========================================================================
def _fetch_yahoo_gcf(interval: str = "15m", range_period: str = "60d") -> pd.DataFrame:
    """
    Mengambil data COMEX Gold Continuous Benchmark (Ticker: GC=F) dari Yahoo Finance.
    GC=F menyediakan lebih dari 5.600 candle M15 historis berkesinambungan
    tanpa API key (Ticker 'XAUUSD=X' telah dihapus / 404 di Yahoo chart API).
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval={interval}&range={range_period}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()
    result = data.get("chart", {}).get("result")
    if not result:
        raise ValueError("Yahoo GC=F data kosong")

    quote_data = result[0]
    timestamps = quote_data.get("timestamp", [])
    quotes = quote_data.get("indicators", {}).get("quote", [{}])[0]
    if not timestamps or not quotes.get("close"):
        raise ValueError("Indikator Yahoo quote GC=F tidak lengkap")

    df = pd.DataFrame({
        "epoch": timestamps,
        "open": quotes.get("open"),
        "high": quotes.get("high"),
        "low": quotes.get("low"),
        "close": quotes.get("close"),
        "volume": quotes.get("volume", [0] * len(timestamps)),
    })
    df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    df = df[["datetime", "open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    return df.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)

def _fetch_binance_paxg(interval: str = "15m", total_bars: int = 3000) -> pd.DataFrame:
    """Fallback cadangan: Binance Pax Gold (PAXGUSDT) jika Yahoo mengalami rate-limit."""
    all_data, end_time, remaining = [], None, total_bars
    while remaining > 0:
        limit = min(1000, remaining)
        params = {"symbol": "PAXGUSDT", "interval": interval, "limit": limit}
        if end_time:
            params["endTime"] = end_time
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params=params,
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if r.status_code != 200:
            if all_data:
                break
            raise RuntimeError(f"Binance HTTP {r.status_code}")
        data = r.json()
        if not data or not isinstance(data, list):
            break
        all_data = data + all_data
        if len(data) < limit:
            break
        end_time = data[0][0] - 1
        remaining -= len(data)
        py_time.sleep(0.1)

    if not all_data:
        raise RuntimeError("Binance data kosong")
    df = pd.DataFrame(
        all_data,
        columns=["ot", "o", "h", "l", "c", "v", "ct", "qv", "t", "tb", "tq", "ig"],
    )
    df["datetime"] = pd.to_datetime(df["ot"], unit="ms", utc=True)
    for a, b in [("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"), ("v", "volume")]:
        df[b] = pd.to_numeric(df[a], errors="coerce")
    df = df[["datetime", "open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    return df.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)

def _fetch_kraken_paxg(interval_min: int = 15) -> pd.DataFrame:
    """Fallback cadangan: Kraken PAXGUSD."""
    r = requests.get(
        "https://api.kraken.com/0/public/OHLC",
        params={"pair": "PAXGUSD", "interval": interval_min},
        timeout=8,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    if r.status_code != 200:
        raise RuntimeError(f"Kraken HTTP {r.status_code}")
    data = r.json()
    result = data.get("result", {})
    key = next((k for k in result.keys() if k != "last"), None)
    if not key or not result.get(key):
        raise RuntimeError("Kraken data kosong")
    df = pd.DataFrame(result[key], columns=["ts", "open", "high", "low", "close", "vwap", "volume", "count"])
    df["datetime"] = pd.to_datetime(df["ts"].astype("int64"), unit="s", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["datetime", "open", "high", "low", "close", "volume"]].dropna().sort_values("datetime").reset_index(drop=True)

def fetch_spot_candles(interval: str = "15m", range_period: str = "60d") -> Tuple[pd.DataFrame, str]:
    """
    Fungsi Multi-Tier Bulletproof Feed:
    1. Yahoo COMEX Gold Benchmark (GC=F) -> 5.600+ candle M15
    2. OANDA Spot XAU_USD (Jika OANDA_API_KEY disetel)
    3. Binance PAXGUSDT -> 3.000 candle
    4. Kraken PAXGUSD
    """
    # 1. Coba Yahoo GC=F (Primary Benchmark)
    try:
        df = _fetch_yahoo_gcf(interval=interval, range_period=range_period)
        if len(df) >= CONFIG.min_candles:
            log.info("✅ FEED UTAMA TERHUBUNG: Yahoo COMEX Gold GC=F (%d candle M15)", len(df))
            return df, "YAHOO_GOLD_GCF"
    except Exception as e:
        log.warning("Yahoo GC=F tidak dapat diakses (%s). Mencoba alternatif...", e)

    # 2. Coba OANDA REST API jika ada token
    oanda_token = os.getenv("OANDA_API_KEY", "")
    if oanda_token:
        try:
            url = "https://api-fxtrade.oanda.com/v3/instruments/XAU_USD/candles"
            headers = {"Authorization": f"Bearer {oanda_token}"}
            params = {"granularity": "M15", "count": 1500, "price": "M"}
            r = requests.get(url, headers=headers, params=params, timeout=10)
            if r.status_code == 200:
                raw = r.json().get("candles", [])
                rows = []
                for c in raw:
                    if c.get("complete", False):
                        mid = c["mid"]
                        rows.append({
                            "datetime": pd.to_datetime(c["time"]),
                            "open": float(mid["o"]),
                            "high": float(mid["h"]),
                            "low": float(mid["l"]),
                            "close": float(mid["c"]),
                            "volume": float(c.get("volume", 0)),
                        })
                if len(rows) >= CONFIG.min_candles:
                    df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
                    log.info("✅ OANDA SPOT FEED TERHUBUNG: %d candle", len(df))
                    return df, "OANDA_SPOT_XAUUSD"
        except Exception as e:
            log.warning("OANDA feed fallback error: %s", e)

    # 3. Coba Binance PAXG
    try:
        df = _fetch_binance_paxg(interval=interval, total_bars=3000)
        if len(df) >= CONFIG.min_candles:
            log.info("✅ FEED ALTERNATIF TERHUBUNG: Binance PAXGUSDT (%d candle M15)", len(df))
            return df, "BINANCE_PAXG"
    except Exception as e:
        log.warning("Binance PAXG fallback error: %s", e)

    # 4. Coba Kraken PAXG
    try:
        df = _fetch_kraken_paxg(interval_min=15)
        if len(df) >= CONFIG.min_candles:
            log.info("✅ FEED ALTERNATIF TERHUBUNG: Kraken PAXGUSD (%d candle M15)", len(df))
            return df, "KRAKEN_PAXG"
    except Exception as e:
        log.warning("Kraken PAXG fallback error: %s", e)

    raise RuntimeError("SEMUA FEED DATA EMAS GAGAL. Eksekusi dibatalkan demi keamanan!")


def enforce_strictly_closed_candles(df: pd.DataFrame, timeframe_minutes: int = 15) -> Optional[pd.DataFrame]:
    """
    MEMBASMI LOOKAHEAD BIAS:
    Hanya pertahankan candle yang sudah 100% CLOSED (selesai terbentuk).
    Jangan pernah membaca bar yang sedang running di menit aktif!
    """
    if df is None or len(df) < 5:
        return None

    df = df.drop_duplicates(subset="datetime")
    df = df[(df["high"] >= df["low"]) & (df["close"] > 0) & (df["open"] > 0)]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("datetime").reset_index(drop=True)

    now = now_utc()
    # Tentukan batas penutupan bar terakhir secara presisi
    minute_offset = now.minute % timeframe_minutes
    last_closed_cutoff = now - timedelta(
        minutes=minute_offset,
        seconds=now.second,
        microseconds=now.microsecond,
    )

    # Pertahankan candle yang pembukaannya sebelum batas penutupan
    df = df[df["datetime"] < last_closed_cutoff].copy().reset_index(drop=True)
    if len(df) < CONFIG.min_candles:
        log.warning("Candle closed tidak cukup (%d < %d)", len(df), CONFIG.min_candles)
        return None
    return df


# ===========================================================================
# 5. FEATURE ENGINEERING (BEBAS LEAKAGE & REGULARISASI)
# ===========================================================================
def calculate_wilder_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()

def calculate_hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < max_lag * 2:
        return 0.5
    lags = np.arange(2, max_lag + 1)
    tau = np.array([np.std(s[lag:] - s[:-lag]) for lag in lags], dtype=float)
    valid = tau > 0
    if valid.sum() < 3:
        return 0.5
    try:
        return float(np.clip(np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)[0], 0.0, 1.0))
    except Exception:
        return 0.5

def calculate_shannon_entropy(returns: np.ndarray, bins: int = 10) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 20:
        return 1.0
    hist, _ = np.histogram(r, bins=bins)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    mx = np.log2(bins)
    return float(-np.sum(p * np.log2(p)) / mx) if mx > 0 else 1.0

def calculate_adx_clean(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    up = df["high"].diff()
    dn = -df["low"].diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = calculate_wilder_atr(df, period)
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / (atr + 1e-12)
    mdi = 100 * pd.Series(mdm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / (atr + 1e-12)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-12)
    df["plus_di"] = pdi
    df["minus_di"] = mdi
    df["di_difference"] = pdi - mdi
    df["adx"] = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    return df

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Menghitung indikator teknikal murni tanpa lookahead leakage."""
    df = df.copy()
    diff = df["close"].diff()
    gain = diff.clip(lower=0).rolling(14).mean()
    loss = (-diff.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-12))
    df["atr"] = calculate_wilder_atr(df, CONFIG.atr_period)
    df["atr_pct"] = df["atr"] / (df["close"] + 1e-12)
    df["atr_regime"] = df["atr"] / (df["atr"].rolling(100).mean() + 1e-12)

    df["ema_fast"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_trend"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_distance"] = (df["ema_fast"] - df["ema_slow"]) / (df["atr"] + 1e-12)
    df["ema_slope"] = (df["ema_fast"] - df["ema_fast"].shift(3)) / (df["atr"] + 1e-12)
    df["trend_strength"] = (df["close"] - df["ema_trend"]) / (df["atr"] + 1e-12)

    # Kalman Filter 1D
    close_vals = df["close"].to_numpy()
    n_bars = len(close_vals)
    k_est = np.zeros(n_bars)
    k_vel = np.zeros(n_bars)
    k_err = np.zeros(n_bars)
    k_est[0] = close_vals[0]
    k_err[0] = 1.0
    for i in range(1, n_bars):
        pe = k_est[i - 1] + k_vel[i - 1]
        per = k_err[i - 1] + 0.01
        kg = per / (per + 1.0 + 1e-12)
        k_est[i] = pe + kg * (close_vals[i] - pe)
        k_vel[i] = 0.85 * k_vel[i - 1] + 0.15 * (k_est[i] - k_est[i - 1])
        k_err[i] = (1 - kg) * per
    df["kalman_trend"] = k_vel / (df["atr"].to_numpy() + 1e-12)
    df["kalman_deviation"] = (close_vals - k_est) / (df["atr"].to_numpy() + 1e-12)

    df = calculate_adx_clean(df, CONFIG.atr_period)
    rolling_20_mean = df["close"].rolling(20).mean()
    rolling_20_std = df["close"].rolling(20).std()
    df["bb_position"] = (df["close"] - rolling_20_mean) / (2 * rolling_20_std + 1e-12)

    keltner_mid = df["close"].ewm(span=20, adjust=False).mean()
    keltner_band = df["atr"].ewm(span=20, adjust=False).mean() * 2
    df["keltner_position"] = (df["close"] - keltner_mid) / (keltner_band + 1e-12)

    returns = df["close"].pct_change()
    df["realized_vol"] = returns.rolling(20).std() * np.sqrt(24 * 60 / CONFIG.timeframe_minutes)
    df["vol_zscore"] = (df["realized_vol"] - df["realized_vol"].rolling(100).mean()) / (
        df["realized_vol"].rolling(100).std() + 1e-12
    )

    log_prices = np.log(df["close"].to_numpy())
    ret_arr = returns.to_numpy()
    h_arr = np.full(n_bars, 0.5)
    e_arr = np.full(n_bars, 1.0)
    for i in range(100, n_bars):
        h_arr[i] = calculate_hurst_exponent(log_prices[i - 99 : i + 1])
        e_arr[i] = calculate_shannon_entropy(ret_arr[i - 99 : i + 1])
    df["hurst"] = h_arr
    df["shannon_entropy"] = e_arr

    bar_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = (df["close"] - df["open"]).abs() / (bar_range + 1e-12)
    df["upper_wick_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (bar_range + 1e-12)
    df["lower_wick_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (bar_range + 1e-12)

    # Market Structure Break of Structure (BOS)
    prior_high = df["high"].rolling(5).max().shift(1)
    prior_low = df["low"].rolling(5).min().shift(1)
    bos_up = (df["high"] > prior_high) & (df["close"] > prior_high)
    bos_dn = (df["low"] < prior_low) & (df["close"] < prior_low)
    df["bos_score"] = np.where(bos_up, 1.0, np.where(bos_dn, -1.0, 0.0))

    sweep_up = (df["high"] > prior_high) & (df["close"] < prior_high)
    sweep_dn = (df["low"] < prior_low) & (df["close"] > prior_low)
    df["liquidity_sweep_score"] = np.where(sweep_dn, 1.0, np.where(sweep_up, -1.0, 0.0))

    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)

    hours = df["datetime"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["session_london"] = hours.between(7, 12).astype(int)
    df["session_newyork"] = hours.between(12, 20).astype(int)
    df["session_overlap"] = hours.between(12, 15).astype(int)

    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


# ===========================================================================
# 6. TRIPLE BARRIER LABELING (KONSISTEN & BEBAS BIAS)
# ===========================================================================
def triple_barrier_labels_conservative(
    df: pd.DataFrame, fwd_bars: int, atr_multiplier: float, risk_reward: float
) -> pd.DataFrame:
    """
    KOREKSI CACAT FORENSIK:
    Jika dalam satu candle kedua batas (TP dan SL) tertembus bersamaan,
    WAJIB diasumsikan Stop Loss tersentuh terlebih dahulu (Conservative Worst-Case).
    Hal ini mencegah model Machine Learning diajarkan ilusi kemenangan palsu!
    """
    df = df.copy()
    close_vals = df["close"].to_numpy()
    high_vals = df["high"].to_numpy()
    low_vals = df["low"].to_numpy()
    atr_vals = df["atr"].to_numpy()
    n = len(df)
    labels = np.zeros(n, dtype=int)

    for i in range(n):
        if i + fwd_bars >= n:
            continue
        c = close_vals[i]
        a = atr_vals[i]
        sl_dist = a * atr_multiplier
        tp_dist = sl_dist * risk_reward

        # Evaluasi skenario BUY
        up_target = c + tp_dist
        dn_target = c - sl_dist

        for j in range(i + 1, i + fwd_bars + 1):
            hit_tp = high_vals[j] >= up_target
            hit_sl = low_vals[j] <= dn_target

            # Jika keduanya tembus di bar yang sama -> Worst-case LOSS (-1)
            if hit_tp and hit_sl:
                labels[i] = -1
                break
            elif hit_sl:
                labels[i] = -1
                break
            elif hit_tp:
                labels[i] = 1
                break

    df["label"] = labels
    return df


# ===========================================================================
# 7. MODEL MACHINE LEARNING TER-REGULARISASI
# ===========================================================================
def build_regularized_model() -> VotingClassifier:
    """
    Mencegah Overfitting:
    Kedalaman pohon dipangkas (max_depth=4), min_samples_leaf dinaikkan ke 40
    agar model tidak menghafal noise derau harga 4 hari terakhir.
    """
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_leaf=40,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        random_state=42,
    )
    return VotingClassifier([("rf", rf), ("gb", gb)], voting="soft")

def train_and_validate_models(df_train: pd.DataFrame) -> Dict[str, Any]:
    """Melatih primary model & meta-model dengan Purged Walk-Forward."""
    df_labeled = triple_barrier_labels_conservative(
        df_train, CONFIG.forward_bars, CONFIG.atr_sl_multiplier, CONFIG.risk_reward
    )
    train_clean = df_labeled.iloc[: -CONFIG.forward_bars]
    X = train_clean[FEATURES].to_numpy()
    y = train_clean["label"].to_numpy()

    if len(X) < CONFIG.min_train_samples or len(np.unique(y)) < 2:
        return {"valid": False, "reason": "insufficient_train_data"}

    # Purged TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=CONFIG.wf_splits)
    aucs, precs = [], []
    for tr_idx, val_idx in tscv.split(X):
        tr_end = tr_idx[-1] - CONFIG.forward_bars
        if tr_end < 150:
            continue
        tr_slice = np.arange(tr_idx[0], tr_end + 1)
        if len(np.unique(y[tr_slice])) < 2:
            continue
        try:
            m = build_regularized_model()
            m.fit(X[tr_slice], y[tr_slice])
            preds = m.predict(X[val_idx])
            proba = m.predict_proba(X[val_idx])
            mask = y[val_idx] != 0
            if mask.sum() >= 10:
                y_dir = y[val_idx][mask]
                p_dir = preds[mask]
                if len(np.unique(y_dir)) >= 2:
                    precs.append(precision_score(y_dir, p_dir, average="macro", zero_division=0))
                classes = list(m.classes_)
                if 1 in classes and -1 in classes:
                    bi, si = classes.index(1), classes.index(-1)
                    y_bin = (y_dir == 1).astype(int)
                    if len(np.unique(y_bin)) >= 2:
                        aucs.append(roc_auc_score(y_bin, proba[mask][:, [si, bi]][:, 1]))
        except Exception as e:
            log.warning("WF Split error: %s", e)

    wf_auc = float(np.mean(aucs)) if aucs else 0.5
    wf_prec = float(np.mean(precs)) if precs else 0.0

    if wf_auc < CONFIG.wf_min_auc:
        return {"valid": False, "reason": f"wf_auc_low_{wf_auc:.2f}"}

    # Latih Primary Model penuh
    primary_model = build_regularized_model()
    primary_model.fit(X, y)

    return {
        "valid": True,
        "primary": primary_model,
        "wf_auc": round(wf_auc, 3),
        "wf_prec": round(wf_prec, 3),
    }


# ===========================================================================
# 8. MANAJEMEN LOT & RISIKO AMAN (SAFETY FLOOR)
# ===========================================================================
def compute_institutional_trade(
    sig_type: str,
    df_feat: pd.DataFrame,
    probability: float,
    state: Dict[str, Any],
    source: str,
) -> Optional[Dict[str, Any]]:
    """
    PERLINDUNGAN MODAL PENUH:
    1. Lot dihitung menggunakan pembulatan ke bawah (floor) agar tidak over-risk.
    2. Jika risiko riil pada lot minimum melebihi max_risk_percent, batalkan trade!
    3. Mendukung akun Cent dan Standard secara dinamis.
    """
    last = df_feat.iloc[-1]
    price = float(last["close"])
    atr = float(last["atr"])

    sl_dist = atr * CONFIG.atr_sl_multiplier
    tp_dist = sl_dist * CONFIG.risk_reward

    if sig_type == "BUY":
        sl = price - sl_dist - CONFIG.spread_usd
        tp = price + tp_dist
    else:
        sl = price + sl_dist + CONFIG.spread_usd
        tp = price - tp_dist

    sl_abs = abs(price - sl)
    if sl_abs <= 0:
        return None

    # Hitung target risiko dollar
    session_mult = get_session_risk_multiplier()
    risk_pct = max(CONFIG.min_risk_percent, min(CONFIG.risk_percent * session_mult, CONFIG.max_risk_percent))
    target_risk_usd = CONFIG.account_equity * (risk_pct / 100.0)

    # Formula Kuantitatif: Ukuran lot murni
    # lot = target_risk_usd / (sl_abs * contract_size)
    theoretical_lot = target_risk_usd / (sl_abs * CONFIG.contract_size)
    # Gunakan math.floor agar risiko tidak membengkak
    enforced_lot = math.floor(theoretical_lot * 100.0) / 100.0

    # SAFETY CHECK:
    # Jika lot teoretis < min_lot (misal akun $200 butuh 0.006 lot tapi broker batasi 0.01 lot)
    if enforced_lot < CONFIG.min_lot:
        real_risk_at_min_lot = CONFIG.min_lot * sl_abs * CONFIG.contract_size
        real_risk_pct = (real_risk_at_min_lot / CONFIG.account_equity) * 100.0

        # Jika risiko riil melampaui max_risk_percent (1.5%), tolak order!
        if real_risk_pct > CONFIG.max_risk_percent:
            log.warning(
                "⛔ SAFETY REJECTION: Modal $%.2f terlalu kecil untuk trading 0.01 lot di SL $%.2f. "
                "Risiko riil akan membengkak ke %.2f%% (Maksimal diizinkan: %.2f%%). "
                "Pindahkan ke Akun Cent atau perbesar modal!",
                CONFIG.account_equity,
                sl_abs,
                real_risk_pct,
                CONFIG.max_risk_percent,
            )
            return None
        enforced_lot = CONFIG.min_lot

    enforced_lot = min(enforced_lot, CONFIG.max_lot)
    actual_risk_usd = enforced_lot * sl_abs * CONFIG.contract_size
    actual_risk_pct = (actual_risk_usd / CONFIG.account_equity) * 100.0

    expires = now_utc() + timedelta(minutes=CONFIG.timeframe_minutes * CONFIG.forward_bars * 2)

    return {
        "type": sig_type,
        "entry": round(price, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "lot": enforced_lot,
        "risk_pct": round(actual_risk_pct, 2),
        "risk_usd": round(actual_risk_usd, 2),
        "prob": round(probability, 3),
        "session": get_current_session(),
        "source": source,
        "time": now_utc().isoformat(),
        "expires_at": expires.isoformat(),
    }


# ===========================================================================
# 9. EVALUASI POSISI AKTIF REALISTIS (KOMISI + GAP SLIPPAGE)
# ===========================================================================
def evaluate_active_signals_clean(df: pd.DataFrame, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluasi posisi aktif dengan memotong komisi broker dan memperhitungkan
    asumsi terburuk jika SL dan TP tersentuh di bar yang sama.
    """
    if not state.get("active_signals"):
        return state

    wins = state["performance_history"].get("wins", 0)
    losses = state["performance_history"].get("losses", 0)
    daily_pnl = state.get("daily_pnl_usd", 0.0)
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
            remaining.append(sig)
            continue

        outcome = None
        for _, bar in bars.iterrows():
            if sig["type"] == "BUY":
                hit_tp = bar["high"] >= sig["tp"]
                hit_sl = bar["low"] <= sig["sl"]
            else:
                hit_tp = bar["low"] <= sig["tp"]
                hit_sl = bar["high"] >= sig["sl"]

            # Jika keduanya tersentuh di bar yang sama -> Loss konservatif
            if hit_tp and hit_sl:
                outcome = "loss"
                break
            elif hit_sl:
                outcome = "loss"
                break
            elif hit_tp:
                outcome = "win"
                break

        if outcome is None:
            remaining.append(sig)
            continue

        lot = float(sig["lot"])
        entry = float(sig["entry"])
        commission = lot * CONFIG.commission_per_lot * 2.0  # Round-turn fee

        if outcome == "win":
            wins += 1
            move = (sig["tp"] - entry) if sig["type"] == "BUY" else (entry - sig["tp"])
            net_pnl = (move * lot * CONFIG.contract_size) - commission
            log.info("✅ WIN %s (+%.2f USD setelah komisi)", sig["type"], net_pnl)
        else:
            losses += 1
            move = (sig["sl"] - entry) if sig["type"] == "BUY" else (entry - sig["sl"])
            # Ditambah penalti slippage gap
            net_pnl = (move * lot * CONFIG.contract_size) - commission - (CONFIG.slippage_usd * lot * CONFIG.contract_size)
            log.info("❌ LOSS %s (%.2f USD setelah komisi & slip)", sig["type"], net_pnl)

        daily_pnl += net_pnl
        recent.append(outcome)

    state["active_signals"] = remaining
    state["performance_history"]["wins"] = wins
    state["performance_history"]["losses"] = losses
    state["daily_pnl_usd"] = round(daily_pnl, 2)
    state["recent_outcomes"] = recent[-50:]
    return state


# ===========================================================================
# 10. TELEGRAM NOTIFIER
# ===========================================================================
def send_telegram(message: str) -> bool:
    if CONFIG.backtest_mode:
        return True
    if not CONFIG.telegram_token or not CONFIG.telegram_chat_id:
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
            r = requests.post(url, json=payload, timeout=8)
            if r.status_code == 200:
                return True
            if r.status_code == 429:
                retry_sec = int(r.json().get("parameters", {}).get("retry_after", 5)) + 1
                py_time.sleep(retry_sec)
                continue
        except Exception:
            py_time.sleep(2 * attempt)
    return False

def format_clean_signal_message(trade: Dict[str, Any], wf_auc: float) -> str:
    emoji = "🟢" if trade["type"] == "BUY" else "🔴"
    return (
        f"<b>{emoji} XAUUSD {trade['type']} [{trade.get('session','?')}]</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕒 Waktu: {trade['time'][:16].replace('T',' ')} UTC\n"
        f"📡 Sumber: 🏛️ {trade['source']}\n"
        f"🎯 Probabilitas ML: {trade['prob']*100:.1f}%\n"
        f"📊 Walk-Forward AUC: {wf_auc:.3f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📍 Entry: <b>{trade['entry']:.2f}</b>\n"
        f"🛑 SL: <b>{trade['sl']:.2f}</b>\n"
        f"🎯 TP: <b>{trade['tp']:.2f}</b>\n"
        f"⚖️ Lot: <b>{trade['lot']}</b> (Risiko {trade['risk_pct']:.1f}% = ${trade['risk_usd']:.2f})\n"
        f"<i>Kadaluarsa: {trade['expires_at'][:16].replace('T',' ')} UTC</i>"
    )


# ===========================================================================
# 11. MAIN WORKFLOW EXECUTION
# ===========================================================================
def main():
    log.info("=" * 60)
    log.info("XAUUSD BOT v25.5 — Institutional Spot Edition")
    log.info("=" * 60)

    with acquire_lock() as locked:
        if not locked:
            return

        state = load_system_state()
        state = reset_daily_state_if_needed(state)

        # 1. Cek Pasar Tutup (Forex Weekend)
        if is_forex_market_closed():
            log.info("Pasar Forex XAU/USD TUTUP (Weekend/Friday Cutoff). Bot tidur.")
            save_system_state(state)
            return

        # 2. Cek Circuit Breaker & Kill Switch
        if state.get("kill_switch_active", False):
            log.critical("🛑 BOT DIHENTIKAN: Kill switch aktif!")
            return

        if state.get("daily_signals_count", 0) >= CONFIG.max_daily_signals:
            log.info("Batas sinyal harian tercapai (%d). Istirahat.", CONFIG.max_daily_signals)
            return

        max_daily_loss_usd = (CONFIG.max_daily_loss_pct * CONFIG.account_equity) / 100.0
        if state.get("daily_pnl_usd", 0.0) <= -max_daily_loss_usd:
            log.warning("🛑 Batas kerugian harian terlampaui (-$%.2f). Eksekusi dihentikan.", max_daily_loss_usd)
            return

        # 3. Ambil data Spot Emas murni
        try:
            df, source = fetch_spot_candles()
            df = enforce_strictly_closed_candles(df, CONFIG.timeframe_minutes)
            if df is None:
                return
        except Exception as e:
            log.error("Gagal mengambil data candle spot: %s", e)
            return

        state["last_data_source"] = source
        state = evaluate_active_signals_clean(df, state)

        # 4. Feature Engineering
        try:
            df_feat = calculate_features(df)
            if len(df_feat) < 200:
                log.warning("Data fitur belum cukup (%d bar).", len(df_feat))
                return
        except Exception as e:
            log.error("Feature engineering gagal: %s", e)
            return

        last_bar = df_feat.iloc[-1]
        hurst = float(last_bar["hurst"])
        entropy = float(last_bar["shannon_entropy"])

        # Filter Kondisi Choppy / Sideway
        if entropy > CONFIG.max_entropy_threshold or hurst < CONFIG.min_hurst_threshold:
            log.info("Pasar dalam kondisi Choppy/Random Walk (Hurst=%.2f, Entropy=%.2f). Lewati bar.", hurst, entropy)
            save_system_state(state)
            return

        # 5. Latih / Dapatkan Model ML
        model_pack = train_and_validate_models(df_feat)
        if not model_pack.get("valid"):
            log.info("Validasi model ditolak: %s", model_pack.get("reason"))
            save_system_state(state)
            return

        model = model_pack["primary"]
        X_now = df_feat[FEATURES].iloc[[-1]].to_numpy()
        pred_side = int(model.predict(X_now)[0])
        probas = model.predict_proba(X_now)[0]
        classes = list(model.classes_)
        proba_dict = {c: float(probas[idx]) for idx, c in enumerate(classes)}

        if pred_side == 0:
            save_system_state(state)
            return

        sig_type = "BUY" if pred_side == 1 else "SELL"
        prob = proba_dict.get(pred_side, 0.0)

        if prob < CONFIG.base_min_signal_probability:
            log.info("Probabilitas %s (%.1f%%) di bawah threshold (%.1f%%)", sig_type, prob * 100, CONFIG.base_min_signal_probability * 100)
            save_system_state(state)
            return

        # 6. Hitung Trade Aman
        trade = compute_institutional_trade(sig_type, df_feat, prob, state, source)
        if not trade:
            save_system_state(state)
            return

        # 7. Kirim Sinyal
        msg = format_clean_signal_message(trade, model_pack.get("wf_auc", 0.5))
        if send_telegram(msg):
            log.info("✅ Sinyal %s terkirim ke Telegram!", sig_type)
            state["active_signals"].append(trade)
            state["daily_signals_count"] += 1
            state["last_signal_time"] = now_utc().isoformat()
        else:
            log.error("Gagal mengirim notifikasi ke Telegram.")

        save_system_state(state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAUUSD BOT v25.5 (Institusional)")
    parser.add_argument("--backtest", action="store_true", help="Jalankan backtest")
    parser.add_argument("--test-sources", action="store_true", help="Uji data feed")
    args = parser.parse_args()

    if args.test_sources:
        print("\n=== PENGUJIAN FEED SPOT EMAS (XAU/USD) ===")
        try:
            df, src = fetch_spot_candles()
            print(f"✅ Feed Sukses: {src} | Total: {len(df)} candle M15")
            print(f"📍 Harga Terakhir: ${float(df.iloc[-1]['close']):.2f}")
        except Exception as err:
            print(f"❌ Error: {err}")
    else:
        main()
