#!/usr/init/env python3
# -*- coding: utf-8 -*-
"""
XAU/USD PREDATOR v10.2 - ULTIMATE QUANT EDITION
================================================
"""

import time
import logging
from datetime import datetime, timezone, timedelta
import pandas as pd
import yfinance as yf

from config import Config
from state_manager import StateManager
from news_filter import NewsFilter
from intermarket import IntermarketFilter
from smc_engine import SMCQuant
from ml_engine import MLClassifierEngine
from risk_manager import RiskManager
from telegram_notifier import TelegramNotifier
from indicators import Indicators

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("PREDATOR_MAIN")


class MarketDataEngine:
    @staticmethod
    def fetch_ohlcv(symbol: str, interval: str, period: str, retries: int = 3) -> pd.DataFrame:
        symbols_to_try = [symbol] + [s for s in Config.FALLBACK_SYMBOLS if s != symbol]
        for sym in symbols_to_try:
            for attempt in range(retries):
                try:
                    ticker = yf.Ticker(sym)
                    df = ticker.history(period=period, interval=interval, timeout=10)
                    if df.empty:
                        df = yf.download(tickers=sym, period=period, interval=interval, progress=False)

                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                    if not df.empty and len(df) >= 10:
                        df.columns = [str(c).lower() for c in df.columns]
                        required = ['open', 'high', 'low', 'close', 'volume']
                        if all(col in df.columns for col in required):
                            df = df[required].dropna()
                            df.index = pd.to_datetime(df.index, utc=True)
                            return df
                except Exception as e:
                    log.warning(f"Percobaan {attempt+1} fetch {sym} [{interval}] gagal: {e}")
                    time.sleep(1.5)
        return pd.DataFrame()


def run_pipeline() -> None:
    log.info(f"==> Menjalankan Scan Pasar Ultimate: {Config.PRIMARY_SYMBOL}...")
    state_mgr = StateManager()

    now_wib = datetime.now(timezone.utc) + timedelta(hours=7)
    if not (Config.SESSION_START_HOUR <= now_wib.hour <= Config.SESSION_END_HOUR):
        log.info(f"Di luar sesi likuiditas optimal ({now_wib.hour}:00 WIB). Scan dilewati.")
        return

    news_ok, news_msg = NewsFilter.check_high_impact_news()
    log.info(f"[LAYER 1 - News] {news_msg}")
    if not news_ok:
        return

    df_d1 = MarketDataEngine.fetch_ohlcv(Config.PRIMARY_SYMBOL, "1d", "1y")
    df_h1 = MarketDataEngine.fetch_ohlcv(Config.PRIMARY_SYMBOL, "1h", "90d")
    df_m15 = MarketDataEngine.fetch_ohlcv(Config.PRIMARY_SYMBOL, "15m", "5d")

    if df_d1.empty or df_h1.empty or df_m15.empty:
        log.error("Datafeed tidak mencukupi.")
        return

    # KUNCI UTAMA ANTI-REPAINT: Buang bar terakhir yang belum close
    closed_m15 = df_m15.iloc[:-1].copy()
    live_price = float(df_m15['close'].iloc[-1])

    df_h4 = df_h1.resample("4h", offset="0h").agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()

    if len(df_h4) < 220:
        return

    df_d1['ema50'] = Indicators.ema(df_d1['close'], 50)
    df_d1['ema200'] = Indicators.ema(df_d1['close'], 200)
    last_d1 = df_d1.iloc[-2]

    if last_d1['close'] > last_d1['ema50'] > last_d1['ema200']:
        bias = "BULLISH"
    elif last_d1['close'] < last_d1['ema50'] < last_d1['ema200']:
        bias = "BEARISH"
    else:
        return

    dxy_ok, dxy_msg = IntermarketFilter.check_dxy_alignment(bias)
    if not dxy_ok:
        return

    sh_h4, sl_h4 = SMCQuant.detect_confirmed_swings(df_h4, Config.H4_LOOKBACK_SWING)
    recent_h4 = df_h4.tail(6)
    sweep_valid, sweep_level = False, 0.0

    if bias == "BULLISH":
        valid_lows = df_h4[sl_h4]['low'].dropna()
        if valid_lows.empty: return
        key_low = valid_lows.iloc[-1]
        for _, row in recent_h4.iloc[::-1].iterrows():
            if row['low'] < key_low and row['close'] > key_low:
                sweep_valid, sweep_level = True, row['low']
                break
    else:
        valid_highs = df_h4[sh_h4]['high'].dropna()
        if valid_highs.empty: return
        key_high = valid_highs.iloc[-1]
        for _, row in recent_h4.iloc[::-1].iterrows():
            if row['high'] > key_high and row['close'] < key_high:
                sweep_valid, sweep_level = True, row['high']
                break

    if not sweep_valid:
        return

    if not SMCQuant.detect_choch(closed_m15, bias):
        return

    df_h1['atr'] = Indicators.atr(df_h1, 14)
    atr_h1 = float(df_h1['atr'].iloc[-2])
    fvg_found, fvg_top, fvg_bottom = SMCQuant.find_fresh_fvg(df_h1, bias, sweep_level, atr_h1)

    if not fvg_found:
        return

    tol = Config.FVG_TOLERANCE
    if not (fvg_bottom * (1 - tol) <= live_price <= fvg_top * (1 + tol)):
        return

    ml_ok, win_prob, ml_msg = MLClassifierEngine.evaluate_probability(df_h1, closed_m15, bias)
    if not ml_ok:
        return

    setup = RiskManager.calculate_setup(
        bias=bias, live_price=live_price, sweep_level=sweep_level,
        atr_h1=atr_h1, df_h4=df_h4, sh_h4=sh_h4, sl_h4=sl_h4, win_prob=win_prob
    )

    if not setup:
        return

    # Validasi Anti-Spam via GitHub Gist Cloud
    if state_mgr.is_spam(setup['signal'], setup['price']):
        log.info("Sinyal diblokir oleh sistem Anti-Spam Cloud (Cooldown aktif).")
        return

    chart_path = TelegramNotifier.generate_candlestick_chart(closed_m15, setup)
    filters_passed = [
        news_msg, f"D1 Bias: {bias}", dxy_msg,
        f"H4 Sweep @ ${sweep_level:.2f}", "M15 ChoCh Konfirmasi",
        f"Fresh FVG [${fvg_bottom:.2f} - ${fvg_top:.2f}]", ml_msg,
        f"Risk:Reward 1:{setup['rr']:.2f}"
    ]

    TelegramNotifier.send_signal(setup, filters_passed, chart_path)
    state_mgr.save_to_gist(setup['signal'], setup['price'])


def main():
    log.info(f"🚀 Memulai {Config.BOT_NAME} v{Config.VERSION}")
    run_pipeline()


if __name__ == "__main__":
    main()
