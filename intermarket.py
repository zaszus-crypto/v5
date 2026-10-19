#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INTERMARKET CORRELATION ENGINE (DXY FILTER)
===========================================
"""

import yfinance as yf
from typing import Tuple
from config import Config
from indicators import Indicators


class IntermarketFilter:
    @staticmethod
    def check_dxy_alignment(bias: str) -> Tuple[bool, str]:
        try:
            ticker = yf.Ticker(Config.DXY_SYMBOL)
            df = ticker.history(period="10d", interval="1h", timeout=8)
            if df.empty or len(df) < 30:
                return True, "Intermarket Check Bypassed (Data DXY minim)"

            df.columns = [c.lower() for c in df.columns]
            df['ema50'] = Indicators.ema(df['close'], 50)
            last = df.iloc[-1]

            dxy_bullish = last['close'] > last['ema50']
            dxy_bearish = last['close'] < last['ema50']

            if bias == "BULLISH" and dxy_bullish:
                return False, f"DXY Bullish > EMA50 ({last['close']:.2f}) - Menahan BUY Gold"
            elif bias == "BEARISH" and dxy_bearish:
                return False, f"DXY Bearish < EMA50 ({last['close']:.2f}) - Menahan SELL Gold"

            return True, f"DXY Aligned ({last['close']:.2f})"
        except Exception as e:
            return True, f"Intermarket Check Bypassed ({e})"
