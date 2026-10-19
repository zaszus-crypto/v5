#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMC QUANT ENGINE (CAUSAL & UNBIASED)
====================================
"""

import pandas as pd
import numpy as np
from typing import Tuple
from config import Config


class SMCQuant:
    @staticmethod
    def detect_confirmed_swings(df: pd.DataFrame, lookback: int = 4) -> Tuple[pd.Series, pd.Series]:
        n = len(df)
        sh = pd.Series(False, index=df.index)
        sl = pd.Series(False, index=df.index)

        highs = df['high'].values
        lows = df['low'].values

        for i in range(lookback, n - lookback):
            current_h = highs[i]
            current_l = lows[i]

            is_sh = True
            for k in range(1, lookback + 1):
                if highs[i - k] >= current_h or highs[i + k] > current_h:
                    is_sh = False
                    break
            if is_sh:
                sh.iloc[i] = True

            is_sl = True
            for k in range(1, lookback + 1):
                if lows[i - k] <= current_l or lows[i + k] < current_l:
                    is_sl = False
                    break
            if is_sl:
                sl.iloc[i] = True

        return sh, sl

    @staticmethod
    def detect_choch(df_m15: pd.DataFrame, direction: str) -> bool:
        sh, sl = SMCQuant.detect_confirmed_swings(df_m15, Config.M15_LOOKBACK_CHOCH)
        recent = df_m15.tail(8)

        if direction == "BULLISH":
            highs = df_m15[sh]['high'].dropna()
            if highs.empty:
                return False
            last_swing_high = highs.iloc[-1]
            return bool((recent['close'] > last_swing_high).any())
        else:
            lows = df_m15[sl]['low'].dropna()
            if lows.empty:
                return False
            last_swing_low = lows.iloc[-1]
            return bool((recent['close'] < last_swing_low).any())

    @staticmethod
    def find_fresh_fvg(df_h1: pd.DataFrame, bias: str, sweep_level: float, atr_h1: float) -> Tuple[bool, float, float]:
        n = len(df_h1)
        max_lookback = min(Config.FVG_LOOKBACK_BARS, n - 3)

        for i in range(n - 2, n - max_lookback - 1, -1):
            c1, c2, c3 = df_h1.iloc[i-2], df_h1.iloc[i-1], df_h1.iloc[i]

            if bias == "BULLISH" and c3['low'] > c1['high']:
                gap_size = c3['low'] - c1['high']
                if gap_size >= (atr_h1 * Config.FVG_MIN_ATR_RATIO) and c1['high'] >= sweep_level:
                    fvg_top = float(c3['low'])
                    fvg_bottom = float(c1['high'])
                    subsequent = df_h1.iloc[i+1:]
                    mitigated = (subsequent['low'] <= fvg_bottom).any()
                    if not mitigated:
                        return True, fvg_top, fvg_bottom

            elif bias == "BEARISH" and c3['high'] < c1['low']:
                gap_size = c1['low'] - c3['high']
                if gap_size >= (atr_h1 * Config.FVG_MIN_ATR_RATIO) and c1['low'] <= sweep_level:
                    fvg_top = float(c1['low'])
                    fvg_bottom = float(c3['high'])
                    subsequent = df_h1.iloc[i+1:]
                    mitigated = (subsequent['high'] >= fvg_top).any()
                    if not mitigated:
                        return True, fvg_top, fvg_bottom

        return False, 0.0, 0.0
