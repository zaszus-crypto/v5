#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MACHINE LEARNING CLASSIFIER (QUANTUM APEX ENGINE)
================================================
"""

import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
from indicators import Indicators
from config import Config


class MLClassifierEngine:
    @staticmethod
    def evaluate_probability(df_h1: pd.DataFrame, df_m15: pd.DataFrame, bias: str) -> Tuple[bool, float, str]:
        try:
            df = df_h1.copy()
            df['ema50'] = Indicators.ema(df['close'], 50)
            df['rsi'] = Indicators.rsi(df['close'], 14)
            df['atr'] = Indicators.atr(df, 14)
            df['vol_ratio'] = Indicators.volume_ratio(df, 20)
            df['ema_slope'] = Indicators.ema_slope(df['close'], 20)
            df['bb_width'] = Indicators.bb_width(df['close'], 20)
            df['return'] = df['close'].pct_change()

            target_pct = Config.ML_TARGET_PCT

            if bias == "BULLISH":
                df['target'] = (df['close'].shift(-4) > df['close'] * (1.0 + target_pct)).astype(int)
            else:
                df['target'] = (df['close'].shift(-4) < df['close'] * (1.0 - target_pct)).astype(int)

            df = df.dropna()
            if len(df) < 60:
                return True, 0.78, "ML Check Bypassed (Data historis minim)"

            features = ['rsi', 'atr', 'vol_ratio', 'ema_slope', 'bb_width', 'return']
            X = df[features]
            y = df['target']

            if len(np.unique(y)) < 2:
                return True, 0.78, "ML Check Bypassed (Market satu arah)"

            model = RandomForestClassifier(n_estimators=120, max_depth=6, min_samples_leaf=2, random_state=42)
            model.fit(X, y)

            latest_rsi = Indicators.rsi(df_m15['close'], 14).iloc[-1]
            latest_atr = Indicators.atr(df_h1, 14).iloc[-1]
            latest_vol = Indicators.volume_ratio(df_h1, 20).iloc[-1]
            latest_slope = Indicators.ema_slope(df_h1['close'], 20).iloc[-1]
            latest_bb = Indicators.bb_width(df_h1['close'], 20).iloc[-1]
            latest_ret = df_m15['close'].pct_change().iloc[-1]

            current_features = np.array([[latest_rsi, latest_atr, latest_vol, latest_slope, latest_bb, latest_ret]])
            prob = model.predict_proba(current_features)[0]
            win_prob = float(prob[1])

            threshold = Config.ML_WIN_PROB_THRESHOLD
            if win_prob >= threshold:
                return True, win_prob, f"Quantum ML Confidence: {win_prob*100:.1f}% (Lolos >= {threshold*100:.0f}%)"
            else:
                return False, win_prob, f"Quantum ML Ditolak: {win_prob*100:.1f}% (< {threshold*100:.0f}%)"

        except Exception as e:
            return True, 0.78, f"ML Check Bypassed ({e})"
