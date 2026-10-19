#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RISK MANAGEMENT & LOT SIZING ENGINE
===================================
"""

import logging
from typing import Optional, Dict, Any
from config import Config

log = logging.getLogger("RISK_MANAGER")


class RiskManager:
    @staticmethod
    def calculate_setup(
        bias: str,
        live_price: float,
        sweep_level: float,
        atr_h1: float,
        df_h4: Any,
        sh_h4: Any,
        sl_h4: Any,
        win_prob: float
    ) -> Optional[Dict[str, Any]]:
        buffer = atr_h1 * Config.SL_BUFFER_ATR_MULT

        if bias == "BULLISH":
            sl = sweep_level - buffer
            sl_dist = live_price - sl
            h4_highs = df_h4[sh_h4]['high'].dropna()
            min_tp = live_price + (sl_dist * Config.MIN_RR_RATIO)
            tp = h4_highs.iloc[-1] if not h4_highs.empty and h4_highs.iloc[-1] > min_tp else min_tp
            tp_dist = tp - live_price
            signal_type = "BUY"
        else:
            sl = sweep_level + buffer
            sl_dist = sl - live_price
            h4_lows = df_h4[sl_h4]['low'].dropna()
            min_tp = live_price - (sl_dist * Config.MIN_RR_RATIO)
            tp = h4_lows.iloc[-1] if not h4_lows.empty and h4_lows.iloc[-1] < min_tp else min_tp
            tp_dist = live_price - tp
            signal_type = "SELL"

        if sl_dist <= 0:
            return None

        rr = tp_dist / sl_dist
        if rr < Config.MIN_RR_RATIO:
            return None

        risk_amount = Config.ACCOUNT_BALANCE * (Config.RISK_PERCENT / 100.0)
        risk_per_lot = sl_dist * Config.CONTRACT_SIZE
        calculated_lot = risk_amount / risk_per_lot if risk_per_lot > 0 else 0.01

        if calculated_lot < 0.01:
            actual_min_risk = 0.01 * risk_per_lot
            actual_risk_pct = (actual_min_risk / Config.ACCOUNT_BALANCE) * 100.0
            if actual_risk_pct > Config.MAX_ALLOWED_RISK_PERCENT:
                return None
            lot_size = 0.01
        else:
            lot_size = round(min(calculated_lot, 10.0), 2)

        return {
            "signal": signal_type,
            "price": live_price,
            "sl": sl,
            "tp": tp,
            "sl_dist": sl_dist,
            "tp_dist": tp_dist,
            "rr": rr,
            "lot": lot_size,
            "win_prob": win_prob
        }
