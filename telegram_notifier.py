#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TELEGRAM NOTIFICATION & CHART RENDERING ENGINE
==============================================
"""

import os
import time
import logging
import requests
import pandas as pd
import mplfinance as mpf
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
from config import Config

log = logging.getLogger("TELEGRAM_NOTIFIER")


class TelegramNotifier:
    @staticmethod
    def generate_candlestick_chart(df_m15: pd.DataFrame, setup: Dict[str, Any]) -> str:
        try:
            df_plot = df_m15.tail(45).copy()
            df_plot.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
            }, inplace=True)

            mc = mpf.make_marketcolors(
                up='#00B57C', down='#FF3B30', edge='inherit', wick='inherit', volume='in'
            )
            s = mpf.make_mpf_style(
                marketcolors=mc, gridstyle=':', y_on_right=True, rc={'font.size': 9}
            )
            hlines = dict(
                hlines=[setup['price'], setup['tp'], setup['sl']],
                colors=['#0088CC', '#00B57C', '#FF3B30'],
                linestyle='--', linewidths=1.5
            )
            chart_path = Config.CHART_FILE
            mpf.plot(
                df_plot, type='candle', style=s, hlines=hlines,
                title=f"\n{Config.PRIMARY_SYMBOL} ({setup['signal']}) - M15",
                savefig=dict(fname=chart_path, dpi=140, bbox_inches='tight'), figscale=1.1
            )
            return chart_path
        except Exception as e:
            log.error(f"Gagal generate chart: {e}")
            return ""

    @staticmethod
    def send_signal(setup: Dict[str, Any], filters_passed: List[str], chart_path: str = "") -> bool:
        token, chat_id = Config.TELEGRAM_TOKEN, Config.TELEGRAM_CHAT_ID
        if not token or not chat_id:
            return False

        now_wib = (datetime.now(timezone.utc) + timedelta(hours=7)).strftime("%d %b %Y, %H:%M WIB")
        emoji = "🟢" if setup['signal'] == "BUY" else "🔴"
        checklist_text = "\n".join([f"  ✅ {f}" for f in filters_passed])

        caption = (
            f"{emoji} <b>{Config.BOT_NAME} v{Config.VERSION}</b>\n"
            f"⚡ <b>INSTITUTIONAL QUANT SIGNAL</b>\n"
            f"📅 Waktu: {now_wib}\n"
            f"🎯 <b>ML Win Probability: {setup['win_prob']*100:.1f}%</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>AKSI: {setup['signal']} XAUUSD</b>\n"
            f"🔵 <b>Entry:</b> ${setup['price']:.2f}\n"
            f"🟢 <b>TP:</b> ${setup['tp']:.2f} (+${setup['tp_dist']:.2f})\n"
            f"🔴 <b>SL:</b> ${setup['sl']:.2f} (-${setup['sl_dist']:.2f})\n"
            f"⚖️ <b>Risk:Reward:</b> 1:{setup['rr']:.2f}\n"
            f"📏 <b>Lot Size:</b> {setup['lot']} (Risk {Config.RISK_PERCENT}%)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>🧠 CONFLUENCE CHECKLIST:</b>\n"
            f"{checklist_text}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<i>🔵 Entry | 🟢 TP | 🔴 SL</i>"
        )

        url_photo = f"https://api.telegram.org/bot{token}/sendPhoto"
        url_text = f"https://api.telegram.org/bot{token}/sendMessage"

        for attempt in range(3):
            try:
                if chart_path and os.path.exists(chart_path):
                    with open(chart_path, 'rb') as photo:
                        r = requests.post(
                            url_photo,
                            data={"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"},
                            files={'photo': photo},
                            timeout=20
                        )
                else:
                    r = requests.post(
                        url_text,
                        json={"chat_id": chat_id, "text": caption, "parse_mode": "HTML"},
                        timeout=15
                    )

                if r.status_code == 200:
                    return True
                time.sleep(2)
            except Exception as e:
                log.error(f"Error Telegram attempt {attempt+1}: {e}")
                time.sleep(2)
        return False
