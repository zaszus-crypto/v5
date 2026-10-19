#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAIL-SAFE ECONOMIC CALENDAR NEWS FILTER
=======================================
"""

import requests
from datetime import datetime, timezone, timedelta
from typing import Tuple


class NewsFilter:
    @staticmethod
    def check_high_impact_news() -> Tuple[bool, str]:
        now_utc = datetime.now(timezone.utc)
        now_wib = now_utc + timedelta(hours=7)

        try:
            url = "https://netaction.vercel.app/api/news"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                events = response.json()
                for event in events:
                    if event.get('currency') == 'USD' and event.get('impact') == 'High':
                        event_time = datetime.fromisoformat(event['date'].replace('Z', '+00:00'))
                        diff_minutes = abs((event_time - now_utc).total_seconds()) / 60.0
                        if diff_minutes <= 45:
                            return False, f"🚨 Berita High Impact ({event.get('title')}) dalam {int(diff_minutes)} menit"
                return True, "Aman dari Berita High-Impact USD"
        except Exception as e:
            if 19 <= now_wib.hour <= 21:
                return False, f"⚠️ Proteksi Fundamental: News API offline saat jam rilis US ({now_wib.hour}:{now_wib.minute:02d} WIB)"
            return True, f"News Filter Bypassed di luar jam US ({e})"

        return True, "Aman dari Berita High-Impact USD"
