#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLOUD STATE & ANTI-SPAM MANAGER (GITHUB GIST INTEGRATION)
=========================================================
Menyelesaikan kelemahan state persistence di lingkungan serverless/CI-CD.
"""

import json
import logging
import requests
from datetime import datetime, timezone
from typing import Dict, Any
from config import Config

log = logging.getLogger("STATE_MGR")


class StateManager:
    def __init__(self):
        self.token = Config.GH_GIST_TOKEN
        self.gist_id = Config.GH_GIST_ID
        self.filename = "bot_state_xauusd.json"
        self.data = self._load_from_gist()

    def _load_from_gist(self) -> Dict[str, Any]:
        if not self.token or not self.gist_id or "MASUKKAN" in self.token:
            log.warning("Gist Token/ID belum diset. Fallback ke state kosong.")
            return {}
        
        url = f"https://api.github.com/gists/{self.gist_id}"
        headers = {"Authorization": f"token {self.token}", "Accept": "application/vnd.github.v3+json"}
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                files = res.json().get("files", {})
                if self.filename in files:
                    content = files[self.filename].get("content", "{}")
                    return json.loads(content)
        except Exception as e:
            log.error(f"Gagal memuat state dari Gist: {e}")
        return {}

    def save_to_gist(self, signal_type: str, price: float):
        self.data = {
            "last_signal": signal_type,
            "last_price": float(price),
            "last_time": datetime.now(timezone.utc).isoformat()
        }
        
        if not self.token or not self.gist_id or "MASUKKAN" in self.token:
            return

        url = f"https://api.github.com/gists/{self.gist_id}"
        headers = {"Authorization": f"token {self.token}", "Accept": "application/vnd.github.v3+json"}
        payload = {
            "files": {
                self.filename: {
                    "content": json.dumps(self.data, indent=2)
                }
            }
        }
        try:
            requests.patch(url, headers=headers, json=payload, timeout=10)
            log.info("☁️ State anti-spam berhasil disinkronkan ke GitHub Gist.")
        except Exception as e:
            log.error(f"Gagal menyimpan state ke Gist: {e}")

    def is_spam(self, signal_type: str, price: float) -> bool:
        if "last_time" not in self.data or price <= 0:
            return False

        last_time = datetime.fromisoformat(self.data["last_time"])
        now = datetime.now(timezone.utc)
        hours_passed = (now - last_time).total_seconds() / 3600.0

        if hours_passed < Config.COOLDOWN_HOURS:
            if self.data.get("last_signal") == signal_type:
                dollar_diff = abs(price - self.data.get("last_price", 0.0))
                if dollar_diff < Config.MIN_PRICE_MOVE_DOLLARS:
                    return True
        return False
