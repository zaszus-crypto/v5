#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLOBAL CONFIGURATION & ENVIRONMENT VARIABLES (v10.2 ULTIMATE)
=============================================================
"""

import os

class Config:
    BOT_NAME = "XAUUSD QUANT PREDATOR ULTIMATE"
    VERSION = "10.2-PRO"
    
    # Simbol Pasar Utama & Fallback
    PRIMARY_SYMBOL = "GC=F"  # Gold Futures Yahoo Finance sebagai proksi XAUUSD
    FALLBACK_SYMBOLS = ["PAXG-USD", "DX-Y.NYB"]
    
    # Telegram Credentials
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "MASUKKAN_TOKEN_TELEGRAM_ANDA")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "MASUKKAN_CHAT_ID_ANDA")
    
    # GitHub Gist State Persistence Credentials (Anti-Spam Cloud Storage)
    GH_GIST_TOKEN = os.getenv("GH_GIST_TOKEN", "MASUKKAN_GITHUB_GIST_TOKEN_ANDA")
    GH_GIST_ID = os.getenv("GH_GIST_ID", "MASUKKAN_GIST_ID_ANDA")
    
    # Waktu Operasional & Sesi (WIB) - Sesi London & New York
    SESSION_START_HOUR = 13
    SESSION_END_HOUR = 23
    
    # Parameter Teknikal & SMC (Kausal & Anti-Repaint)
    M15_LOOKBACK_CHOCH = 4
    H4_LOOKBACK_SWING = 5
    FVG_LOOKBACK_BARS = 24
    FVG_MIN_ATR_RATIO = 0.5
    FVG_TOLERANCE = 0.0015  # 0.15% toleransi harga mendekati FVG
    
    # Machine Learning Settings (Diperkaya dengan Fitur Volume)
    ML_WIN_PROB_THRESHOLD = 0.72  # Dinaikkan untuk presisi lebih ketat
    ML_TARGET_PCT = 0.0035        
    
    # Manajemen Risiko Akun Mikro
    ACCOUNT_BALANCE = 100.0       
    RISK_PERCENT = 1.0            
    MAX_ALLOWED_RISK_PERCENT = 3.0 
    CONTRACT_SIZE = 100           
    MIN_RR_RATIO = 2.0            
    SL_BUFFER_ATR_MULT = 0.5      
    
    # Anti-Spam & State Management
    COOLDOWN_HOURS = 2.0          
    MIN_PRICE_MOVE_DOLLARS = 8.0  
    
    # File Output Sementara
    CHART_FILE = "signal_chart.png"
    DXY_SYMBOL = "DX-Y.NYB"
