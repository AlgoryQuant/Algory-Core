import json
import os
import sys
import time
from datetime import datetime, timedelta
import warnings
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from colorama import init, Fore, Style

warnings.filterwarnings("ignore")
init()

# --- ⚙️ NASTAVENÍ SYMBOLŮ (Rozděleno do kategorií) ---
MAJORS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCAD=X", "USDCHF=X", "AUDUSD=X", "NZDUSD=X"]
MINORS = [
    "GBPJPY=X", "EURJPY=X", "EURGBP=X", "AUDJPY=X", "CADJPY=X", 
    "EURAUD=X", "GBPAUD=X", "EURCAD=X", "GBPCAD=X", "AUDCAD=X"
]
METALS = ["GC=F", "SI=F"]

TIMEFRAME = "15m"
PERIOD = "60d"

FEATURES = ['RSI', 'DistMA', 'ATR', 'Hour', 'RSI_Lag1', 'RSI_Lag2', 'RSI_Lag3', 'Dist_Lag1', 'Dist_Lag2', 'Dist_Lag3']
WEB_PATH = r"C:\Algory\algory-web\public\results.json"

# 🛠️ PARAMETRY OBCHODU PRO VŠECHNY PÁRY
# [SL, TP, Partial_Dist, BreakEven, Max_Spread]
PARAMS = {
    "GOLD": [800, 9999, 1000, 100, 160.0],
    "SILVER": [400, 9999, 500, 50, 80.0],
    "EURUSD": [150, 9999, 200, 50, 20.0],
    "GBPUSD": [200, 9999, 250, 50, 25.0],
    "USDJPY": [200, 9999, 250, 50, 25.0],
    "USDCAD": [200, 9999, 250, 50, 25.0],
    "USDCHF": [200, 9999, 250, 50, 25.0],
    "AUDUSD": [150, 9999, 200, 50, 20.0],
    "NZDUSD": [150, 9999, 200, 50, 20.0],
    "GBPJPY": [250, 9999, 300, 100, 35.0],
    "EURJPY": [200, 9999, 250, 50, 30.0],
    "EURGBP": [150, 9999, 200, 50, 20.0],
    "AUDJPY": [200, 9999, 250, 50, 30.0],
    "CADJPY": [200, 9999, 250, 50, 30.0],
    "EURAUD": [250, 9999, 300, 50, 35.0],
    "GBPAUD": [300, 9999, 350, 100, 40.0],
    "EURCAD": [250, 9999, 300, 50, 35.0],
    "GBPCAD": [300, 9999, 350, 100, 40.0],
    "AUDCAD": [200, 9999, 250, 50, 30.0]
}

def get_live_spread(ticker):
    try:
        info = yf.Ticker(ticker).info
        ask = info.get('ask')
        bid = info.get('bid')
        
        if ask and bid and ask > bid > 0:
            multiplier = 1000 if "JPY" in ticker or "GC=F" in ticker or "SI=F" in ticker else 100000
            spread_points = round((ask - bid) * multiplier, 1)
            return spread_points
    except Exception:
        pass
    return "N/A"

def calculate_indicators(df):
    try:
        df = df.copy()
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
        df['close'] = df['Close'].astype(float)
        df['high'] = df['High'].astype(float)
        df['low'] = df['Low'].astype(float)
        df['open'] = df['Open'].astype(float)

        delta = df['close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / ema_down
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)

        df['MA'] = df['close'].rolling(50).mean()
        df['DistMA'] = df['close'] - df['MA']
        df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
        df['Hour'] = df.index.hour / 24.0

        for i in range(1, 4):
            df[f'RSI_Lag{i}'] = df['RSI'].shift(i)
            df[f'Dist_Lag{i}'] = df['DistMA'].shift(i)

        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"{Fore.RED}Chyba při výpočtu indikátorů: {e}{Style.RESET_ALL}")
        return None

def evaluate_symbol(ticker):
    df = yf.download(ticker, period=PERIOD, interval=TIMEFRAME, progress=False)
    
    if df.empty:
        return 0.0

    df = calculate_indicators(df)
    if df is None or len(df) < 500:
        return 0.0

    atr_current = df['ATR'].iloc[-1]
    target_dist = atr_current * 0.8
    min_target = 2.5 if "GC=F" in ticker else 0.0010
    if target_dist < min_target: target_dist = min_target

    df['Future'] = df['close'].shift(-8)
    df['Target'] = 0
    df.loc[df['Future'] > df['close'] + target_dist, 'Target'] = 1
    df.loc[df['Future'] < df['close'] - target_dist, 'Target'] = 2
    df.dropna(inplace=True)

    X = df[FEATURES]
    y = df['Target']
    split = int(len(df) * 0.8)
    
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if len(X_test) == 0 or len(np.unique(y_train)) < 2:
        return 0.0

    model = XGBClassifier(
        n_estimators=150, 
        learning_rate=0.05, 
        max_depth=5, 
        objective='multisoftprob', 
        num_class=3, 
        eval_metric='mlogloss', 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    return round(float(accuracy), 3)

def process_category(tickers, category_dict, parameters_dict, category_name):
    print(f"\n{Fore.CYAN}--- Zpracovávám sekci: {category_name} ---{Style.RESET_ALL}")
    for ticker in tickers:
        clean_name = ticker.replace("=X", "").replace("GC=F", "GOLD").replace("SI=F", "SILVER")
        print(f"Stahuji a trénuji: {clean_name}...", end=" ")
        
        win_rate = evaluate_symbol(ticker)
        live_spread = get_live_spread(ticker)
        
        category_dict[clean_name] = win_rate
        
        vals = PARAMS.get(clean_name, [0,0,0,0,0])
        parameters_dict[clean_name] = {
            "SL": vals[0], "TP": vals[1], "Partial": vals[2], "BE": vals[3], 
            "MaxSpread": vals[4], "LiveSpread": live_spread
        }
        
        col = Fore.GREEN if win_rate > 0.52 else Fore.WHITE
        print(f"{col}{win_rate * 100:.1f}%{Style.RESET_ALL} (Spread: {live_spread})")

def run_analytics():
    now_time = datetime.now().strftime('%H:%M:%S')
    print(f"\n{Fore.YELLOW}=== SPUŠTĚNÍ AI ANALÝZY (Zavřená svíčka - {now_time}) ==={Style.RESET_ALL}")
    
    web_data = {"majors": {}, "minors": {}, "metals": {}, "parameters": {}}

    process_category(MAJORS, web_data["majors"], web_data["parameters"], "MAJORS")
    process_category(MINORS, web_data["minors"], web_data["parameters"], "MINORS (CROSSES)")
    process_category(METALS, web_data["metals"], web_data["parameters"], "METALS")

    try:
        os.makedirs(os.path.dirname(WEB_PATH), exist_ok=True)
        with open(WEB_PATH, "w") as f:
            json.dump(web_data, f)
        print(f"\n{Fore.BLUE}--- Výsledky úspěšně exportovány pro Web ---{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Chyba při zápisu JSON: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    run_analytics()
    
    while True:
        now = datetime.now()
        minutes_to_add = 15 - (now.minute % 15)
        next_run = (now + timedelta(minutes=minutes_to_add)).replace(second=10, microsecond=0)
        sleep_seconds = (next_run - datetime.now()).total_seconds()
        
        print(f"\n{Fore.YELLOW}Další aktualizace přesně v: {next_run.strftime('%H:%M:%S')} (za {sleep_seconds:.0f} sekund){Style.RESET_ALL}")
        time.sleep(sleep_seconds)
        run_analytics()
