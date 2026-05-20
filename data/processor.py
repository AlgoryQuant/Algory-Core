import pandas as pd

def add_technical_indicators(df):
    """
    Přidá technické indikátory pro trénování modelu.
    """
    if df is None or df.empty:
        return None
    
    # Vytvoření kopie, abychom neupravovali původní data
    data = df.copy()
    
    # 1. Procentuální změna (denní návratnost) - klíčová vlastnost pro model
    data['returns'] = data['Close'].pct_change()
    
    # 2. Jednoduchý klouzavý průměr (SMA) za 20 dní
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    
    # 3. Volatilita (standardní odchylka za 20 dní)
    data['Volatility_20'] = data['Close'].rolling(window=20).std()
    
    # Odstranění řádků s prázdnými hodnotami (vznikají výpočtem SMA)
    data.dropna(inplace=True)
    
    return data

if __name__ == "__main__":
    # Testovací funkce, pokud bys to chtěl spustit
    print("Processor je připraven.")
    
