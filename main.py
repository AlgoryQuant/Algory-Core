from data.data_fetcher import fetch_market_data
from data.processor import add_technical_indicators
from models.xgboost_model import train_model

def run_algory_engine(ticker):
    print(f"--- Spouštím Algory Engine pro {ticker} ---")
    
    # 1. Stažení dat
    raw_data = fetch_market_data(ticker)
    
    # 2. Zpracování dat (přidání indikátorů)
    processed_data = add_technical_indicators(raw_data)
    
    # 3. Trénování modelu
    if processed_data is not None:
        model = train_model(processed_data)
        print("--- Engine úspěšně dokončil proces ---")
    else:
        print("Chyba: Data nebyla zpracována.")

if __name__ == "__main__":
    # Zde si můžeš změnit ticker na cokoliv, co tě zajímá
    run_algory_engine("AAPL")
