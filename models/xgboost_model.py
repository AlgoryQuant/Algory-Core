import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(df):
    """
    Natrénuje XGBoost model na základě připravených dat.
    """
    # Definujeme, co chceme předpovídat (zda zítřejší cena bude vyšší než dnešní)
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    
    # Výběr příznaků (features), které model používá
    features = ['returns', 'SMA_20', 'Volatility_20']
    X = df[features]
    y = df['target']
    
    # Rozdělení dat na trénovací a testovací (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Inicializace a trénování modelu
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5)
    model.fit(X_train, y_train)
    
    # Evaluace
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model accuracy: {acc:.2f}")
    
    return model

if __name__ == "__main__":
    print("XGBoost module ready for training.")
