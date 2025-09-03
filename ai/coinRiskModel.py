import json
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler


def prepare_features(coin_data):
    """
    Prepare features for machine learning model
    """
    features = []
    risk_scores = []
    
    for coin in coin_data:
        feature_vector = [
            float(coin['current_price']),
            float(coin['market_cap']),
            float(coin['total_volume']),
            float(coin['circulating_supply']),
            float(coin['total_supply']),
            float(coin['price_change_24h']) if coin['price_change_24h'] is not None else 0.0
        ]
        
        features.append(feature_vector)
        risk_scores.append(float(coin['risk']))
    
    return np.array(features), np.array(risk_scores)

def train_random_forest_regressor(X, y):
    """
    Train a Random Forest regressor.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train_scaled, y_train)
    
    y_pred = rf_model.predict(X_test_scaled)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return rf_model, scaler, X_train_scaled, X_test_scaled, y_train, y_test, y_pred, mse, rmse, mae, r2

def analyze_feature_importance(model, feature_names):
    """
    Analyze and visualize feature importance.
    """
    feature_importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    return feature_importance_df

def predict_risk_for_new_coin(model, scaler, coin_features):
    """
    Predict risk score for a new coin.
    """
    scaled_features = scaler.transform([coin_features])
    
    predicted_risk = model.predict(scaled_features)[0]
    
    return predicted_risk

def train():
    with open('data/coin_data.json', 'r', encoding='utf-8') as f:
        coin_data = json.load(f)
    
    print(f"Loaded {len(coin_data)} coins")
    
    risk_scores = [float(coin['risk']) for coin in coin_data]
    print(f"Risk Score Statistics:")
    print(f"Min: {min(risk_scores):.3f}")
    print(f"Max: {max(risk_scores):.3f}")
    print(f"Mean: {np.mean(risk_scores):.3f}")
    print(f"Std: {np.std(risk_scores):.3f}")
    
    X, y = prepare_features(coin_data)
    feature_names = [
        'current_price', 'market_cap', 'total_volume', 
        'circulating_supply', 'total_supply', 'price_change_24h'
    ]
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    print("\nTraining Random Forest Regressor...")
    model, scaler, X_train, X_test, y_train, y_test, y_pred, mse, rmse, mae, r2 = train_random_forest_regressor(X, y)
    
    print(f"\nModel Performance Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"R² Score: {r2:.4f}")
    
    feature_importance_df = analyze_feature_importance(model, feature_names)
    print("\nFeature Importance:")
    print(feature_importance_df)
    
    joblib.dump(model, 'ai/models/coin_risk_model.pkl')
    joblib.dump(scaler, 'ai/models/coin_risk_scaler.pkl')
    
    print("Model and scaler saved to 'models/'")
    
    training_data = []
    for coin in coin_data:
        training_row = {
            'name': coin['name'],
            'symbol': coin['symbol'],
            'id': coin['id'],
            'current_price': coin['current_price'],
            'market_cap': coin['market_cap'],
            'total_volume': coin['total_volume'],
            'circulating_supply': coin['circulating_supply'],
            'total_supply': coin['total_supply'],
            'price_change_24h': coin['price_change_24h'],
            'actual_risk_score': float(coin['risk'])
        }
        training_data.append(training_row)
    
    training_df = pd.DataFrame(training_data)
    training_df.to_csv('data/coin_risk_regression_data.csv', index=False, encoding='utf-8')
    
    print("Enhanced training dataset saved to 'data/coin_risk_regression_data.csv'")
    
    print("Sample predictions from test set:")
    for i, coin in enumerate(coin_data[:5]):
        if i < len(y_pred):
            actual_risk = float(coin['risk'])
            predicted_risk = y_pred[i]
            error = predicted_risk - actual_risk
            print(f"{coin['name']} ({coin['symbol']}): Actual: {actual_risk:.3f}, Predicted: {predicted_risk:.3f}, Error: {error:+.3f}")
    
    print("Testing prediction on a new coin...")
    sample_features = [100.0, 1000000000, 50000000, 100000000, 100000000, 2.5]  # Example features
    predicted_risk = predict_risk_for_new_coin(model, scaler, sample_features)
    print(f"Sample coin features: {sample_features}")
    print(f"Predicted risk score: {predicted_risk:.3f}")
