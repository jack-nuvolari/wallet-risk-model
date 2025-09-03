# AI Models Documentation

This document provides detailed explanations of the three AI models used in the wallet risk assessment system.

---

## Coin Risk Model

**File:** `ai/coinRiskModel.py`

### Overview
The Coin Risk Model is a traditional machine learning approach that uses Random Forest Regression to predict cryptocurrency risk scores based on market and financial features.

### Model Type
- **Algorithm:** Random Forest Regressor
- **Supervision:** Supervised Learning (Regression)
- **Output:** Continuous risk score (0-1 range)

### Model Structure
```python
RandomForestRegressor(
    n_estimators=100,    # Number of decision trees
    max_depth=10,         # Maximum depth of each tree
    random_state=42,      # For reproducibility
    n_jobs=-1            # Use all CPU cores
)
```

### Input Data Structure
The model expects coin data with the following features:

```json
{
    "name": "Bitcoin",
    "symbol": "BTC",
    "id": "bitcoin",
    "current_price": 45000.0,
    "market_cap": 850000000000,
    "total_volume": 25000000000,
    "circulating_supply": 19000000,
    "total_supply": 21000000,
    "price_change_24h": 2.5,
    "risk": 0.75
}
```

### Feature Engineering
**Primary Features:**
- `current_price`: Current market price in USD
- `market_cap`: Total market capitalization
- `total_volume`: 24-hour trading volume
- `circulating_supply`: Currently circulating coins
- `total_supply`: Maximum total supply
- `price_change_24h`: 24-hour price change percentage

**Target Variable:**
- `risk`: Risk score (0-1, where 1 is highest risk)

### Training Process
1. **Data Loading:** Reads from `data/coin_data.json`
2. **Feature Preparation:** Extracts numerical features and normalizes them
3. **Data Splitting:** 80% training, 20% testing
4. **Feature Scaling:** StandardScaler for normalization
5. **Model Training:** Random Forest on scaled features
6. **Evaluation:** MSE, RMSE, MAE, R² metrics
7. **Model Persistence:** Saves trained model and scaler

### Training Data Example
```python
# Sample training data
features = [
    [45000.0, 850000000000, 25000000000, 19000000, 21000000, 2.5],
    [3200.0, 380000000000, 15000000000, 120000000, 120000000, -1.2],
    [0.50, 50000000, 2000000, 100000000, 100000000, 5.8]
]

risk_scores = [0.75, 0.45, 0.90]
```

### Model Performance Metrics
- **Mean Squared Error (MSE):** Average squared prediction error
- **Root Mean Squared Error (RMSE):** Square root of MSE
- **Mean Absolute Error (MAE):** Average absolute prediction error
- **R² Score:** Coefficient of determination (0-1, higher is better)

### Usage Example
```python
# Load trained model
model = joblib.load('ai/models/coin_risk_model.pkl')
scaler = joblib.load('ai/models/coin_risk_scaler.pkl')

# Predict risk for new coin
new_coin_features = [100.0, 1000000000, 50000000, 100000000, 100000000, 2.5]
predicted_risk = predict_risk_for_new_coin(model, scaler, new_coin_features)
```

---

## Pairwise Coin Risk Model

**File:** `ai/pairWiseModel.py`

### Overview
The Pairwise Coin Risk Model uses a novel approach that learns from relative comparisons between cryptocurrencies rather than absolute risk scores. This approach is particularly effective when absolute risk scores are noisy or subjective.

### Model Type
- **Algorithm:** Random Forest or Gradient Boosting Regressor
- **Supervision:** Supervised Learning (Binary Classification on pairs)
- **Output:** Risk score derived from pairwise rankings

### Model Structure
```python
# Random Forest option
RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Gradient Boosting option
GradientBoostingRegressor(
    n_estimators=100,
    random_state=42
)
```

### Input Data Structure
The model expects coin data with the same structure as the Coin Risk Model, but processes it differently:

```json
{
    "name": "Ethereum",
    "symbol": "ETH",
    "market_cap": 450000000000,
    "total_supply": 120000000,
    "circulating_supply": 120000000,
    "current_price": 3800.0,
    "price_change_24h": -2.1,
    "total_volume": 18000000000,
}
```

### Feature Engineering
**Primary Features:**
- `market_cap`: Market capitalization
- `total_supply`: Maximum supply
- `circulating_supply`: Current circulation
- `current_price`: Current market price
- `price_change_24h`: 24-hour price change
- `total_volume`: Trading volume

**Derived Features:**
- `supply_utilization`: circulating_supply / total_supply
- `price_volatility`: |price_change_24h| / current_price
- `cap_volume_ratio`: market_cap / total_volume

### Training Process
1. **Feature Extraction:** Creates 9-dimensional feature vectors
2. **Pairwise Data Creation:** Generates all possible coin pairs
3. **Label Generation:** Binary labels based on risk comparison
4. **Feature Differences:** Computes feature differences between pairs
5. **Model Training:** Trains on pairwise feature differences
6. **Ranking Score Prediction:** Converts pairwise predictions to risk scores

### Pairwise Data Structure
```python
# For each coin pair (A vs B)
pairwise_features = [
    market_cap_diff,      # B_market_cap - A_market_cap
    total_supply_diff,    # B_total_supply - A_total_supply
    circulating_supply_diff,
    current_price_diff,
    price_change_diff,
    total_volume_diff,
    supply_utilization_diff,
    price_volatility_diff,
    cap_volume_ratio_diff
]

# Label: 1 if B has higher risk than A, 0 otherwise
label = 1 if risk_B > risk_A else 0
```

### Training Data Example
```python
# Sample pairwise training data
pairwise_features = [
    [1000000000, 50000000, 20000000, 100.0, 2.5, 10000000, 0.8, 0.025, 100.0],
    [-500000000, -25000000, -10000000, -50.0, -1.2, -5000000, -0.1, -0.012, -50.0]
]

pairwise_labels = [1, 0]  # B is riskier than A, A is riskier than B
```

### Prediction Process
1. **Feature Extraction:** Extract features for new coin
2. **Pairwise Comparisons:** Compare against all training samples
3. **Probability Calculation:** Predict probability of higher risk
4. **Ranking Score:** Convert to percentile-based risk score
5. **Risk Mapping:** Map to actual risk score range

### Usage Example
```python
# Initialize and train model
model = PairwiseCoinRiskModel(model_type='random_forest')
model.train(coin_data)

# Predict risk for new coin
new_coin = {
    'market_cap': 1000000000,
    'total_supply': 100000000,
    'circulating_supply': 80000000,
    'current_price': 100.0,
    'price_change_24h': 2.5,
    'total_volume': 10000000
}

predicted_risk = model.predict_risk(new_coin)
```

---

## Risk Weight Model

**File:** `ai/riskWeightModel.py`

### Overview
The Risk Weight Model learns optimal weights for different risk components in wallet risk assessment. It uses pairwise comparisons between wallets to determine which risk factors are most important for distinguishing between high and low-risk wallets.

### Model Type
- **Algorithm:** Logistic Regression (with L1 regularization option)
- **Supervision:** Supervised Learning (Binary Classification)
- **Output:** Optimal weights for risk components

### Model Structure
```python
# Standard Logistic Regression
LogisticRegression(
    random_state=42,
    max_iter=2000
)

# L1-regularized Logistic Regression
LogisticRegression(
    random_state=42,
    max_iter=4000,
    penalty='l1',
    solver='saga',
    C=0.5  # Regularization strength
)
```

### Input Data Structure
The model works with wallet risk component data:

```json
{
    "address": "0x1234...",
    "transaction_frequency_risk": 75.5,
    "transaction_amounts_risk": 60.2,
    "interaction_patterns_risk": 45.8,
    "recent_activity_risk": 30.1,
    "contract_interactions_risk": 80.3,
    "balance_risk": 55.7,
    "defi_positions_risk": 70.4
}
```

### Risk Components
**Primary Risk Factors:**
- `transaction_frequency_risk`: Risk based on transaction frequency
- `transaction_amounts_risk`: Risk based on transaction amounts
- `interaction_patterns_risk`: Risk based on interaction patterns
- `recent_activity_risk`: Risk based on recent activity
- `contract_interactions_risk`: Risk based on smart contract interactions
- `balance_risk`: Risk based on wallet balance
- `defi_positions_risk`: Risk based on DeFi positions

### Training Process
1. **Mock Data Generation:** Creates synthetic wallet pairs
2. **True Weight Sampling:** Generates ground-truth weights using Dirichlet distribution
3. **Pairwise Labeling:** Labels pairs based on true weights
4. **Feature Engineering:** Creates feature differences between wallet pairs
5. **Model Training:** Trains logistic regression on pairwise data
6. **Weight Extraction:** Converts coefficients to normalized weights

### Training Data Structure
```python
# For each wallet pair (A vs B)
pairwise_features = [
    transaction_frequency_diff,    # B_freq - A_freq
    transaction_amounts_diff,      # B_amounts - A_amounts
    interaction_patterns_diff,     # B_patterns - A_patterns
    recent_activity_diff,          # B_activity - A_activity
    contract_interactions_diff,    # B_contracts - A_contracts
    balance_risk_diff,             # B_balance - A_balance
    defi_positions_diff            # B_defi - A_defi
]

# Label: 1 if B has higher risk than A, 0 otherwise
label = 1 if risk_B > risk_A else 0
```

### Mock Data Generation
```python
# Sample mock wallet
mock_wallet = {
    'address': 'Wallet_A_001',
    'transaction_frequency_risk': 75.5,
    'transaction_amounts_risk': 60.2,
    'interaction_patterns_risk': 45.8,
    'recent_activity_risk': 30.1,
    'contract_interactions_risk': 80.3,
    'balance_risk': 55.7,
    'defi_positions_risk': 70.4
}

# True weights (ground truth)
true_weights = {
    'transaction_frequency': 0.25,
    'transaction_amounts': 0.15,
    'interaction_patterns': 0.10,
    'recent_activity': 0.05,
    'contract_interactions': 0.20,
    'balance_risk': 0.15,
    'defi_positions': 0.10
}
```

### Training Data Example
```python
# Sample pairwise training data
pairwise_features = [
    [25.5, -15.2, 8.7, -5.1, 12.3, -8.9, 4.6],
    [-12.3, 8.9, -4.6, 15.2, -8.7, 5.1, -12.3]
]

pairwise_labels = [1, 0]  # B riskier than A, A riskier than B
```

### Weight Calculation Process
1. **Coefficient Extraction:** Get logistic regression coefficients
2. **Absolute Values:** Convert to positive weights
3. **Normalization:** Ensure weights sum to 1.0
4. **Component Mapping:** Map to risk component names

### Current vs Trained Weights
```python
# Initial weights
current_weights = {
    'transaction_frequency': 0.15,
    'transaction_amounts': 0.15,
    'interaction_patterns': 0.15,
    'recent_activity': 0.10,
    'contract_interactions': 0.15,
    'balance_risk': 0.20,
    'defi_positions': 0.10
}

# Trained weights (example)
trained_weights = {
    'transaction_frequency': 0.25,
    'transaction_amounts': 0.12,
    'interaction_patterns': 0.08,
    'recent_activity': 0.05,
    'contract_interactions': 0.22,
    'balance_risk': 0.18,
    'defi_positions': 0.10
}
```


---

## Model Comparison Summary

| Aspect | Coin Risk Model | Pairwise Model | Risk Weight Model |
|--------|----------------|----------------|-------------------|
| **Purpose** | Predict absolute coin risk | Predict relative coin risk | Learn optimal risk weights |
| **Input** | Market/financial features | Market/financial features | Wallet risk components |
| **Output** | Continuous risk score | Ranking-based risk score | Component weights |
| **Training** | Direct regression | Pairwise classification | Pairwise classification |
| **Use Case** | Individual coin assessment | Comparative coin analysis | Wallet risk weighting |
| **Advantage** | Simple, interpretable | Robust to noise | Learns from comparisons |

