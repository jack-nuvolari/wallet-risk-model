# Wallet Risk Model

A comprehensive system for analyzing cryptocurrency wallet risk through machine learning models and real-time data collection.

## Overview

This system provides tools for:
- **Data Collection**: Gathering real-time data from Uniswap, Aave, cryptocurrency markets, and news sources
- **Risk Modeling**: Training machine learning models to predict coin and wallet risk scores
- **Wallet Analysis**: Comprehensive risk assessment of cryptocurrency wallets
- **DeFi Position Monitoring**: Tracking and analyzing DeFi protocol positions


## Usage

### Command Line Interface

The main entry point is `test.py` which provides a comprehensive CLI:

```bash
# Update all data sources
python test.py --update-all

# Train risk models
python test.py --train-model
python test.py --train-pairwise-model

# Analyze specific wallet
python test.py --wallet 0x7a29aE65Bf25Dfb6e554BF0468a6c23ed99a8DC2

# Use mock wallet for testing
python test.py --wallet mock

# Update specific data sources
python test.py --update-uniswap --update-coins

# Enable verbose output
python test.py --update-all -v
```

### Available Commands

| Command | Description |
|---------|-------------|
| `--update-all` | Update all data sources |
| `--update-uniswap` | Update Uniswap pool data |
| `--update-aave` | Update Aave market data |
| `--update-coins` | Update cryptocurrency data |
| `--update-news` | Update news sentiment data |
| `--train-model` | Train coin risk regression model |
| `--train-pairwise-model` | Train pairwise risk model |
| `--wallet <ADDRESS>` | Analyze specific wallet address |
| `--wallet-mock` | Use mock wallet data for analysis |
| `--config <FILE>` | Use custom configuration file |
| `--output-dir <DIR>` | Set custom output directory |
| `-v, --verbose` | Enable verbose output |
| `--debug` | Enable debug mode |


## Core Functions

### 1. Data Collection Functions

#### `update_data_sources(config, args)`
Updates data sources based on command line arguments.

**Parameters:**
- `config`: Configuration dictionary loaded from YAML
- `args`: Parsed command line arguments

**Returns:**
- List of successfully updated data sources

**Features:**
- Handles Uniswap, Aave, cryptocurrency, and news data updates
- Error handling for each data source
- Progress reporting and status updates

#### `load_config(config_path)`
Loads and validates configuration from YAML file.

**Parameters:**
- `config_path`: Path to configuration file

**Returns:**
- Configuration dictionary or None if error

**Error Handling:**
- File not found errors
- YAML parsing errors

### 2. Model Training Functions

#### `train_coin_risk_model()`
Trains the coin risk regression model.

**Features:**
- Imports and trains coin risk model from `ai.coinRiskModel`
- Error handling and status reporting
- Returns success/failure status

#### `pair_wise_risk_model()`
Trains the pairwise risk model for relative risk assessment.

**Features:**
- Loads coin data from JSON
- Trains pairwise comparison model
- Handles training errors gracefully


#### `risk_weight_model()`
Trains the risk weight model using pairwise learning to optimize risk component weights.

**Features:**
- Loads mock wallet risk component data from JSON
- Creates pairwise training data with extreme true weights
- Trains logistic regression model on pairwise comparisons
- Optimizes weights for 7 risk components using L1 regularization
- Handles training errors gracefully and provides weight comparison analysis

### 3. Wallet Analysis Functions

#### `analyze_wallet_risk_safe(wallet_address, config)`
Safely analyzes wallet risk with comprehensive error handling.

**Parameters:**
- `wallet_address`: Ethereum wallet address or "mock" for testing
- `config`: Configuration dictionary

**Returns:**
- Risk analysis results or None if error

**Features:**
- Supports real wallet addresses via Etherscan API
- Mock wallet support for testing
- Returns structured risk analysis

## Data Sources

### 1. Uniswap Data (`utils/collectors/uniswap.py`)
- **Purpose**: Collects Uniswap pool data for liquidity analysis
- **Data**: Pool addresses, token pairs, liquidity, fees, volume
- **Usage**: Risk assessment of DeFi liquidity positions

### 2. Aave Data (`utils/collectors/aave.py`)
- **Purpose**: Gathers Aave lending market information
- **Data**: Market addresses, assets, APYs, risk parameters
- **Usage**: Lending protocol risk evaluation

### 3. Cryptocurrency Data (`utils/collectors/coin.py`)
- **Purpose**: Collects general cryptocurrency market data
- **Data**: Prices, market caps, volumes, price changes
- **Usage**: Market risk assessment and trend analysis

### 4. News Data (`utils/collectors/news.py`)
- **Purpose**: Gathers news sentiment data for cryptocurrencies
- **Data**: News articles, sentiment scores, keywords
- **Usage**: Sentiment-based risk analysis

### 5. Etherscan Data (`utils/collectors/etherscan.py`)
- **Purpose**: Retrieves blockchain transaction data
- **Data**: Transaction history, token transfers, contract interactions
- **Usage**: Wallet behavior analysis and risk scoring

## Risk AI Models

### 1. Coin Risk Model (`ai/coin_risk_model.py`)
- **Type**: Regression model for absolute risk scoring
- **Features**: Market cap, supply, price volatility, volume analysis
- **Output**: Numerical risk score (0-100)

### 2. Pairwise Risk Model (`ai/pairwise_model.py`)
- **Type**: Learning-to-rank model for relative risk assessment
- **Features**: Pairwise comparisons between coins
- **Output**: Relative risk rankings and scores

### 3. Risk Weight Model (`ai/riskWeightModel.py`)
- **Type**: Pairwise learning-to-rank for deriving component weights
- **Goal**: Learn optimal weights for wallet risk components using only pairwise preferences
- **Data**: Mock pairwise comparisons produced with an "extreme" ground-truth weight vector to create strong training signal
- **Methods**:
  - Standard Logistic Regression (`method='logreg'`)
  - L1-regularized Logistic Regression (`method='logreg_l1'`, solver=`saga`) to encourage larger weight shifts
- **Outputs**:
  - `trained_risk_weights.json` with normalized weights (sum to 1)
  - `wallet_pairwise_training_meta.json` including `true_weights` used to generate labels


## Wallet Analysis

### Risk Components

The wallet risk calculator evaluates multiple risk factors:

1. **Frequency Risk (15%)**
   - Daily transaction patterns
   - Transaction frequency distribution
   - Peak activity periods

2. **Amount Risk (15%)**
   - Transaction volume analysis
   - Balance-to-volume ratios
   - Large transaction patterns

3. **Interaction Risk (15%)**
   - Address diversity
   - Interaction patterns
   - New address ratios

4. **Recent Activity Risk (10%)**
   - Recent transaction analysis
   - Activity trends
   - Temporal patterns

5. **Contract Risk (15%)**
   - Smart contract interactions
   - Suspicious contract detection
   - Zero-value transaction analysis

6. **Balance Risk (20%)**
   - Portfolio composition
   - Token diversification
   - Value concentration

7. **DeFi Position Risk (10%)**
   - Protocol exposure
   - Position values
   - Risk level assessment
