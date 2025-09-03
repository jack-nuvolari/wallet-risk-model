# Wallet Risk Assessment Documentation

This document provides detailed explanations of all risk calculations performed by the `WalletRiskCalculator` class in `utils/walletRiskCalculator.py`.


## Overview

The `WalletRiskCalculator` analyzes cryptocurrency wallets by evaluating 7 distinct risk components and combining them into a comprehensive risk score. Each component is weighted and contributes to the final risk assessment.

**Final Risk Score Formula:**
```
Total Risk Score = Σ(Component Score × Component Weight)
Final Score = max(1, min(100, round(Total Risk Score)))
```

---

## Risk Components & Weights

| Component | Weight | Description |
|-----------|--------|-------------|
| Transaction Frequency | 15% | Daily transaction patterns and frequency analysis |
| Transaction Amounts | 15% | Volume analysis and amount distribution |
| Interaction Patterns | 15% | Address diversity and interaction behavior |
| Recent Activity | 10% | Recent transaction patterns and trends |
| Contract Interactions | 15% | Smart contract interaction analysis |
| Balance Risk | 20% | Portfolio composition and token risk |
| DeFi Positions | 10% | DeFi protocol exposure and position risk |

**Total Weight: 100%**

---

## Risk Calculation Methods

### 1. Transaction Frequency Risk (15%)

**Purpose:** Analyzes how frequently a wallet performs transactions and identifies unusual patterns.

**Risk Factors:**
- **High Frequency (>1.67 tx/day):** +40 points
- **Medium Frequency (>0.67 tx/day):** +20 points
- **Peak Daily Activity (>10 tx):** +30 points
- **Moderate Daily Activity (>5 tx):** +15 points
- **Very Low Activity (<0.1 tx/day):** +10 points

**Thresholds:**
- `high_frequency`: 50 transactions per month
- `medium_frequency`: 20 transactions per month

**Output Details:**
- Total transactions count
- Total days of activity
- Average daily transactions
- Maximum daily transactions
- Frequency pattern classification

---

### 2. Transaction Amounts Risk (15%)

**Purpose:** Evaluates transaction volume patterns, amount distribution, and volume-to-balance ratios.


**Risk Factors:**
- **Large Transactions (>10 ETH):** +30 points
- **Medium Transactions (>1 ETH):** +15 points
- **High Turnover (>100x balance):** +25 points
- **Medium Turnover (>50x balance):** +15 points
- **Low Turnover (>10x balance):** +5 points
- **High Amount Variance (>100):** +20 points
- **High Zero-Value Ratio (>30%):** +10 points

**Thresholds:**
- `high_amount`: 10.0 ETH
- `medium_amount`: 1.0 ETH

**Output Details:**
- Total volume in ETH
- Average transaction amount
- Maximum transaction amount
- Current balance
- Volume-to-balance ratio
- Zero-amount transaction count

---

### 3. Interaction Patterns Risk (15%)

**Purpose:** Assesses the diversity of addresses a wallet interacts with and identifies suspicious patterns.

**Risk Factors:**
- **Very High Diversity (>50 addresses):** +25 points
- **High Diversity (>20 addresses):** +15 points
- **Medium Diversity (>10 addresses):** +5 points
- **High Concentration (>50% with one address):** +20 points
- **Medium Concentration (>30% with one address):** +10 points
- **High New Address Ratio (>2.0):** +15 points

**Output Details:**
- Unique addresses count
- Total interactions
- Interaction diversity ratio
- Most interactive address

---

### 4. Recent Activity Risk (10%)

**Purpose:** Analyzes recent transaction patterns and temporal behavior trends.


**Risk Factors:**
- **Very High Recent Activity (>80%):** +25 points
- **High Recent Activity (>50%):** +15 points
- **Very Low Recent Activity (<10%):** +10 points
- **High Recent Daily Activity (>5 tx/day):** +20 points

**Thresholds:**
- `recent_days`: 30 days

**Output Details:**
- Recent transaction count
- Recent token transfer count
- Total recent activity
- Recent activity ratio
- Maximum recent daily activity

---

### 5. Contract Interaction Risk (15%)

**Purpose:** Evaluates smart contract interactions and identifies suspicious contract behavior.


**Risk Factors:**
- **Very High Contract Ratio (>70%):** +25 points
- **High Contract Ratio (>50%):** +15 points
- **Medium Contract Ratio (>30%):** +5 points
- **Suspicious Contract Interaction:** +30 points
- **High Zero-Value Ratio (>50%):** +20 points

**Thresholds:**
- `suspicious_contracts`: List of known suspicious contract addresses

**Output Details:**
- Contract interaction count
- Contract interaction ratio
- Suspicious contracts list
- Zero-value transaction count
- Zero-value transaction ratio

---

### 6. Balance Risk (20%)

**Purpose:** Assesses portfolio composition and individual token risk scores.

**Risk Factors:**
- **Individual Token Risk Scores:** Based on coin_data.json
- **Unknown Token Penalty:** Default 0.8 risk score
- **Portfolio Weighting:** Value-weighted risk calculation

**Output Details:**
- Portfolio risk score
- Token breakdown with individual risks
- Total portfolio value
- Individual token values and risk scores

---

### 7. DeFi Position Risk (10%)

**Purpose:** Evaluates DeFi protocol exposure and position-specific risks.

**Risk Factors:**
- **Protocol-Specific Risk:** Aave and Uniswap risk scores
- **Position Value Weighting:** Value-weighted risk calculation
- **Default Risk Levels:** Very low (0.2) to Very high (1.0)

**Risk Level Mapping:**
- `very_low`: 0.2
- `low`: 0.4
- `medium`: 0.6
- `high`: 0.8
- `very_high`: 1.0

**Output Details:**
- Overall position risk score
- Total position value
- Position count
- Protocol-specific risk breakdown

---

## Risk Scoring System

### Final Risk Score Calculation

```python
# Component risk scores (0-100)
risk_components = {
    'frequency_risk': frequency_score,
    'amount_risk': amount_score,
    'interaction_risk': interaction_score,
    'recent_activity_risk': recent_score,
    'contract_risk': contract_score,
    'balance_risk': balance_score,
    'defi_position_risk': position_score
}

# Weighted calculation
total_risk_score = sum(
    component['score'] * self.risk_weights[weight_key]
    for weight_key, component in zip(self.risk_weights.keys(), risk_components.values())
)

# Final score normalization (1-100)
final_risk_score = max(1, min(100, round(total_risk_score)))
```
