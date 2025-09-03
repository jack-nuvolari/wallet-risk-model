#!/usr/bin/env python3
"""
Script to create mock training dataset for pairwise risk model.
Generates synthetic coin data with realistic features and risk scores.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def generate_mock_coin_data(num_coins=100):
    """
    Generate mock coin data with realistic features and risk scores.
    
    Args:
        num_coins: Number of coins to generate
        
    Returns:
        List of coin dictionaries
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Define coin categories with different risk profiles
    coin_categories = {
        'stablecoin': {'risk_range': (0.05, 0.15), 'price_volatility': (0.001, 0.01)},
        'major_crypto': {'risk_range': (0.15, 0.35), 'price_volatility': (0.02, 0.08)},
        'defi_token': {'risk_range': (0.25, 0.45), 'price_volatility': (0.05, 0.15)},
        'meme_coin': {'risk_range': (0.40, 0.70), 'price_volatility': (0.10, 0.30)},
        'new_token': {'risk_range': (0.50, 0.80), 'price_volatility': (0.15, 0.40)}
    }
    
    # Popular coin names and symbols for realism
    coin_names = [
        "Bitcoin", "Ethereum", "Cardano", "Solana", "Polkadot", "Chainlink", "Uniswap",
        "Aave", "Compound", "SushiSwap", "Yearn Finance", "Curve", "Balancer", "Synthetix",
        "Maker", "Ren", "Kyber", "Bancor", "1inch", "Synthetix", "Loopring", "ZRX",
        "BAT", "ZEC", "DASH", "LTC", "BCH", "ETC", "XMR", "XRP", "ADA", "DOT", "LINK",
        "UNI", "AAVE", "COMP", "SUSHI", "YFI", "CRV", "BAL", "SNX", "MKR", "REN", "KNC"
    ]
    
    # Generate coins
    coins = []
    
    for i in range(num_coins):
        # Select random category
        category = random.choice(list(coin_categories.keys()))
        category_config = coin_categories[category]
        
        # Generate realistic market cap (log-normal distribution)
        market_cap = np.random.lognormal(20, 1.5)  # Most coins between 1M and 100B
        
        # Generate supply based on market cap and price
        current_price = np.random.uniform(0.001, 50000)
        total_supply = market_cap / current_price
        
        # Circulating supply (usually 70-100% of total)
        circulating_ratio = np.random.uniform(0.7, 1.0)
        circulating_supply = total_supply * circulating_ratio
        
        # Generate 24h price change based on category volatility
        volatility_range = category_config['price_volatility']
        price_change_24h = np.random.uniform(-volatility_range[1], volatility_range[1]) * current_price
        
        # Generate trading volume (usually 1-20% of market cap daily)
        volume_ratio = np.random.uniform(0.01, 0.20)
        total_volume = market_cap * volume_ratio
        
        # Generate risk score based on category and features
        base_risk = np.random.uniform(*category_config['risk_range'])
        
        # Adjust risk based on features
        risk_adjustments = []
        
        # Higher volatility = higher risk
        if abs(price_change_24h) / current_price > 0.1:
            risk_adjustments.append(0.1)
        
        # Lower market cap = higher risk
        if market_cap < 1000000000:  # Less than 1B
            risk_adjustments.append(0.15)
        
        # Lower volume relative to market cap = higher risk
        if volume_ratio < 0.05:
            risk_adjustments.append(0.1)
        
        # Supply concentration (if circulating is much less than total)
        if circulating_ratio < 0.8:
            risk_adjustments.append(0.05)
        
        # Calculate final risk score
        final_risk = min(0.95, base_risk + sum(risk_adjustments))
        
        # Create coin data
        coin = {
            "id": f"mock-coin-{i+1}",
            "name": coin_names[i % len(coin_names)] + f" Mock {i+1}",
            "symbol": f"MC{i+1:02d}",
            "market_cap": round(market_cap, 2),
            "total_supply": round(total_supply, 2),
            "circulating_supply": round(circulating_supply, 2),
            "current_price": round(current_price, 6),
            "price_change_24h": round(price_change_24h, 6),
            "total_volume": round(total_volume, 2),
            "risk": round(final_risk, 3),
            "category": category,
            "created_date": (datetime.now() - timedelta(days=random.randint(1, 365))).strftime("%Y-%m-%d")
        }
        
        coins.append(coin)
    
    return coins

def create_pairwise_training_data(coins):
    """
    Create pairwise training data from coin list.
    
    Args:
        coins: List of coin dictionaries
        
    Returns:
        List of pairwise comparison examples
    """
    pairwise_data = []
    
    for i in range(len(coins)):
        for j in range(i + 1, len(coins)):
            coin_a = coins[i]
            coin_b = coins[j]
            
            # Determine which coin has higher risk
            if coin_a['risk'] > coin_b['risk']:
                higher_risk_coin = coin_a
                lower_risk_coin = coin_b
                label = 1  # Coin A has higher risk
            else:
                higher_risk_coin = coin_b
                lower_risk_coin = coin_a
                label = 0  # Coin B has higher risk
            
            # Create feature difference
            features_a = [
                coin_a['market_cap'], coin_a['total_supply'], coin_a['circulating_supply'],
                coin_a['current_price'], coin_a['price_change_24h'], coin_a['total_volume']
            ]
            
            features_b = [
                coin_b['market_cap'], coin_b['total_supply'], coin_b['circulating_supply'],
                coin_b['current_price'], coin_b['price_change_24h'], coin_b['total_volume']
            ]
            
            # Add derived features
            if coin_a['total_supply'] > 0:
                features_a.append(coin_a['circulating_supply'] / coin_a['total_supply'])
            else:
                features_a.append(0)
                
            if coin_b['total_supply'] > 0:
                features_b.append(coin_b['circulating_supply'] / coin_b['total_supply'])
            else:
                features_b.append(0)
            
            if coin_a['current_price'] > 0:
                features_a.append(abs(coin_a['price_change_24h']) / coin_a['current_price'])
            else:
                features_a.append(0)
                
            if coin_b['current_price'] > 0:
                features_b.append(abs(coin_b['price_change_24h']) / coin_b['current_price'])
            else:
                features_b.append(0)
            
            if coin_a['total_volume'] > 0:
                features_a.append(coin_a['market_cap'] / coin_a['total_volume'])
            else:
                features_a.append(0)
                
            if coin_b['total_volume'] > 0:
                features_b.append(coin_b['market_cap'] / coin_b['total_volume'])
            else:
                features_b.append(0)
            
            # Feature difference
            feature_diff = [a - b for a, b in zip(features_a, features_b)]
            
            pairwise_example = {
                "coin_a": {
                    "name": coin_a['name'],
                    "symbol": coin_a['symbol'],
                    "risk": coin_a['risk']
                },
                "coin_b": {
                    "name": coin_b['name'],
                    "symbol": coin_b['symbol'],
                    "risk": coin_b['risk']
                },
                "features": feature_diff,
                "label": label,
                "risk_difference": abs(coin_a['risk'] - coin_b['risk'])
            }
            
            pairwise_data.append(pairwise_example)
    
    return pairwise_data

def analyze_dataset(coins):
    """Analyze the generated dataset."""
    print("=== Dataset Analysis ===")
    print(f"Total coins: {len(coins)}")
    
    # Risk distribution
    risks = [coin['risk'] for coin in coins]
    print(f"\nRisk Score Distribution:")
    print(f"  Min: {min(risks):.3f}")
    print(f"  Max: {max(risks):.3f}")
    print(f"  Mean: {np.mean(risks):.3f}")
    print(f"  Median: {np.median(risks):.3f}")
    print(f"  Std: {np.std(risks):.3f}")
    
    # Category distribution
    categories = {}
    for coin in coins:
        cat = coin['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\nCategory Distribution:")
    for cat, count in categories.items():
        print(f"  {cat}: {count} coins")
    
    # Market cap distribution
    market_caps = [coin['market_cap'] for coin in coins]
    print(f"\nMarket Cap Distribution:")
    print(f"  Min: ${min(market_caps):,.0f}")
    print(f"  Max: ${max(market_caps):,.0f}")
    print(f"  Mean: ${np.mean(market_caps):,.0f}")
    
    # Price volatility
    volatilities = [abs(coin['price_change_24h']) / coin['current_price'] for coin in coins if coin['current_price'] > 0]
    print(f"\nPrice Volatility (24h):")
    print(f"  Mean: {np.mean(volatilities):.3f}")
    print(f"  Max: {max(volatilities):.3f}")

def main():
    """Main function to generate and save mock dataset."""
    
    print("=== Generating Mock Coin Dataset for Pairwise Risk Model ===\n")
    
    # Generate mock data
    num_coins = 150  # Generate 150 coins for good training data
    coins = generate_mock_coin_data(num_coins)
    
    # Create pairwise training data
    pairwise_data = create_pairwise_training_data(coins)
    
    # Analyze dataset
    analyze_dataset(coins)
    
    print(f"\nGenerated {len(pairwise_data)} pairwise training examples")
    
    # Save datasets
    output_dir = "data/mock/"
    
    # Save full coin data
    with open("../data/mock_coins.json", "w") as f:
        json.dump(coins, f, indent=2)
    
    # Save pairwise training data
    with open("../data/mock_pairwise_training.json", "w") as f:
        json.dump(pairwise_data, f, indent=2)
    
    # Save a smaller version for quick testing
    test_coins = coins[:50]  # First 50 coins
    test_pairwise = create_pairwise_training_data(test_coins)
    
    with open("../data/mock_coins_test.json", "w") as f:
        json.dump(test_coins, f, indent=2)
    
    with open("../data/mock_pairwise_test.json", "w") as f:
        json.dump(test_pairwise, f, indent=2)
    
    print(f"\nDatasets saved:")
    print(f"  - Full dataset: {len(coins)} coins, {len(pairwise_data)} pairs")
    print(f"  - Test dataset: {len(test_coins)} coins, {len(test_pairwise)} pairs")
    
    # Show some examples
    print(f"\n=== Sample Pairwise Examples ===")
    for i, example in enumerate(pairwise_data[:5]):
        print(f"\nExample {i+1}:")
        print(f"  {example['coin_a']['name']} (Risk: {example['coin_a']['risk']:.3f}) vs")
        print(f"  {example['coin_b']['name']} (Risk: {example['coin_b']['risk']:.3f})")
        print(f"  Label: {example['label']} (1 if A > B, 0 if B > A)")
        print(f"  Risk Difference: {example['risk_difference']:.3f}")
    
    print(f"\n=== Dataset Generation Complete ===")

if __name__ == "__main__":
    main()
