import math
import json
import os
import requests
from pathlib import Path
current_folder = Path(__file__).parent

class Coin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @property
    def coin_data_local_path(self):
        return os.path.join(
            current_folder, "..", "..", "data", f"coin_data.json"
        )
    
    def clamp(self, value, min_value=0, max_value=100):
        return f'{max(min(value, max_value), min_value):.2f}'

    def calculate_price_volatility(self, price_change_percentage_24h):
        if price_change_percentage_24h is None:
            return 50
        if float(price_change_percentage_24h) < 0:
            risk = 20 + 60 / (1 + math.exp(-abs(price_change_percentage_24h) / 5))
        else:
            risk = 20 * math.exp(-price_change_percentage_24h / 15)
        
        return min(100, max(0, risk)) 

    def calculate_liquidity_risk(self, total_volume, market_cap):
        liquidity_ratio = total_volume / market_cap  
        
        if liquidity_ratio > 0:
            normalized_ratio = min(liquidity_ratio * 100, 10) 
            risk = 80 / (1 + math.exp(normalized_ratio - 2))  
        else:
            risk = 80  
        
        return min(100, max(0, risk))

    def calculate_concentration_risk(self, circulating_supply, max_supply):
        if max_supply is not None and max_supply > 0:
            supply_ratio = circulating_supply / max_supply
        else:
            supply_ratio = 1  
        
        risk = 70 * (1 - supply_ratio) ** 2
        
        return min(100, max(0, risk))

    def calculate_market_cap_risk(self, market_cap):    
        if market_cap > 0:
            normalized_cap = min(market_cap / 1e9, 100) 
            risk = 60 / (1 + math.exp(normalized_cap - 5))  
        else:
            risk = 60  
        
        return min(100, max(0, risk))

    def calculate_short_term_volatility_risk(self, price_change_1h, price_change_7d):
        
        short_term_vol = abs(price_change_1h)
        long_term_vol = abs(price_change_7d) / 7  # Normalize to daily
        
        if long_term_vol > 0:
            volatility_ratio = short_term_vol / long_term_vol
        else:
            volatility_ratio = short_term_vol * 5  # Reduced penalty for zero long-term volatility
        
        # Apply sigmoid scaling for better distribution
        normalized_ratio = min(volatility_ratio, 10)  # Cap at 10x
        risk = 50 / (1 + math.exp(-normalized_ratio + 2))  # Sigmoid with center at 2x
        
        return min(100, max(0, risk))

    def calculate_momentum_risk(self, price_change_1h, price_change_24h, price_change_7d):
        """Calculate momentum risk based on price change consistency."""
        # Continuous formula: inconsistent momentum patterns = higher risk
        
        if price_change_1h is None or price_change_24h is None or price_change_7d is None:
            return 50
        
        # Calculate momentum consistency
        momentum_1h = price_change_1h
        momentum_24h = price_change_24h
        momentum_7d = price_change_7d
        
        # Check for momentum consistency (all positive or all negative)
        signs = [math.copysign(1, momentum_1h), math.copysign(1, momentum_24h), math.copysign(1, momentum_7d)]
        sign_consistency = len(set(signs))  # 1 = consistent, 2-3 = inconsistent
        
        # Calculate average momentum strength
        avg_momentum = (abs(momentum_1h) + abs(momentum_24h) + abs(momentum_7d)) / 3
        
        # Risk formula: sigmoid scaling for better distribution
        consistency_penalty = (sign_consistency - 1) * 20  # 0-40 penalty (reduced)
        momentum_risk = min(avg_momentum * 1.5, 40)  # 0-40 based on momentum strength (reduced)
        
        risk = consistency_penalty + momentum_risk
        
        return min(100, max(0, risk))

    # Risk Calculation function based on Exponential Finance framework
    def calculate_asset_risk(self, data):
        """Calculate comprehensive asset risk following Exponential Finance framework."""
        # Extract data from the provided dictionary
        price_change_24h = data.get('price_change_percentage_24h', 0)
        total_volume = data.get('total_volume', 0)
        market_cap = data.get('market_cap', 0)
        circulating_supply = data.get('circulating_supply', 0)
        max_supply = data.get('max_supply', None)
        price_change_1h = data.get('price_change_percentage_1h', 0)
        price_change_7d = data.get('price_change_percentage_7d', 0)
        
        # Calculate individual risk components using continuous formulas
        volatility_risk = self.calculate_price_volatility(price_change_24h)
        liquidity_risk = self.calculate_liquidity_risk(total_volume, market_cap)
        concentration_risk = self.calculate_concentration_risk(circulating_supply, max_supply)
        
        # Additional risk factors based on Exponential Finance framework
        market_cap_risk = self.calculate_market_cap_risk(market_cap)
        short_term_volatility_risk = self.calculate_short_term_volatility_risk(price_change_1h, price_change_7d)
        momentum_risk = self.calculate_momentum_risk(price_change_1h, price_change_24h, price_change_7d)
        
        # Weighted risk calculation with proper scaling
        risk_weights = {
            'volatility': 0.25,      # 25% - Price volatility is critical
            'liquidity': 0.20,       # 20% - Liquidity affects exit ability
            'concentration': 0.15,    # 15% - Supply concentration risk
            'market_cap': 0.15,      # 15% - Market size and maturity
            'short_term_vol': 0.15,  # 15% - Short-term price stability
            'momentum': 0.10         # 10% - Price momentum patterns
        }
        
        # Calculate weighted average risk (normalized approach)
        risk_factors = {
            'volatility': volatility_risk,
            'liquidity': liquidity_risk,
            'concentration': concentration_risk,
            'market_cap': market_cap_risk,
            'short_term_vol': short_term_volatility_risk,
            'momentum': momentum_risk
        }
        
        # Calculate weighted average risk
        weighted_risk = 0
        for risk_type, weight in risk_weights.items():
            risk_value = risk_factors[risk_type]
            weighted_risk += weight * risk_value
        
        # Apply risk compounding factor (moderate exponential effect)
        # This ensures high-risk components have more impact but don't explode
        compounded_risk = weighted_risk * (1 + (weighted_risk / 100) ** 0.5)
        
        # Scale to 1-100 range with proper distribution
        # Use sigmoid-like scaling to ensure good distribution
        scaled_risk = 1 + (99 * (compounded_risk / 100) ** 0.7)
        
        # Ensure bounds and return
        final_risk = min(100, max(1, scaled_risk)) / 100.0
        
        return self.clamp(final_risk)

    def get_data(self, vs_currency='usd'):
        page = 1
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            'vs_currency': vs_currency,
            'order': 'market_cap_desc',
            'per_page': 50,
            'page': page,
            'price_change_percentage': '1h,24h,7d',
            'sparkline': False,
            'locale': 'en',
        }
        
        response = requests.get(url, params=params, timeout=50)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            return "Error"
        data = response.json()
        return data
    
    def process_data(self):
        data = self.get_data()
        result = []
        for coin in data:
            res = {}
            res['name'] = coin['name']
            res['id'] = coin['id']
            res['symbol'] = coin['symbol']
            res['risk'] = self.calculate_asset_risk(coin)
            res['current_price'] = coin['current_price']
            res['total_supply'] = coin['total_supply']
            res['circulating_supply'] = coin['circulating_supply']
            res['total_volume'] = coin['total_volume']
            res['market_cap'] = coin['market_cap']
            res['price_change_24h'] = coin['price_change_24h']
            result.append(res)
        with open(self.coin_data_local_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, sort_keys=True)
        return result
