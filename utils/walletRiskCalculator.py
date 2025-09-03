import json
import os
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict


class WalletRiskCalculator:
    "wallet risk calculator"
    def __init__(self):
        self.risk_weights = {
            'transaction_frequency': 0.15,
            'transaction_amounts': 0.15,
            'interaction_patterns': 0.15,
            'recent_activity': 0.10,
            'contract_interactions': 0.15,
            'balance_risk': 0.20,
            'defi_positions': 0.10,
        }
        
        self.thresholds = {
            'high_frequency': 50,
            'medium_frequency': 20,
            'high_amount': 10.0,
            'medium_amount': 1.0,
            'suspicious_contracts': ['0x0000000000000068f116a894984e2db1123eb395'],
            'recent_days': 30
        }
    
    def calculate_wallet_risk_score(self, wallet_summary: Dict[str, Any]) -> any:
        """Calculate comprehensive risk score for a wallet."""
        try:
            transactions = wallet_summary.get('transaction_summary', {}).get('transactions', [])
            token_transfers = wallet_summary.get('token_summary', {}).get('token_transfers', [])
            balance = wallet_summary.get('balance', {})
            defi_positions = wallet_summary.get('defi_positions', {})
            
            risk_components = {
                'frequency_risk': self._calculate_frequency_risk(transactions),
                'amount_risk': self._calculate_amount_risk(transactions, balance),
                'interaction_risk': self._calculate_interaction_risk(transactions, token_transfers),
                'recent_activity_risk': self._calculate_recent_activity_risk(transactions, token_transfers),
                'contract_risk': self._calculate_contract_interaction_risk(transactions),
                'balance_risk': self._calculate_balance_risk(balance),
                'defi_position_risk': self._calculate_position_risk(defi_positions)
            }
            
            total_risk_score = sum(
                component['score'] * self.risk_weights[weight_key]
                for weight_key, component in zip(self.risk_weights.keys(), risk_components.values())
            )
            
            final_risk_score = max(1, min(100, round(total_risk_score)))
            risk_level = self._get_risk_level(final_risk_score)
            
            result = {
                'overall_risk_score': final_risk_score,
                'risk_level': risk_level,
                'risk_components': risk_components,
                'analysis_timestamp': datetime.now().isoformat()
            }

            self._save_result(result)
            return result
            
        except (ValueError, KeyError, TypeError, OSError, IOError) as e:
            return {
                'error': f"Failed to calculate risk score: {str(e)}",
                'overall_risk_score': 50,
                'risk_level': 'medium'
            }
    
    def _calculate_position_risk(self, defi_positions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk based on DeFi positions."""
        if not defi_positions:
            return {'score': 0, 'details': 'No DeFi positions found'}
        
        try:
            aave_risks = self._load_aave_risks()
            uniswap_risks = self._load_uniswap_risks()
            
            total_position_value = 0
            weighted_risk_sum = 0
            
            for protocol, positions in defi_positions.items():
                for position in positions:
                    position_value = self._get_position_value(position)
                    position_risk = self._get_position_risk_score(position, protocol, aave_risks, uniswap_risks)
                    weighted_risk_sum += position_risk * position_value
                    total_position_value += position_value
            
            overall_risk = (weighted_risk_sum / total_position_value * 100) if total_position_value > 0 else 0
            
            return {
                'score': overall_risk,
                'details': {
                    'total_position_value_usd': total_position_value,
                    'overall_risk_score': overall_risk,
                    'position_count': sum(len(positions) for positions in defi_positions.values())
                }
            }
            
        except (ValueError, KeyError, TypeError, OSError, IOError) as e:
            return {'score': 0, 'details': f'Error calculating position risk: {str(e)}'}
    
    def _load_aave_risks(self) -> Dict[str, float]:
        """Load Aave market risk scores."""
        try:
            aave_path = os.path.join('data', 'aave_markets.json')
            if os.path.exists(aave_path):
                with open(aave_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return {item['id']: item['risk'] for item in data}
            return {}
        except (OSError, IOError, json.JSONDecodeError):
            return {}
    
    def _load_uniswap_risks(self) -> Dict[str, float]:
        """Load Uniswap pool risk scores."""
        try:
            uniswap_path = os.path.join('data', 'uniswap_pools.json')
            if os.path.exists(uniswap_path):
                with open(uniswap_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return {item['id']: item['risk'] for item in data}
            return {}
        except (OSError, IOError, json.JSONDecodeError):
            return {}
    
    def _get_position_value(self, position: Dict[str, Any]) -> float:
        """Extract position value in USD."""
        value_fields = ['position_value_usd', 'total_value_usd', 'value_usd']
        for field in value_fields:
            if field in position:
                return float(position[field])
        
        if 'total_supplied_usd' in position:
            supplied = float(position.get('total_supplied_usd', 0))
            borrowed = float(position.get('total_borrowed_usd', 0))
            return supplied + borrowed
        
        return 0.0
    
    def _get_position_risk_score(self, position: Dict[str, Any], protocol: str, aave_risks: Dict[str, float], uniswap_risks: Dict[str, float]) -> float:
        """Get risk score for a specific position."""
        position_id = position.get('id', '')
        
        if protocol == 'aave_v3' and position_id in aave_risks:
            return aave_risks[position_id]
        elif protocol == 'uniswap_v3' and position_id in uniswap_risks:
            return uniswap_risks[position_id]
        
        risk_level = position.get('risk_level', 'medium')
        risk_mapping = {
            'very_low': 0.2, 'low': 0.4, 'medium': 0.6, 'high': 0.8, 'very_high': 1.0
        }
        
        return risk_mapping.get(risk_level, 0.6)
    
    def _calculate_frequency_risk(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate risk based on transaction frequency patterns."""
        if not transactions:
            return {'score': 0, 'details': 'No transactions found'}
        
        daily_transactions = defaultdict(int)
        for tx in transactions:
            tx_date = datetime.fromtimestamp(tx['timestamp']).date()
            daily_transactions[tx_date] += 1
        
        total_days = len(daily_transactions) or 1
        avg_daily_tx = len(transactions) / total_days
        max_daily_tx = max(daily_transactions.values()) if daily_transactions else 0
        
        frequency_score = 0
        
        if avg_daily_tx > self.thresholds['high_frequency'] / 30:
            frequency_score += 40
        elif avg_daily_tx > self.thresholds['medium_frequency'] / 30:
            frequency_score += 20
        
        if max_daily_tx > 10:
            frequency_score += 30
        elif max_daily_tx > 5:
            frequency_score += 15
        
        if avg_daily_tx < 0.1:
            frequency_score += 10
        
        return {
            'score': min(100, frequency_score),
            'details': {
                'total_transactions': len(transactions),
                'total_days': total_days,
                'avg_daily_transactions': round(avg_daily_tx, 2),
                'max_daily_transactions': max_daily_tx,
                'frequency_pattern': self._classify_frequency_pattern(avg_daily_tx)
            }
        }
    
    def _calculate_amount_risk(self, transactions: List[Dict[str, Any]], balance: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk based on transaction amounts and patterns."""
        if not transactions:
            return {'score': 0, 'details': 'No transactions found'}
        
        amounts = []
        for tx in transactions:
            try:
                amount = float(tx['value'])
                amounts.append(amount)
            except (ValueError, KeyError):
                continue
        
        if not amounts:
            return {'score': 0, 'details': 'No valid amounts found'}
        
        total_volume = sum(amounts)
        avg_amount = total_volume / len(amounts)
        max_amount = max(amounts)
        current_balance = float(balance.get('balance_in_eth', 0))
        
        amount_score = 0
        
        if max_amount > self.thresholds['high_amount']:
            amount_score += 30
        elif max_amount > self.thresholds['medium_amount']:
            amount_score += 15
        
        if current_balance > 0:
            volume_to_balance_ratio = total_volume / current_balance
            if volume_to_balance_ratio > 100:
                amount_score += 25
            elif volume_to_balance_ratio > 50:
                amount_score += 15
            elif volume_to_balance_ratio > 10:
                amount_score += 5
        
        if len(amounts) > 1:
            amount_variance = sum((x - avg_amount) ** 2 for x in amounts) / len(amounts)
            if amount_variance > 100:
                amount_score += 20
        
        zero_amount_count = sum(1 for amount in amounts if amount == 0)
        if zero_amount_count > len(amounts) * 0.3:
            amount_score += 10
        
        return {
            'score': min(100, amount_score),
            'details': {
                'total_volume_eth': round(total_volume, 4),
                'average_amount': round(avg_amount, 4),
                'max_amount': round(max_amount, 4),
                'current_balance': round(current_balance, 4),
                'volume_to_balance_ratio': round(volume_to_balance_ratio if current_balance > 0 else 0, 2),
                'zero_amount_transactions': zero_amount_count
            }
        }
    
    def _calculate_interaction_risk(self, transactions: List[Dict[str, Any]], token_transfers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate risk based on interaction patterns with other addresses."""
        if not transactions and not token_transfers:
            return {'score': 0, 'details': 'No interactions found'}
        
        all_addresses = set()
        interaction_counts = defaultdict(int)
        
        for tx in transactions:
            all_addresses.update([tx['from'], tx['to']])
            interaction_counts[tx['to']] += 1
            interaction_counts[tx['from']] += 1
        
        for transfer in token_transfers:
            all_addresses.update([transfer['from'], transfer['to']])
            interaction_counts[transfer['to']] += 1
            interaction_counts[transfer['from']] += 1
        
        unique_addresses = len(all_addresses)
        total_interactions = len(transactions) + len(token_transfers)
        
        interaction_score = 0
        
        if unique_addresses > 50:
            interaction_score += 25
        elif unique_addresses > 20:
            interaction_score += 15
        elif unique_addresses > 10:
            interaction_score += 5
        
        if interaction_counts:
            max_interactions = max(interaction_counts.values())
            if max_interactions > total_interactions * 0.5:
                interaction_score += 20
            elif max_interactions > total_interactions * 0.3:
                interaction_score += 10
        
        if total_interactions > 0:
            new_address_ratio = unique_addresses / total_interactions
            if new_address_ratio > 2:
                interaction_score += 15
        
        return {
            'score': min(100, interaction_score),
            'details': {
                'unique_addresses': unique_addresses,
                'total_interactions': total_interactions,
                'interaction_diversity': round(unique_addresses / total_interactions if total_interactions > 0 else 0, 2),
                'most_interactive_address': max(interaction_counts.items(), key=lambda x: x[1]) if interaction_counts else None
            }
        }
    
    def _calculate_recent_activity_risk(self, transactions: List[Dict[str, Any]], token_transfers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate risk based on recent activity patterns."""
        import time
        current_time = int(time.time())
        recent_cutoff = current_time - (self.thresholds['recent_days'] * 24 * 60 * 60)
        
        recent_transactions = [tx for tx in transactions if tx['timestamp'] > recent_cutoff]
        recent_token_transfers = [tx for tx in token_transfers if tx['timestamp'] > recent_cutoff]
        
        total_recent = len(recent_transactions) + len(recent_token_transfers)
        total_all = len(transactions) + len(token_transfers)
        
        recent_score = 0
        
        if total_all > 0:
            recent_ratio = total_recent / total_all
            if recent_ratio > 0.8:
                recent_score += 25
            elif recent_ratio > 0.5:
                recent_score += 15
            elif recent_ratio < 0.1:
                recent_score += 10
        
        if recent_transactions:
            daily_recent = defaultdict(int)
            for tx in recent_transactions:
                tx_date = datetime.fromtimestamp(tx['timestamp']).date()
                daily_recent[tx_date] += 1
            
            max_recent_daily = max(daily_recent.values()) if daily_recent else 0
            if max_recent_daily > 5:
                recent_score += 20
        
        return {
            'score': min(100, recent_score),
            'details': {
                'recent_transactions': len(recent_transactions),
                'recent_token_transfers': len(recent_token_transfers),
                'total_recent_activity': total_recent,
                'recent_activity_ratio': round(total_recent / total_all if total_all > 0 else 0, 2),
                'max_recent_daily_activity': max_recent_daily if 'max_recent_daily' in locals() else 0
            }
        }
    
    def _calculate_contract_interaction_risk(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate risk based on smart contract interactions."""
        if not transactions:
            return {'score': 0, 'details': 'No transactions found'}
        
        contract_interactions = []
        suspicious_contracts = []
        
        for tx in transactions:
            if tx['to'] and tx['to'] != 'Contract Creation':
                if float(tx['value']) == 0:
                    contract_interactions.append(tx['to'])
                
                if tx['to'] in self.thresholds['suspicious_contracts']:
                    suspicious_contracts.append(tx['to'])
        
        contract_score = 0
        
        if contract_interactions:
            contract_ratio = len(contract_interactions) / len(transactions)
            if contract_ratio > 0.7:
                contract_score += 25
            elif contract_ratio > 0.5:
                contract_score += 15
            elif contract_ratio > 0.3:
                contract_score += 5
        
        if suspicious_contracts:
            contract_score += 30
        
        zero_value_count = sum(1 for tx in transactions if float(tx['value']) == 0)
        if zero_value_count > len(transactions) * 0.5:
            contract_score += 20
        
        return {
            'score': min(100, contract_score),
            'details': {
                'contract_interactions': len(contract_interactions),
                'contract_interaction_ratio': round(len(contract_interactions) / len(transactions) if transactions else 0, 2),
                'suspicious_contracts': suspicious_contracts,
                'zero_value_transactions': zero_value_count,
                'zero_value_ratio': round(zero_value_count / len(transactions) if transactions else 0, 2)
            }
        }
    
    def _classify_frequency_pattern(self, avg_daily_tx: float) -> str:
        """Classify transaction frequency pattern."""
        if avg_daily_tx > 2:
            return 'high_frequency'
        elif avg_daily_tx > 0.5:
            return 'medium_frequency'
        elif avg_daily_tx > 0.1:
            return 'low_frequency'
        else:
            return 'very_low_frequency'
    
    def _calculate_balance_risk(self, balance: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk based on balance data and coin risk scores."""
        try:
            coin_data = self._load_coin_data()
            if not coin_data:
                return {'score': 0, 'details': 'Unable to load coin risk data'}
            
            token_balances = self._extract_token_balances(balance)
            if not token_balances:
                return {'score': 0, 'details': 'No token balances found'}
            
            total_value = 0
            weighted_risk_sum = 0
            token_risks = []
            
            for token_symbol, balance_data in token_balances.items():
                matching_coin = self._find_matching_coin(coin_data, token_symbol)
                
                if matching_coin:
                    coin_risk = float(matching_coin.get('risk', 0.5))
                    current_price = float(matching_coin.get('current_price', 0))
                    token_value = float(balance_data) * current_price
                    
                    token_risks.append({
                        'symbol': token_symbol,
                        'name': matching_coin.get('name', 'Unknown'),
                        'risk_score': coin_risk,
                        'current_price': current_price,
                        'balance': balance_data,
                        'value_usd': token_value
                    })
                    total_value += token_value
                else:
                    token_risks.append({
                        'symbol': token_symbol,
                        'name': 'Unknown Token',
                        'risk_score': 0.8,
                        'current_price': 0,
                        'balance': balance_data,
                        'value_usd': 0
                    })
            
            if total_value > 0:
                for token in token_risks:
                    if token['name'] != 'Unknown Token':
                        token_weight = token['value_usd'] / total_value
                        weighted_risk_sum += token['risk_score'] * token_weight * 100
                    else:
                        weighted_risk_sum += 50 / len(token_risks)
            
            return {
                'score': min(100, max(0, weighted_risk_sum)),
                'details': f'Portfolio risk based on {len(token_risks)} tokens',
                'token_breakdown': token_risks,
                'total_portfolio_value': total_value
            }
                        
        except (ValueError, KeyError, TypeError, OSError, IOError) as e:
            return {
                'score': 50,
                'details': f'Error calculating balance risk: {str(e)}',
                'token_breakdown': [],
                'total_portfolio_value': 0
            }
    
    def _load_coin_data(self) -> List[Dict[str, Any]]:
        """Load coin risk data from coin_data.json."""
        try:
            coin_data_path = os.path.join('data', 'coin_data.json')
            if os.path.exists(coin_data_path):
                with open(coin_data_path, 'r') as f:
                    return json.load(f)
            return []
        except (OSError, IOError, json.JSONDecodeError):
            return []
    
    def _extract_token_balances(self, balance: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract token balances from wallet balance data."""
        return {
            balance_type[11:]: balance_data
            for balance_type, balance_data in balance.items()
            if balance_type.startswith('balance_in_') and balance_type != 'balance_in_wei'
        }
    
    def _find_matching_coin(self, coin_data: List[Dict[str, Any]], token_symbol: str) -> Dict[str, Any]:
        """Find matching coin in coin_data by symbol (case-insensitive)."""
        normalized_search_symbol = token_symbol.lower()
        
        for coin in coin_data:
            coin_symbol = coin.get('symbol', '').lower()
            if coin_symbol == normalized_search_symbol:
                return coin
        
        for coin in coin_data:
            coin_symbol = coin.get('symbol', '').lower()
            if normalized_search_symbol in coin_symbol or coin_symbol in normalized_search_symbol:
                return coin
        
        return None
    
    def _get_risk_level(self, risk_score: int) -> str:
        """Convert risk score to risk level."""
        if risk_score <= 20:
            return 'very_low'
        elif risk_score <= 40:
            return 'low'
        elif risk_score <= 60:
            return 'medium'
        elif risk_score <= 80:
            return 'high'
        else:
            return 'very_high'
    
    def _save_result(self, result: Dict[str, Any]) -> None:
        """Save risk assessment result to file."""
        try:
            with open("data/wallet_risk_score.json", "w", encoding='utf-8') as f:
                json.dump(result, f, indent=4, sort_keys=True)
        except (OSError, IOError):
            pass


