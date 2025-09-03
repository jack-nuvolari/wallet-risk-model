import time
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import requests

current_folder = Path(__file__).parent


@dataclass
class Transaction:
    hash: str
    from_address: str
    to_address: str
    value: str
    timestamp: int
    block_number: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'hash': self.hash,
            'from': self.from_address,
            'to': self.to_address,
            'value': self.value,
            'timestamp': self.timestamp,
            'block_number': self.block_number
        }


@dataclass
class TokenTransfer:
    token: str
    token_name: str
    token_symbol: str
    from_address: str
    to_address: str
    value: str
    timestamp: int
    block_number: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'token': self.token,
            'token_name': self.token_name,
            'token_symbol': self.token_symbol,
            'from': self.from_address,
            'to': self.to_address,
            'value': self.value,
            'timestamp': self.timestamp,
            'block_number': self.block_number
        }


@dataclass
class GasPrice:
    safe_gwei: str
    propose_gwei: str
    fast_gwei: str


class EtherscanService:
    """Service class for interacting with Etherscan API with fallback to mock data."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.etherscan.io/api"
        self._mock_data_path = os.path.join(
            current_folder, "..", "..", "data", "wallet_summary_mock.json"
        )
    
    @property
    def wallet_local_path(self):
        return os.path.join(
            current_folder, "..", "..", "data", f"etherscan_wallet_summary.json"
        )
    
    def _validate_address(self, address: str) -> str:
        """Validate and normalize Ethereum address."""
        if not address or not address.startswith('0x') or len(address) != 42:
            raise ValueError(f"Invalid Ethereum address: {address}")
        return address.lower()
    
    def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to Etherscan API with error handling."""
        try:
            params['apikey'] = self.api_key
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}") from e
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse API response: {str(e)}")
    
    def _is_api_success(self, data: Dict[str, Any]) -> bool:
        """Check if API response indicates success."""
        return data.get('status') == "1" and data.get('result')
    
    def _load_mock_data(self, address: str) -> Optional[Dict[str, Any]]:
        """Load mock data for a specific address if available."""
        try:
            if os.path.exists(self._mock_data_path):
                with open(self._mock_data_path, 'r') as f:
                    mock_data = json.load(f)
                    if mock_data.get('address', '').lower() == address.lower():
                        return mock_data
        except Exception as e:
            print(f"Failed to load mock data: {e}")
        return None

    def get_address_balance(self, address: str) -> Dict[str, Any]:
        """Get ETH balance for an address with fallback to mock data."""
        try:
            valid_address = self._validate_address(address)
            
            params = {
                'module': 'account',
                'action': 'balance',
                'address': valid_address,
                'tag': 'latest'
            }
            
            data = self._make_request(params)
            
            if self._is_api_success(data):
                balance_wei = int(data['result'])
                balance_eth = balance_wei / (10 ** 18)
                
                return {
                    'address': valid_address,
                    'balance_in_wei': balance_wei,
                    'balance_in_eth': f"{balance_eth:.18f}"
                }
            
            print(f"Etherscan API error: {data.get('message', 'Failed to get balance')}")
            return self._get_mock_balance(valid_address)
            
        except Exception as e:
            print(f"Failed to get balance from API: {str(e)}")
            return self._get_mock_balance(address)

    def get_transaction_history(self, address: str, limit: int = 10) -> List[Transaction]:
        """Get transaction history for an address with fallback to mock data."""
        try:
            valid_address = self._validate_address(address)
            
            params = {
                'module': 'account',
                'action': 'txlist',
                'address': valid_address,
                'startblock': 0,
                'endblock': 99999999,
                'page': 1,
                'offset': limit,
                'sort': 'desc'
            }
            
            data = self._make_request(params)
            
            if self._is_api_success(data):
                return self._parse_transactions(data['result'], valid_address, limit)
            
            print(f"Etherscan API error for transactions: {data.get('message', 'Failed to fetch transactions')}")
            return self._get_mock_transactions(valid_address, limit)
            
        except Exception as e:
            print(f"Failed to get transaction history from API: {str(e)}")
            return self._get_mock_transactions(address, limit)
    
    def _parse_transactions(self, tx_data: List[Dict], address: str, limit: int) -> List[Transaction]:
        """Parse raw transaction data into Transaction objects."""
        transactions = []
        for tx in tx_data[:limit]:
            try:
                transaction = Transaction(
                    hash=tx['hash'],
                    from_address=tx['from'],
                    to_address=tx.get('to', 'Contract Creation'),
                    value=str(int(tx['value']) / (10 ** 18)),
                    timestamp=int(tx.get('timeStamp', 0)),
                    block_number=int(tx.get('blockNumber', 0))
                )
                transactions.append(transaction)
            except (KeyError, ValueError) as e:
                print(f"Warning: Skipping malformed transaction: {e}")
                continue
        
        return transactions
    
    def _get_mock_transactions(self, address: str, limit: int) -> List[Transaction]:
        """Generate mock transaction data."""
        mock_data = self._load_mock_data(address)
        max_mock_txs = 5 if mock_data else 3
        
        transactions = []
        for i in range(min(limit, max_mock_txs)):
            tx = Transaction(
                hash=f"0x{'0' * 64}",
                from_address=address,
                to_address=address,
                value="0.1",
                timestamp=int(time.time()) - (i * 3600),
                block_number=18000000 + i
            )
            transactions.append(tx)
        
        return transactions

    def get_token_transfers(self, address: str, limit: int = 10) -> List[TokenTransfer]:
        """Get token transfer history for an address with fallback to mock data."""
        try:
            valid_address = self._validate_address(address)
            
            params = {
                'module': 'account',
                'action': 'tokentx',
                'address': valid_address,
                'page': 1,
                'offset': limit,
                'sort': 'desc'
            }
            
            data = self._make_request(params)
            
            if self._is_api_success(data):
                return self._parse_token_transfers(data['result'], valid_address, limit)
            
            print(f"Etherscan API error for token transfers: {data.get('message', 'Failed to fetch token transfers')}")
            return self._get_mock_token_transfers(valid_address, limit)
            
        except Exception as e:
            print(f"Failed to get token transfers from API: {str(e)}")
            return self._get_mock_token_transfers(address, limit)
    
    def _parse_token_transfers(self, tx_data: List[Dict], address: str, limit: int) -> List[TokenTransfer]:
        """Parse raw token transfer data into TokenTransfer objects."""
        transfers = []
        for tx in tx_data[:limit]:
            try:
                token_decimals = int(tx.get('tokenDecimal', 18))
                token_value = int(tx['value']) / (10 ** token_decimals)
                
                transfer = TokenTransfer(
                    token=tx['contractAddress'],
                    token_name=tx.get('tokenName', 'Unknown'),
                    token_symbol=tx.get('tokenSymbol', 'Unknown'),
                    from_address=tx['from'],
                    to_address=tx['to'],
                    value=str(token_value),
                    timestamp=int(tx.get('timeStamp', 0)),
                    block_number=int(tx.get('blockNumber', 0))
                )
                transfers.append(transfer)
            except (KeyError, ValueError) as e:
                print(f"Warning: Skipping malformed token transfer: {e}")
                continue
        
        return transfers
    
    def _get_mock_token_transfers(self, address: str, limit: int) -> List[TokenTransfer]:
        """Generate mock token transfer data."""
        mock_data = self._load_mock_data(address)
        max_mock_transfers = 5 if mock_data else 3
        
        transfers = []
        for i in range(min(limit, max_mock_transfers)):
            transfer = TokenTransfer(
                token=f"0x{'0' * 40}",
                token_name="Mock Token",
                token_symbol="MTK",
                from_address=address,
                to_address=address,
                value="1.0",
                timestamp=int(time.time()) - (i * 3600),
                block_number=18000000 + i
            )
            transfers.append(transfer)
        
        return transfers
    
    def _get_mock_balance(self, address: str) -> Dict[str, Any]:
        """Get mock balance data for an address."""
        mock_data = self._load_mock_data(address)
        
        if mock_data and 'balance' in mock_data:
            return mock_data['balance']
        
        # Return default mock data
        return {
            'address': address,
            'balance_in_wei': 1000000000000000000,  # 1 ETH
            'balance_in_eth': "1.0"
        }
    
    def get_contract_abi(self, address: str) -> str:
        """Get contract ABI for a given address."""
        try:
            valid_address = self._validate_address(address)
            
            params = {
                'module': 'contract',
                'action': 'getabi',
                'address': valid_address
            }
            
            data = self._make_request(params)
            
            if not self._is_api_success(data):
                raise Exception(data.get('message', 'Failed to fetch contract ABI'))
            
            return data['result']
            
        except Exception as e:
            raise Exception(f"Failed to get contract ABI: {str(e)}")

    def get_gas_oracle(self) -> GasPrice:
        """Get current gas prices from Etherscan."""
        try:
            params = {
                'module': 'gastracker',
                'action': 'gasoracle'
            }
            
            data = self._make_request(params)
            
            if not self._is_api_success(data):
                raise Exception(data.get('message', 'Failed to fetch gas prices'))
            
            result = data['result']
            return GasPrice(
                safe_gwei=result.get('SafeGasPrice', '0'),
                propose_gwei=result.get('ProposeGasPrice', '0'),
                fast_gwei=result.get('FastGasPrice', '0')
            )
            
        except Exception as e:
            raise Exception(f"Failed to get gas prices: {str(e)}")

    def get_ens_name(self, address: str) -> Optional[str]:
        """Get ENS name for a given address (placeholder implementation)."""
        try:
            valid_address = self._validate_address(address)
            print(f"ENS resolution not implemented. Address: {valid_address}")
            return None
            
        except Exception as e:
            raise Exception(f"Failed to get ENS name: {str(e)}")

    def get_wallet_summary(self, address: str) -> Dict[str, Any]:
        """Get comprehensive wallet summary including balance, transactions, and token transfers."""
        try:
            valid_address = self._validate_address(address)
            
            # Fetch all data in parallel (could be optimized with async)
            balance = self.get_address_balance(valid_address)
            transactions = self.get_transaction_history(valid_address, limit=20)
            token_transfers = self.get_token_transfers(valid_address, limit=20)
            
            # Calculate metrics
            summary = self._calculate_wallet_metrics(
                valid_address, balance, transactions, token_transfers
            )
            
            # Save to local file
            self._save_wallet_summary(summary)
            
            return summary
            
        except Exception as e:
            raise Exception(f"Failed to get wallet summary: {str(e)}")

    def _calculate_wallet_metrics(self, address: str, balance: Dict, 
                                 transactions: List[Transaction], 
                                 token_transfers: List[TokenTransfer]) -> Dict[str, Any]:
        """Calculate wallet metrics and create summary."""
        current_time = int(time.time())
        thirty_days_ago = current_time - (30 * 24 * 60 * 60)
        
        recent_transactions = [tx for tx in transactions if tx.timestamp > thirty_days_ago]
        recent_token_transfers = [tx for tx in token_transfers if tx.timestamp > thirty_days_ago]
        
        return {
            'address': address,
            'balance': balance,
            'transaction_summary': {
                'total_transactions': len(transactions),
                'recent_transactions_30d': len(recent_transactions),
                'transactions': [tx.to_dict() for tx in transactions[:10]]
            },
            'token_summary': {
                'total_token_transfers': len(token_transfers),
                'recent_token_transfers_30d': len(recent_token_transfers),
                'token_transfers': [tx.to_dict() for tx in token_transfers[:10]]
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def _save_wallet_summary(self, summary: Dict[str, Any]) -> None:
        """Save wallet summary to local file."""
        try:
            with open(self.wallet_local_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=4, sort_keys=True)
        except Exception as e:
            print(f"Warning: Failed to save wallet summary: {e}")
