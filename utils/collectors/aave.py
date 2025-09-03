import os
import json
from typing import Any, Dict, TypedDict
from functools import cache
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport

current_folder = os.path.realpath(os.path.dirname(__file__))

class AaveConfig(TypedDict):
    graphql: str


class Aave:
    _config: AaveConfig

    def __init__(self, config: AaveConfig):
        self._config = config

    @property
    def markets_local_path(self):
        return os.path.join(
            current_folder, "..", "..", "data", f"aave_markets_gql.json"
        )

    @property
    def extracted_markets_local_path(self):
        return os.path.join(
            current_folder, "..", "..", "data", f"aave_markets.json"
        )

    def _process_market(self, market: Dict[str, Any]):
        """Process individual Aave market data"""
        # Calculate risk metrics
        total_value_locked = float(market.get("totalValueLockedUSD", 0))
        total_deposit = float(market.get("totalDepositBalanceUSD", 0))
        total_borrow = float(market.get("totalBorrowBalanceUSD", 0))
        
        # Utilization rate
        utilization_rate = 0
        if total_deposit > 0:
            utilization_rate = total_borrow / total_deposit
        
        # Liquidation risk parameters (based on LTV and liquidation threshold)
        max_ltv = float(market.get("maximumLTV", 0)) / 10000  # Convert from basis points
        liquidation_threshold = float(market.get("liquidationThreshold", 0)) / 10000
        liquidation_penalty = float(market.get("liquidationPenalty", 0)) / 10000
        
        # Activity metrics
        transaction_count = int(market.get("transactionCount", 0))
        borrow_count = int(market.get("borrowCount", 0))
        
        # Calculate comprehensive risk metrics using the reference function
        risk_metrics = self._calculate_risk_metrics(
            collateral_value=total_deposit,
            borrowed_value=total_borrow,
            max_ltv=max_ltv,
            liquidation_threshold=liquidation_threshold,
            liquidation_penalty=liquidation_penalty
        )
        
        # Calculate comprehensive risk score similar to Uniswap
        # Combine multiple risk factors into a single risk value (0-1, where 1 is highest risk)
        
        # Factor 1: Utilization risk (higher utilization = higher risk)
        utilization_risk = min(utilization_rate, 1.0)
        
        # Factor 2: LTV risk (higher LTV = higher risk)
        ltv_risk = max_ltv if max_ltv > 0 else 0
        
        # Factor 3: Liquidation threshold risk (lower threshold = higher risk)
        liquidation_threshold_risk = 1 - liquidation_threshold if liquidation_threshold > 0 else 1
        
        # Factor 4: Liquidation penalty risk (higher penalty = higher risk)
        liquidation_penalty_risk = liquidation_penalty if liquidation_penalty > 0 else 0
        
        # Factor 5: Market size risk (smaller markets = higher risk)
        market_size_risk = 1 / (1 + total_value_locked / 1000000)  # Normalize by 1M USD
        
        # Factor 6: Activity risk (lower activity = higher risk)
        activity_risk = 1 / (1 + transaction_count / 1000) if transaction_count > 0 else 1
        
        # Factor 7: Health score risk (lower health = higher risk)
        health_score = risk_metrics['HealthScore']
        health_risk = 1 / (1 + health_score) if health_score > 0 else 1
        
        # Weighted combination of risk factors
        risk_weights = {
            'utilization': 0.25,      # 25% weight
            'ltv': 0.20,              # 20% weight  
            'liquidation_threshold': 0.15,  # 15% weight
            'liquidation_penalty': 0.10,    # 10% weight
            'market_size': 0.10,      # 10% weight
            'activity': 0.10,         # 10% weight
            'health': 0.10            # 10% weight
        }
        
        # Calculate weighted risk score
        risk_score = (
            utilization_risk * risk_weights['utilization'] +
            ltv_risk * risk_weights['ltv'] +
            liquidation_threshold_risk * risk_weights['liquidation_threshold'] +
            liquidation_penalty_risk * risk_weights['liquidation_penalty'] +
            market_size_risk * risk_weights['market_size'] +
            activity_risk * risk_weights['activity'] +
            health_risk * risk_weights['health']
        )
        
        # Ensure risk score is between 0 and 1
        risk_score = max(0, min(1, risk_score))
        
        # Market-specific risk score based on utilization and LTV (legacy)
        market_risk_score = utilization_rate * (1 - max_ltv)
        
        # Revenue metrics
        cumulative_revenue = float(market.get("cumulativeTotalRevenueUSD", 0))
        protocol_revenue = float(market.get("cumulativeProtocolSideRevenueUSD", 0))
        
        # Get daily snapshot data if available
        daily_snapshots = market.get("dailySnapshots", [])
        daily_deposit = 0
        daily_borrow = 0
        if daily_snapshots:
            snapshot = daily_snapshots[0]
            daily_deposit = float(snapshot.get("dailyDepositUSD", 0))
            daily_borrow = float(snapshot.get("dailyBorrowUSD", 0))
        
        # Get rates if available
        rates = market.get("rates", [])
        supply_rate = 0
        borrow_rate = 0
        for rate in rates:
            if rate.get("side") == "LENDER":
                supply_rate = float(rate.get("rate", 0)) / 10000  # Convert from basis points
            elif rate.get("side") == "BORROWER":
                borrow_rate = float(rate.get("rate", 0)) / 10000
        
        return {
            "id": market["id"],
            "risk": risk_score,  # Single risk value (0-1, similar to Uniswap)
            # "riskMetrics": risk_metrics
        }

    def _calculate_risk_metrics(self, collateral_value, borrowed_value, max_ltv, liquidation_threshold, liquidation_penalty):
        """
        Calculate the risk metrics for an Aave lending position.

        Parameters:
        collateral_value (float): The current value of the collateral in USD.
        borrowed_value (float): The current borrowed value in USD.
        max_ltv (float): The maximum loan-to-value ratio (0-1).
        liquidation_threshold (float): The liquidation threshold (0-1).
        liquidation_penalty (float): The liquidation penalty as a decimal (e.g. 0.05 for 5%).

        Returns:
        dict: A dictionary with the following risk metrics:
            - LTV (Loan to Value Ratio)
            - Health Score
            - Liquidation Risk (True/False)
            - Liquidation Penalty (USD value to pay)
        """
        # Calculate LTV (Loan to Value Ratio)
        ltv = borrowed_value / collateral_value if collateral_value > 0 else 0

        # Calculate Health Score
        health_score = collateral_value * liquidation_threshold / borrowed_value if borrowed_value > 0 else float('inf')

        # Check if liquidation risk exists (if Health Score < 1, then liquidation is possible)
        liquidation_risk = health_score < 1

        # If liquidation happens, calculate Liquidation Penalty
        liquidation_penalty_value = 0
        if liquidation_risk:
            liquidation_value = collateral_value * liquidation_penalty
            borrowed_value_after_liquidation = borrowed_value - (borrowed_value * (1 - liquidation_penalty))
            liquidation_penalty_value = liquidation_value

        # Additional risk metrics
        ltv_ratio_to_max = ltv / max_ltv if max_ltv > 0 else 0
        safety_margin = liquidation_threshold - ltv if ltv > 0 else liquidation_threshold
        
        # Risk level classification
        risk_level = "LOW"
        if liquidation_risk:
            risk_level = "HIGH"
        elif ltv_ratio_to_max > 0.8:
            risk_level = "MEDIUM"
        elif ltv_ratio_to_max > 0.6:
            risk_level = "LOW"
        else:
            risk_level = "VERY_LOW"

        # Return all metrics in a dictionary
        return {
            'LTV': ltv,
            'LTVRatioToMax': ltv_ratio_to_max,
            'HealthScore': health_score,
            'LiquidationRisk': liquidation_risk,
            'LiquidationPenaltyUSD': liquidation_penalty_value,
            'SafetyMargin': safety_margin,
            'RiskLevel': risk_level,
            'MaxLTV': max_ltv,
            'LiquidationThreshold': liquidation_threshold
        }

    def process_markets(self):
        data = self.fetch_all_markets()
        res = [self._process_market(item) for item in data]
        with open(self.extracted_markets_local_path, "w") as f:
            json.dump(res, f, indent=4, sort_keys=True)
        return res

    def fetch_all_markets(self):
        """Fetch all Aave markets"""
        first = 50
        markets = self.get_markets(first)  # Fetch top 50 markets by TVL
        with open(self.markets_local_path, "w") as f:
            json.dump(markets, f, indent=4, sort_keys=True)
        return markets

    def get_markets(self, first: int = 50):
        """Get Aave markets from GraphQL"""
        query = gql(
            """
        query getMarkets{
            markets(orderBy: totalValueLockedUSD, orderDirection: desc) {
                id
                name
                isActive
                borrowCount
                cumulativeBorrowUSD
                cumulativeDepositUSD
                maximumLTV
                transactionCount
                totalValueLockedUSD
                totalDepositBalanceUSD
                totalBorrowBalanceUSD
                
                # Market size and rates
                inputTokenBalance
                inputTokenPriceUSD
                outputTokenSupply
                
                # Revenue metrics
                cumulativeSupplySideRevenueUSD
                cumulativeProtocolSideRevenueUSD
                cumulativeTotalRevenueUSD
                
                # Risk parameters
                liquidationThreshold
                liquidationPenalty
                
                # Token info
                inputToken {
                    lastPriceUSD
                    name
                    symbol
                }
                
                # Historical performance
                dailySnapshots(first: 1, orderBy: timestamp, orderDirection: desc) {
                    dailyDepositUSD
                    dailyBorrowUSD
                    dailyRepayUSD
                    dailyWithdrawUSD
                }
                
                # APR value for borrow and supply
                rates(orderDirection: asc) {
                    rate
                    side
                    tranche
                    type
                }
            }
        }
        """
        # % (first)
        )
        client = self._client()
        retries = 3
        while True:
            try:
                result = client.execute(query, variable_values={"first": first})
                print(f"Fetched {len(result['markets'])} Aave markets")
                return result["markets"]
            except Exception as e:
                retries -= 1
                if retries == 0:
                    raise e

    @cache
    def _client(self) -> Client:
        transport = RequestsHTTPTransport(url=self._config["graphql"])
        return Client(transport=transport, fetch_schema_from_transport=False)