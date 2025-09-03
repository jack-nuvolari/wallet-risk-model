import os
from typing import Dict, TypedDict, Any
import json
import statistics
import math
from functools import cache
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport

current_folder = os.path.realpath(os.path.dirname(__file__))

class UniswapConfig(TypedDict):
    graphql: Dict[int, str]


class Uniswap:
    _config: UniswapConfig

    def __init__(self, config: UniswapConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = config

    @property
    def pools_local_path(self):
        return os.path.join(
            current_folder, "..", "..", "data", f"uniswap_pools_gql.json"
        )

    @property
    def extracted_pools_local_path(self):
        return os.path.join(
            current_folder, "..", "..", "data", f"uniswap_pools.json"
        )
        
    def _process_pool(self, pool: Dict[str, Any]):
        price_factor = 10 ** (
            (int(pool["token0"]["decimals"]) - int(pool["token1"]["decimals"])) / 2
        )
        price_feed = [
            (int(item["sqrtPrice"])) ** 2 / 2**192 * price_factor**2
            for item in pool["poolDayData"]
        ]
        price_feed_returns = []
        if len(price_feed) > 2:
            price_feed_returns = [
                price_feed[i] / price_feed[i - 1] - 1 if price_feed[i - 1] != 0 else 0
                for i in range(1, len(price_feed))
            ]
            price_std_dev_annual = statistics.stdev(price_feed_returns) * math.sqrt(365)
        else:
            price_std_dev_annual = 0

        avg_liquidity_wei = statistics.mean(
            [int(item["liquidity"]) for item in pool["poolDayData"]]
        )
        liq_factor = 10 ** (
            (-int(pool["token0"]["decimals"]) - int(pool["token1"]["decimals"])) / 2
        )
        avg_liquidity = avg_liquidity_wei * liq_factor
        liquidity = int(pool["liquidity"]) * liq_factor

        token0_tvl_usd = float(pool["token0"]["totalValueLockedUSD"])
        token1_tvl_usd = float(pool["token1"]["totalValueLockedUSD"])
        token0_price_usd = 0
        token1_price_usd = 0
        normalized_tvl_usd_0 = 0
        normalized_tvl_usd_1 = 0
        normalized_avg_tvl_usd_0 = 0
        normalized_avg_tvl_usd_1 = 0

        sqrt_price = int(pool["sqrtPrice"]) * price_factor / 2**96
        if token1_tvl_usd > 0:
            token1_price_usd = (
                float(pool["token1"]["totalValueLocked"]) / token1_tvl_usd
            )
            normalized_tvl_usd_1 = liquidity * sqrt_price / token1_price_usd
            normalized_avg_tvl_usd_1 = avg_liquidity * sqrt_price / token1_price_usd
        if token0_tvl_usd > 0:
            token0_price_usd = (
                float(pool["token0"]["totalValueLocked"]) / token0_tvl_usd
            )
            normalized_tvl_usd_0 = liquidity / sqrt_price / token0_price_usd
            normalized_avg_tvl_usd_0 = avg_liquidity / sqrt_price / token0_price_usd

        tvl_norm_factor_0 = normalized_tvl_usd_0 / float(pool["totalValueLockedUSD"])
        tvl_norm_factor_1 = normalized_tvl_usd_1 / float(pool["totalValueLockedUSD"])
        tvl_norm_factor_0_avg = normalized_avg_tvl_usd_0 / float(
            pool["totalValueLockedUSD"]
        )
        tvl_norm_factor_1_avg = normalized_avg_tvl_usd_1 / float(
            pool["totalValueLockedUSD"]
        )
        normalized_tvl_usd = (normalized_tvl_usd_1 + normalized_tvl_usd_0) / 2
        if normalized_tvl_usd / float(pool["totalValueLockedUSD"]) < 0.01:
            normalized_tvl_usd = float(pool["totalValueLockedUSD"])
        last_month_fees = statistics.mean(
            [float(day["feesUSD"]) for day in pool["poolDayData"]]
        )
        returns = 0
        if normalized_tvl_usd > 0:
            returns = last_month_fees * 12 / normalized_tvl_usd
        risk = 1 - 2 * math.sqrt(1 + price_std_dev_annual) / (2 + price_std_dev_annual)
        sharpe = 0
        if risk > 0:
            sharpe = returns / risk
        return {
            "id": pool["id"],
            "risk": risk,
            "sharpe": sharpe,
        }

    def process_pools(self):
        data = self.fetch_all_pools()
        res = [self._process_pool(item) for item in data]
        with open(self.extracted_pools_local_path, "w") as f:
            json.dump(res, f, indent=4, sort_keys=True)
        return res

    def fetch_all_pools(self):
        # Use chain ID 1 (Ethereum mainnet) as default
        chain_id = 1
        res = self.get_all_pools_for_chain(chain_id)
        with open(self.pools_local_path,"w") as f:
            json.dump(res, f, indent=4, sort_keys=True)
        return res

    def get_all_pools_for_chain(self, chain_id: int):
        first = 50
        pools = self.get_pools(first, chain_id, 180)
        return pools

    def get_pools(self, first: int, chain_id: int, pool_data_days: int):
        query = gql(
            """
        query getPools{
            pools(
                first: 100
                orderBy: feesUSD
                orderDirection: desc
            ) {
                id
                token0 {
                    id
                    name
                    symbol
                    decimals
                    totalValueLocked
                    totalValueLockedUSD
                }
                token1 {
                    id
                    name
                    symbol
                    decimals
                    totalValueLocked
                    totalValueLockedUSD
                }
                feeTier
                liquidity
                sqrtPrice
                token0Price
                token1Price
                totalValueLockedUSD
                totalValueLockedToken0
                totalValueLockedToken1
                volumeUSD
                feesUSD
                poolDayData(first: %s, orderBy:date, orderDirection: desc) {
                    date
                    sqrtPrice
                    liquidity
                    feesUSD
                }
            }
        }
        """
            % (pool_data_days)
        )
        client = self._client(chain_id)
        retries = 3
        while True:
            try:
                result = client.execute(query)
                # Only print summary, not full data
                print(f"Fetched {len(result['pools'])} pools for chain {chain_id}")
                return result["pools"]
            except Exception as e:
                retries -= 1
                if retries == 0:
                    raise e

    @cache
    def _client(self, chain_id: int) -> Client:
        transport = RequestsHTTPTransport(url=self._config["graphql"])
        return Client(transport=transport, fetch_schema_from_transport=False)