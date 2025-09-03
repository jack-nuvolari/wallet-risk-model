import os
import sys
import yaml
from typing import Dict, Any
import asyncio
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# All imports after path modification
try:
    from mcp.server.fastmcp import FastMCP
    from utils.walletRiskCalculator import WalletRiskCalculator
    from utils.collectors.uniswap import Uniswap
    from utils.collectors.aave import Aave
    from utils.collectors.coin import Coin
    from utils.collectors.news import News
    from utils.collectors.etherscan import EtherscanService
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load config
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_local.yml")

with open(config_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# Initialize the MCP server instance
server = FastMCP("risk_score_calculator")

@server.tool()
async def get_risk_summary(wallet_address: str = "") -> str:
    """
    Get risk summary for a given wallet address
    """
    try:
        print(f"Received request for wallet: {wallet_address}")
        if not wallet_address:
            return {"error": "No wallet address provided"}
            
        etherscan = EtherscanService(os.getenv("ETHERSCAN_API_KEY"))
        wallet_summary = etherscan.get_wallet_summary(wallet_address)
        calculator = WalletRiskCalculator()
        result = calculator.calculate_wallet_risk_score(wallet_summary)
        result = str(result)
        return result
    
    except (ValueError, KeyError, TypeError, OSError, IOError, AttributeError, UnicodeDecodeError) as e:
        error_msg = f"Failed to get risk summary: {str(e)}"
        return {"error": error_msg}

@server.tool()
async def update_risk_info() -> Dict[str, Any]:
    """
    Update all risk information from various sources
    """
    try:
        uniswap = Uniswap(config["uniswap"])
        uniswap.process_pools()
        
        aave = Aave(config["aave"])
        aave.process_markets()
        
        if "coin" in config:
            try:
                coin = Coin(config["coin"])
                coin.process_data()
            except (ValueError, KeyError, TypeError, OSError, IOError, AttributeError):
                pass
        
        if "news" in config:
            try:
                news = News(config["news"])
                news.process_data("ETH Price")
            except (ValueError, KeyError, TypeError, OSError, IOError, AttributeError):
                pass
        
        return {
            "status": "success",
            "message": "Risk info updated successfully",
            "updated_sources": ["uniswap", "aave"]
        }
        
    except (ValueError, KeyError, TypeError, OSError, IOError, AttributeError, UnicodeDecodeError) as e:
        return {"error": f"Failed to update risk info: {str(e)}"}

if __name__ == "__main__":
    asyncio.run(server.run())