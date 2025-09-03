import yaml
import json
import argparse
import os
from utils.collectors.uniswap import Uniswap
from utils.collectors.aave import Aave
from utils.collectors.coin import Coin
from utils.collectors.news import News
from utils.collectors.etherscan import EtherscanService
from utils.walletRiskCalculator import WalletRiskCalculator
from dotenv import load_dotenv
load_dotenv()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Wallet Risk Model - Data Collection and Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update all data sources
  python test.py --update-all
  
  # Train the coin risk model
  python test.py --train-model
  
  # Analyze specific wallet
  python test.py --wallet 0x7a29aE65Bf25Dfb6e554BF0468a6c23ed99a8DC2
  
  # Update only specific data sources
  python test.py --update-uniswap --update-coins
  
  # Run with custom config
  python test.py --config custom_config_local.yml --update-all
        """
    )
    
    # Data update options
    parser.add_argument('--update-all', action='store_true',
                       help='Update all data sources (uniswap, aave, coins, news)')
    parser.add_argument('--update-uniswap', action='store_true',
                       help='Update Uniswap pool data')
    parser.add_argument('--update-aave', action='store_true',
                       help='Update Aave market data')
    parser.add_argument('--update-coins', action='store_true',
                       help='Update cryptocurrency data')
    parser.add_argument('--update-news', action='store_true',
                       help='Update news data')
    
    # Model training options
    parser.add_argument('--train-model', action='store_true',
                       help='Train the coin risk regression model')
    parser.add_argument('--train-pairwise-model', action='store_true',
                       help='Train the pairwise risk model')
    parser.add_argument('--train-risk-weight-model', action='store_true',
                       help='Train the risk weight model')
    # Wallet analysis options
    parser.add_argument('--wallet', type=str, metavar='ADDRESS',
                       help='Analyze risk for specific wallet address')
    parser.add_argument('--wallet-mock', action='store_true',
                       help='Use mock wallet data for analysis')
    
    # Configuration options
    parser.add_argument('--config', type=str, default='config_local.yml',
                       help='Path to configuration file (default: config_local.yml)')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory for data (default: data)')
    
    # Verbose and debug options
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in configuration file: {e}")
        return None

def update_data_sources(config, args):
    """Update data sources based on arguments."""
    updated_sources = []
    
    if args.update_all or args.update_uniswap:
        print("Updating Uniswap pool data...")
        try:
            uniswap = Uniswap(config["uniswap"])
            uniswap.process_pools()
            updated_sources.append("Uniswap")
            print("Uniswap data updated successfully")
        except Exception as e:
            print(f"Error updating Uniswap data: {e}")
    
    if args.update_all or args.update_aave:
        print("Updating Aave market data...")
        try:
            aave = Aave(config["aave"])
            aave.process_markets()
            updated_sources.append("Aave")
            print("Aave data updated successfully")
        except Exception as e:
            print(f"Error updating Aave data: {e}")
    
    if args.update_all or args.update_coins:
        print("Updating cryptocurrency data...")
        try:
            coin = Coin()
            coin.process_data()
            updated_sources.append("Coins")
            print("Cryptocurrency data updated successfully")
        except Exception as e:
            print(f"Error updating cryptocurrency data: {e}")
    
    if args.update_all or args.update_news:
        print("Updating news data...")
        try:
            news = News(config["news"])
            news_result = news.process_data("ETH Price")
            if args.verbose:
                print(f"News data: {news_result}")
            updated_sources.append("News")
            print("News data updated successfully")
        except Exception as e:
            print(f"Error updating news data: {e}")
    
    return updated_sources

def train_coin_risk_model():
    """Train the coin risk regression model."""
    try:
        print("Training coin risk regression model...")
        from ai.coinRiskModel import train
        train()
        print("Coin risk model trained successfully")
        return True
    except Exception as e:
        print(f"Error training coin risk model: {e}")
        return False

def pair_wise_risk_model():
    """Train the pairwise risk model."""
    try:
        print("Training pairwise risk model...")
        from ai.pairWiseModel import train
        import pandas as pd
        coin_data = pd.read_json("data/coin_data.json")
        train()
        print("Pairwise risk model trained successfully")
        return True
    except Exception as e:
        print(f"Error training pairwise risk model: {e}")
        return False

def risk_weight_model():

    try:
        print("Training risk weight model...")
        from ai.riskWeightModel import train
        train()
        print("Risk weight model trained successfully")
        return True
    except Exception as e:
        print(f"Error training risk weight model: {e}")
        return False

def analyze_wallet_risk_safe(wallet_address, config):
    """Safely analyze wallet risk."""
    try:
        etherscan = EtherscanService(os.getenv("ETHERSCAN_API_KEY"))
        if wallet_address != "mock":
            wallet_summary = etherscan.get_wallet_summary(wallet_address)
        else:
            with open("data/wallet_mock_summary.json", "r", encoding="utf-8") as f:
                wallet_summary = json.load(f)
        risk_analysis = WalletRiskCalculator().calculate_wallet_risk_score(wallet_summary)
        return risk_analysis
    except Exception as e:
        print(f"Error analyzing wallet risk: {e}")
        return None

def main():
    """Main function."""
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    if config is None:
        return
    
    if args.verbose:
        print(f"Configuration loaded from: {args.config}")
        print(f"Output directory: {args.output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Update data sources if requested
    if any([args.update_all, args.update_uniswap, args.update_aave, args.update_coins, args.update_news]):
        print("=" * 50)
        print("UPDATING DATA SOURCES")
        print("=" * 50)
        updated_sources = update_data_sources(config, args)
        
        if updated_sources:
            print(f"\nSuccessfully updated: {', '.join(updated_sources)}")
        else:
            print("\nNo data sources were updated successfully")
    
    # Train model if requested
    if args.train_model:
        print("\n" + "=" * 50)
        print("TRAINING COIN RISK MODEL")
        print("=" * 50)
        train_coin_risk_model()

    if args.train_pairwise_model:
        print("\n" + "=" * 50)
        print("TRAINING PAIRWISE RISK MODEL")
        print("=" * 50)
        pair_wise_risk_model()
    
    if args.train_risk_weight_model:   
        print("\n" + "=" * 50)
        print("TRAINING RISK WEIGHT MODEL")
        print("=" * 50)
        risk_weight_model()
    
    # Analyze wallet if requested
    if args.wallet:
        print("\n" + "=" * 50)
        print("WALLET RISK ANALYSIS")
        print("=" * 50)
        print(f"Analyzing wallet: {args.wallet}")
        risk_analysis = analyze_wallet_risk_safe(args.wallet, config)
        if risk_analysis:
            print("Risk Analysis Result:")
            print(json.dumps(risk_analysis, indent=2))
    
    # Use mock wallet data if requested
    if args.wallet_mock or args.wallet == "mock":
        print("\n" + "=" * 50)
        print("MOCK WALLET ANALYSIS")
        print("=" * 50)
        try:
            with open("data/wallet_mock_summary.json", "r", encoding="utf-8") as f:
                wallet_summary = json.load(f)
            risk_analysis = WalletRiskCalculator().calculate_wallet_risk_score(wallet_summary)
            print("Mock Wallet Risk Analysis:")
            print(json.dumps(risk_analysis, indent=2))
        except FileNotFoundError:
            print("Mock wallet data file not found: data/wallet_mock_summary.json")
        except Exception as e:
            print(f"Error analyzing mock wallet: {e}")
    
    # If no specific actions requested, show help
    if not any([args.update_all, args.update_uniswap, args.update_aave, args.update_coins, args.update_news, 
                args.train_model, args.train_pairwise_model, args.train_risk_weight_model, args.wallet, args.wallet_mock]):
        print("No actions specified. Use --help to see available options.")
        print("\nQuick start examples:")
        print("  python test.py --update-all          # Update all data")
        print("  python test.py --train-model         # Train risk model")
        print("  python test.py --wallet <ADDRESS>    # Analyze wallet")

if __name__ == "__main__":
    main()