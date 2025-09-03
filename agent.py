import os
from typing import Dict, Any, Optional
import yaml
import asyncio
import json
import openai
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from dotenv import load_dotenv
load_dotenv()

class RiskAnalysisAgent:
    def __init__(self, config_path: str = "config_local.yml", use_mock_mcp: bool = True):
        """Initialize the risk analysis agent"""
        self.config = self._load_config(config_path)
        
        # Initialize OpenAI client only if API key is available
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            try:
                # For openai version >= 1.0.0, use OpenAI client
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
            except (ValueError, KeyError, TypeError, OSError, IOError, AttributeError) as e:
                print(f"Warning: Failed to initialize OpenAI client: {e}")
                self.openai_client = None
        else:
            print("Warning: No OpenAI API key found in config. AI recommendations will be disabled.")
            self.openai_client = None
            
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.stdio = None
        self.write = None
    
    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        try:
            is_python = server_script_path.endswith('.py')
            if not (is_python):
                raise ValueError("Server script must be a .py file")
                
            command = "python"
            server_params = StdioServerParameters(
                command=command,
                args=[server_script_path],
                env=None
            )
            
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
            
            await self.session.initialize()
            
            # List available tools
            response = await self.session.list_tools()
            tools = response.tools
            print("\nConnected to server with tools:", [tool.name for tool in tools])
            
        except (ValueError, KeyError, TypeError, OSError, IOError, AttributeError) as e:
            print(f"Failed to connect to MCP server: {e}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        # Try to load local config first, then fall back to default
        local_config_path = config_path.replace('.yml', '_local.yml')
        
        if os.path.exists(local_config_path):
            print(f"Loading configuration from {local_config_path}")
            with open(local_config_path, 'r') as file:
                return yaml.safe_load(file)
        else:
            print(f"{local_config_path} not found, using {config_path}")
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
    
    async def update_risk_info(self) -> Dict[str, Any]:
        """Update risk information using MCP server"""
        try:
            print("Updating risk information via MCP...")
            if hasattr(self, 'mock_client'):
                result = await self.mock_client.call_tool("update_risk_info")
            else:
                result = await self.session.call_tool("update_risk_info", {})
            return result
            
        except (ValueError, KeyError, TypeError, OSError, IOError, AttributeError) as e:
            return {"error": f"Failed to update risk info: {str(e)}"}
    
    async def get_risk_summary(self, wallet_address: str = None) -> Dict[str, Any]:
        """Get risk summary using MCP server"""
        try:
            print("Getting risk summary via MCP...")
            
            if wallet_address:
                result = await self.session.call_tool("get_risk_summary", {"wallet_address": wallet_address})
            else:
                result = await self.session.call_tool("get_risk_summary", {})
            
            return result.content[0].text
            
        except (ValueError, KeyError, TypeError, OSError, IOError, AttributeError) as e:
            return {"error": f"Failed to get risk summary: {str(e)}"}
    
    async def get_risk_recommendations(self, platform: str, wallet_address: str) -> str:
        """Get AI-powered recommendations for risk mitigation"""
        if self.openai_client is None:
            return "AI recommendations are disabled due to missing API key."
            
        try:
            
            risk_analysis = await self.get_risk_summary(wallet_address)
            
            print(risk_analysis)
            
            # Create a prompt for OpenAI
            prompt = f"""
            You are a senior DeFi Risk Analyst specializing in cryptocurrency wallet risk assessment. Your task is to analyze the provided risk analysis data and provide personalized, actionable recommendations.

            RISK ANALYSIS DATA:
            {risk_analysis}

            INSTRUCTIONS:
            1. Carefully examine the risk analysis data above
            2. Focus ONLY on the specific data provided - do not make assumptions
            3. Provide personalized analysis based on the actual numbers and metrics shown
            4. Give concrete, actionable advice that addresses the specific risks identified

            REQUIRED ANALYSIS FORMAT:

            ## OVERALL RISK ASSESSMENT
            - Current Risk Score: [extract from data]
            - Risk Level: [extract from data]
            - Summary: Brief overview of the wallet's current risk profile

            ## DETAILED RISK FACTOR ANALYSIS
            Analyze the top 3-5 risk factors from the data, for each factor:
            - **Risk Factor Name**: [extract specific factor from data]
            - **Current Score**: [extract exact number from data]
            - **Risk Level**: [High/Medium/Low based on the score]
            - **Why It's Risky**: Explain using the specific numbers from the data
            - **Personalized Impact**: How this specific risk affects this wallet

            ## IMMEDIATE ACTIONS (Next 24-48 hours)
            Provide 3-5 specific actions based on the actual risk data:
            - **Action**: [Specific step]
            - **Target**: [Exact number/percentage to achieve]
            - **Reason**: [Why this action addresses the specific risk]

            ## MEDIUM-TERM STRATEGIES (Next 1-4 weeks)
            Based on the risk patterns identified:
            - **Strategy**: [Specific approach]
            - **Implementation**: [How to execute]
            - **Expected Outcome**: [What risk reduction to expect]

            ## LONG-TERM RISK MANAGEMENT (Next 1-6 months)
            Structural improvements based on the wallet's risk profile:
            - **Framework Changes**: [Specific modifications]
            - **Monitoring Systems**: [What to track and how]
            - **Success Metrics**: [How to measure improvement]

            IMPORTANT: 
            - Use ONLY the data provided in the risk analysis
            - Reference specific numbers, scores, and metrics from the data
            - Provide personalized recommendations, not generic advice
            - Make all recommendations actionable and measurable
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a senior DeFi Risk Analyst with expertise in cryptocurrency wallet security and risk management. Your responses must be data-driven, personalized, and actionable. Always reference specific numbers and metrics from the provided data. Avoid generic advice - provide recommendations that directly address the specific risks identified in the data."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )

            return response.choices[0].message.content
            
        except (ValueError, KeyError, TypeError, OSError, IOError, AttributeError) as e:
            return f"Failed to get AI recommendations: {str(e)}"
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.session:
                # ClientSession doesn't have a close method, just let it be garbage collected
                pass
            if self.exit_stack:
                await self.exit_stack.aclose()
        except (ValueError, KeyError, TypeError, OSError, IOError, AttributeError) as e:
            print(f"Error during cleanup: {e}")
        finally:
            # Ensure we clean up any remaining resources
            if hasattr(self, 'stdio') and self.stdio:
                try:
                    self.stdio.close()
                except:
                    pass
            if hasattr(self, 'write') and self.write:
                try:
                    self.write.close()
                except:
                    pass


async def main():
    """Main function to run the risk analysis agent"""
    print("🚀 DeFi Risk Analysis Agent Starting...")
    
    # Initialize the agent
    agent = RiskAnalysisAgent(use_mock_mcp=True)  # Use mock MCP client for now
    
    try:
        await agent.connect_to_server("servers/mcpServer.py")
    except (ValueError, KeyError, TypeError, OSError, IOError, AttributeError) as e:
        print(f"Error connecting to server: {e}")
        exit(1)
    
    # Example usage
    wallet_address = input("Enter wallet address to analyze (or press Enter to skip): ").strip()
    
    if not wallet_address:
        print("No input provided. Running analysis on all available data...")
        wallet_address = None
     
    
    print("="*50)
    print("RISK RECOMMENDATIONS")
    print("="*50)
    risk_recommendations = await agent.get_risk_recommendations("wallet", wallet_address)
    
    print(risk_recommendations)
    
    # Cleanup
    await agent.cleanup()


if __name__ == "__main__":
    # Run the async main function

    asyncio.run(main())
