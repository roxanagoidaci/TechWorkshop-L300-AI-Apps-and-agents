import os
import sys
import json
from dotenv import load_dotenv
import asyncio
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import FunctionTool, ToolSet
from azure.identity import DefaultAzureCredential
from typing import Callable, Set, Any

env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
else:
    # Try loading from current directory as fallback
    load_dotenv(override=True)

# Add src directory to Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))
# Import MCP client
from app.servers.mcp_inventory_client import get_mcp_client

_mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8050/sse")

IA_PROMPT_TARGET = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'prompts', 'InventoryAgentPrompt.txt')
with open(IA_PROMPT_TARGET, 'r', encoding='utf-8') as file:
    IA_PROMPT = file.read()

project_endpoint = os.environ["AZURE_AI_AGENT_ENDPOINT"]

project_client = AIProjectClient(
    endpoint=project_endpoint,
    credential=DefaultAzureCredential(),
)

# Create wrapper function that uses MCP client
def inventory_check(product_dict: dict) -> list:
    """
    Check inventory for products using MCP client.
    
    Args:
        product_dict (dict): Keys are product names, values are product IDs.
    
    Returns:
        list: Each element is the inventory info for the product ID if found, otherwise None.
    """
    async def _check_inventory():
        mcp_client = await get_mcp_client(_mcp_server_url)
        results = []
        for product_name, product_id in product_dict.items():
            try:
                inventory_data = await mcp_client.check_inventory(product_id)
                results.append(inventory_data)
            except Exception as e:
                print(f"Error checking inventory for {product_id}: {e}")
                results.append(None)
        return results
    
    # Run async function in event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(_check_inventory())


user_functions: Set[Callable[..., Any]] = {
    inventory_check,
}

# Initialize agent toolset with user functions
functions = FunctionTool(user_functions)
toolset = ToolSet()
toolset.add(functions)
project_client.agents.enable_auto_function_calls(tools=functions)

with project_client:
    # Create an agent with the Bing Grounding tool
    agent = project_client.agents.create_agent(
        model=os.getenv("AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME"),  # Model deployment name
        name="Zava Inventory Agent",  # Name of the agent
        instructions=IA_PROMPT,  # Instructions for the agent
        toolset=toolset
    )
    print(f"Created agent, ID: {agent.id}")
