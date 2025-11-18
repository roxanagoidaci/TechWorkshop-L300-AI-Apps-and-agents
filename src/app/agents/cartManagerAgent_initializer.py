"""
Cart Manager Agent Initializer

This agent handles all cart-related operations including:
- Adding/removing items from cart
- Updating cart state
- Providing cart recommendations
- Merging cart updates with conversational responses
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import FunctionTool, ToolSet
from typing import Callable, Set, Any
import json
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
else:
    load_dotenv(override=True)

from agent_processor import create_function_tool_for_agent

# Load the prompt instructions for the cart manager agent
CART_PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
    'prompts', 
    'CartManagerPrompt.txt'
)

# Fallback prompt if file doesn't exist
CART_MANAGER_PROMPT = """You are a Cart Manager Assistant for Contoso, a home improvement retailer.

Your responsibilities:
1. Help customers manage their shopping cart
2. Add items to the cart based on customer requests
3. Remove items from the cart when requested
4. Update quantities of items in the cart
5. Provide cart summaries and totals
6. Suggest related products based on cart contents

When processing cart operations:
- Always confirm what action was taken (added, removed, updated)
- Provide the updated cart state
- Be friendly and helpful
- Suggest complementary products when appropriate

Cart operations should be based on the conversation history and raw I/O history provided.

Format your responses as JSON with the following structure:
{
    "answer": "Your conversational response to the customer",
    "cart": [
        {
            "product_id": "PROD-001",
            "name": "Product Name",
            "quantity": 2,
            "price": 29.99
        }
    ],
    "products": "Optional product recommendations",
    "discount_percentage": ""
}
"""

# Try to load from file, use fallback if not found
try:
    with open(CART_PROMPT_PATH, 'r', encoding='utf-8') as file:
        CART_MANAGER_PROMPT = file.read()
except FileNotFoundError:
    print(f"Cart prompt file not found at {CART_PROMPT_PATH}, using default prompt")

project_endpoint = os.environ["AZURE_AI_AGENT_ENDPOINT"]

project_client = AIProjectClient(
    endpoint=project_endpoint,
    credential=DefaultAzureCredential(),
)

# Create function tools for cart manager
# For now, cart manager uses conversational AI without specific tools
# Tools can be added later if needed (e.g., inventory check, price lookup)
functions = create_function_tool_for_agent("cart_manager")
toolset = ToolSet()
toolset.add(functions)
project_client.agents.enable_auto_function_calls(tools=functions)

# Create the cart manager agent
with project_client:
    agent = project_client.agents.create_agent(
        model=os.environ["AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME"],
        name="Contoso Cart Manager Agent",
        instructions=CART_MANAGER_PROMPT,
        toolset=toolset,
    )
    
    print(f"Created cart manager agent with ID: {agent.id}")
    print(f"Model: {agent.model}")
    print(f"Instructions length: {len(agent.instructions)} characters")
