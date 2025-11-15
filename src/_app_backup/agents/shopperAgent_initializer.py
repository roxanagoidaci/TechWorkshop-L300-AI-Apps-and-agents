import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import CodeInterpreterTool,FunctionTool, ToolSet
from typing import Callable, Set, Any
import json
from tools.mcp_tools import call_mcp_tool
from dotenv import load_dotenv
load_dotenv()

CORA_PROMPT_TARGET = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'prompts', 'ShopperAgentPrompt.txt')
with open(CORA_PROMPT_TARGET, 'r', encoding='utf-8') as file:
    CORA_PROMPT = file.read()

project_endpoint = os.environ["AZURE_AI_AGENT_ENDPOINT"]

project_client = AIProjectClient(
    endpoint=project_endpoint,
    credential=DefaultAzureCredential(),
)

# Add MCP tools to Cora's toolset
user_functions: Set[Callable[..., Any]] = {
    call_mcp_tool,
}

# Initialize agent toolset with MCP tools
functions = FunctionTool(user_functions)
toolset = ToolSet()
toolset.add(functions)
project_client.agents.enable_auto_function_calls(tools=functions)


with project_client:
    agent = project_client.agents.create_agent(
        model=os.environ["AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME"],  # Model deployment name
        name="Cora",  # Name of the agent
        instructions=CORA_PROMPT,  # Instructions for the agent
        toolset=toolset  # Add the MCP tools
    )
    print(f"Created agent, ID: {agent.id}")
