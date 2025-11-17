import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import CodeInterpreterTool,FunctionTool, ToolSet
from typing import Callable, Set, Any
import json
from dotenv import load_dotenv
import asyncio
from pathlib import Path

env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
else:
    # Try loading from current directory as fallback
    load_dotenv(override=True)

from pathlib import Path
from agent_processor import create_function_tool_for_agent

# Load the prompt instructions for the interior design agent from a file
# path = r'prompts\InteriorDesignAgentPrompt.txt'
ID_PROMPT_TARGET = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'prompts', 'InteriorDesignAgentPrompt.txt')
with open(ID_PROMPT_TARGET, 'r', encoding='utf-8') as file:
    ID_PROMPT = file.read()

project_endpoint = os.environ["AZURE_AI_AGENT_ENDPOINT"]

project_client = AIProjectClient(
    endpoint=project_endpoint,
    credential=DefaultAzureCredential(),
)


# Define the set of user-defined callable functions to use as tools (from MCP client)
functions = create_function_tool_for_agent("interior_designer")
toolset = ToolSet()
toolset.add(functions)
project_client.agents.enable_auto_function_calls(tools=functions)



 # Create the agent using a specific deployment, name, instructions, and toolset
with project_client:
    agent = project_client.agents.create_agent(
        model=os.environ["AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME"],  # Model deployment name
        name="Zava Interior Design Agent",  # Name of the agent
        instructions=ID_PROMPT,  # Instructions for the agent
        toolset=toolset)
    print(f"Created agent, ID: {agent.id}")
