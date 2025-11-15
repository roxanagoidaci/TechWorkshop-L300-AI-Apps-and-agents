"""
MCP Tool Wrapper - Calls MCP server tools from Azure AI Agent
"""
import asyncio
from typing import Dict, Any
import sys
import os

# Add mcp directory to path
mcp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'mcp')
sys.path.insert(0, mcp_path)

from src.app.mcp_tools.mcp_tools_client import MCPShopperToolsClient

def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """
    Generic function to call any MCP server tool.
    
    Args:
        tool_name: Name of the tool to call
        arguments: Dictionary of arguments for the tool
        
    Returns:
        The result from the MCP server
    """
    client = MCPShopperToolsClient()
    result = asyncio.run(client.call_tool(tool_name, arguments))
    
    # Extract text content if available
    if hasattr(result, 'content') and len(result.content) > 0:
        return result.content[0].text
    return str(result)
