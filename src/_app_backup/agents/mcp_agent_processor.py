"""
MCP-Enabled Agent Processor
Uses Model Context Protocol for tool execution instead of direct function calls
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from typing import List, Dict, Any, Callable, Set
from azure.ai.agents.models import (
    MessageImageUrlParam,
    MessageInputTextBlock,
    MessageInputImageUrlBlock,FunctionTool, ToolSet
)

from opentelemetry import trace
from azure.monitor.opentelemetry import configure_azure_monitor
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import json
from pathlib import Path

# Import tool functions
from app.tools.mcp_tools import call_mcp_tool

from src.app.servers.mcp_inventory_client import get_mcp_client

# Enable Azure Monitor tracing
application_insights_connection_string = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
if application_insights_connection_string:
    configure_azure_monitor(connection_string=application_insights_connection_string)

# Increase thread pool size for better concurrency
_executor = ThreadPoolExecutor(max_workers=8)

# Cache for toolsets to avoid recreating them
_toolset_cache: Dict[str, ToolSet] = {}

class MCPAgentProcessor:
    """Agent processor that uses MCP for tool execution."""
    
    def __init__(self, project_client, assistant_id, agent_type: str, thread_id=None, use_mcp: bool = True):
        self.project_client = project_client
        self.agent_id = assistant_id
        self.agent_type = agent_type
        self.thread_id = thread_id
        self.use_mcp = use_mcp
        self.mcp_client = None
        
        # Use cached toolset or create new one
        self.toolset, self.functions = self._get_or_create_toolset(agent_type)
        
        # Enable auto function calls with the FunctionTool, not ToolSet
        self.project_client.agents.enable_auto_function_calls(tools=self.functions)

        print(f"[MCP] Initializing {agent_type} agent with MCP: {use_mcp}")
        
    async def _get_mcp_client(self):
        """Get or initialize MCP client."""
        if self.mcp_client is None and self.use_mcp:
            self.mcp_client = await get_mcp_client()
        return self.mcp_client

    async def _execute_tool_via_mcp(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool via MCP."""
        client = await self._get_mcp_client()
        if client is None:
            raise RuntimeError("MCP client not initialized")
        
        print(f"[MCP] Executing tool: {tool_name} with args: {arguments}")
        result = await client.call_tool(tool_name, arguments)
        print(f"[MCP] Tool {tool_name} returned: {str(result)[:100]}...")
        return result

    def _parse_tool_calls_from_message(self, message) -> List[Dict[str, Any]]:
        """Parse tool calls from agent message."""
        tool_calls = []
        
        if hasattr(message, 'content') and isinstance(message.content, list):
            for block in message.content:
                # Check for function/tool call blocks
                if hasattr(block, 'type') and 'tool' in block.type.lower():
                    if hasattr(block, 'function'):
                        func = block.function
                        tool_calls.append({
                            'name': func.name if hasattr(func, 'name') else '',
                            'arguments': json.loads(func.arguments) if hasattr(func, 'arguments') else {}
                        })
        
        return tool_calls

    def run_conversation_with_image(self, input_message: str = "", image_path: str = ""):
        """Run conversation with image input."""
        start_time = time.time()
        span = trace.get_current_span()
        span.set_attribute("message_from_user", input_message)
        span.set_attribute("image_from_user", image_path)
        span.set_attribute("uses_mcp", self.use_mcp)
        
        thread_id = self.thread_id
        url_param = MessageImageUrlParam(url=image_path, detail="high")
        content_blocks = [
            MessageInputTextBlock(text=input_message),
            MessageInputImageUrlBlock(image_url=url_param),
        ]
        
        self.project_client.agents.messages.create(
            thread_id=thread_id,
            role="user",
            content=content_blocks
        )
        print(f"[TIMELOG] Message creation took: {time.time() - start_time:.2f}s")
        
        run_start = time.time()
        self.project_client.agents.runs.create_and_process(
            thread_id=thread_id, 
            agent_id=self.agent_id, 
            tool_choice="auto"
        )
        print(f"[TIMELOG] Thread run took: {time.time() - run_start:.2f}s")
        
        self.project_client.agents.messages.list(thread_id=thread_id)
        print(f"[TIMELOG] Total run_conversation_with_image time: {time.time() - start_time:.2f}s")

    def run_conversation_with_text(self, input_message: str = ""):
        """Run conversation with text input."""
        start_time = time.time()
        thread_id = self.thread_id
        
        self.project_client.agents.messages.create(
            thread_id=thread_id,
            role="user",
            content=input_message,
        )
        print(f"[TIMELOG] Message creation took: {time.time() - start_time:.2f}s")
        
        run_start = time.time()
        self.project_client.agents.runs.create_and_process(
            thread_id=thread_id, 
            agent_id=self.agent_id, 
            tool_choice="auto"
        )
        print(f"[TIMELOG] Thread run took: {time.time() - run_start:.2f}s")
        
        messages = self.project_client.agents.messages.list(thread_id=thread_id)
        for message in messages:
            yield message.content
        print(f"[TIMELOG] Total run_conversation_with_text time: {time.time() - start_time:.2f}s")

    def _run_conversation_sync(self, input_message: str = ""):
        """Synchronous conversation runner with MCP integration."""
        thread_id = self.thread_id
        start_time = time.time()
        
        try:
            # Create message
            self.project_client.agents.messages.create(
                thread_id=thread_id,
                role="user",
                content=input_message,
            )
            print(f"[TIMELOG] Message creation took: {time.time() - start_time:.2f}s")
            
            # Run agent
            run_start = time.time()
            
            # Note: If using MCP, the agent would need to be configured to call MCP tools
            # For now, we'll handle tool execution interception if needed
            self.project_client.agents.runs.create_and_process(
                thread_id=thread_id, 
                agent_id=self.agent_id, 
                tool_choice="auto"
            )
            
            print(f"[TIMELOG] Thread run took: {time.time() - run_start:.2f}s")

            # Retrieve messages
            messages_start = time.time()
            messages = list(self.project_client.agents.messages.list(thread_id=thread_id, limit=1))
            print(f"[TIMELOG] Message retrieval took: {time.time() - messages_start:.2f}s")
            
            # Extract assistant message
            assistant_msg = next((m for m in messages if m.role == "assistant"), None)
            
            if assistant_msg:
                content = assistant_msg.content
                if isinstance(content, list):
                    text_blocks = []
                    for block in content:
                        if isinstance(block, dict):
                            text_val = block.get('text', {}).get('value')
                            if text_val:
                                text_blocks.append(text_val)
                        elif hasattr(block, 'text'):
                            if hasattr(block.text, 'value'):
                                text_val = block.text.value
                                if text_val:
                                    text_blocks.append(text_val)
                    if text_blocks:
                        return ['\n'.join(text_blocks)]
                
                return [str(content)]
            else:
                return [""]
                
        except Exception as e:
            print(f"[ERROR] Conversation failed: {str(e)}")
            return [f"Error processing message: {str(e)}"]

    def _get_or_create_toolset(self, agent_type: str):
        """Get cached toolset or create new one to avoid repeated initialization.
        Returns: tuple of (ToolSet, FunctionTool)
        """
        if agent_type in _toolset_cache:
            return _toolset_cache[agent_type]
        
        # Add MCP tools to the default toolset
        user_functions: Set[Callable[..., Any]] = {
            call_mcp_tool,
        }

        # Initialize agent toolset with MCP tools
        functions = FunctionTool(user_functions)

        # # Create new toolset based on agent type
        # if agent_type == "interior_designer":
        #     interior_functions: Set[Callable[..., Any]] = {create_image, product_recommendations}
        #     functions = FunctionTool(interior_functions)
        # elif agent_type == "customer_loyalty":
        #     loyalty_functions: Set[Callable[..., Any]] = {calculate_discount}
        #     functions = FunctionTool(loyalty_functions)
        # elif agent_type == "inventory_agent":
        #     inventory_functions: Set[Callable[..., Any]] = {inventory_check}
        #     functions = FunctionTool(inventory_functions)
        # else:
        #     default_functions: Set[Callable[..., Any]] = set()
        #     functions = FunctionTool(default_functions)
        
        # Adding attributes to the current span
        span = trace.get_current_span()
        span.set_attribute("selected_agent", agent_type)

        toolset = ToolSet()
        toolset.add(functions)
        
        # Cache both toolset and functions as a tuple
        result = (toolset, functions)
        _toolset_cache[agent_type] = result
        return result

    def get_toolset(self, agent_type: str):
        """Deprecated: Use _get_or_create_toolset instead."""
        return self._get_or_create_toolset(agent_type)



    async def run_conversation_with_text_stream(self, input_message: str = ""):
        """Async wrapper for conversation processing."""
        print("[DEBUG] MCP-enabled async conversation pipeline initiated", flush=True)
        loop = asyncio.get_event_loop()
        try:
            messages = await loop.run_in_executor(
                _executor, self._run_conversation_sync, input_message
            )
            for msg in messages:
                yield msg
        except Exception as e:
            print(f"[ERROR] Async conversation failed: {str(e)}")
            yield f"Error processing message: {str(e)}"
    
    async def cleanup(self):
        """Cleanup MCP resources."""
        if self.mcp_client:
            await self.mcp_client.cleanup()
            self.mcp_client = None
