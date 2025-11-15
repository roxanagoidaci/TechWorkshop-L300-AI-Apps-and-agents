from app.agents.mcp_agent_processor import MCPAgentProcessor
from typing import Dict

_agent_processor_cache: Dict[str, MCPAgentProcessor] = {}

def get_or_create_agent_processor(agent_id: str, agent_type: str, thread_id: str, project_client) -> MCPAgentProcessor:
    """Get cached AgentProcessor or create new one to avoid repeated initialization."""
    cache_key = f"{agent_type}_{agent_id}"
    if cache_key in _agent_processor_cache:
        processor = _agent_processor_cache[cache_key]
        processor.thread_id = thread_id
        # Re-enable auto function calls with the functions
        if processor.functions:
            project_client.agents.enable_auto_function_calls(tools=processor.functions)
        return processor
    processor = MCPAgentProcessor(
        project_client=project_client,  # project_client is now passed as an argument
        assistant_id=agent_id,
        agent_type=agent_type,
        thread_id=thread_id
    )
    _agent_processor_cache[cache_key] = processor
    return processor 