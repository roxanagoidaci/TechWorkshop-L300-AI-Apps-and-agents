from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from collections import deque
from typing import Deque, Tuple, Optional, Dict
import mcp
import orjson  # Faster JSON library
from openai import AzureOpenAI
from opentelemetry import trace
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry.trace import SpanKind
from azure.ai.agents.telemetry import trace_function

from azure.core.credentials import AzureKeyCredential
import asyncio
import datetime
import time
from azure.ai.agents.telemetry import trace_function

from utils.history_utils import format_chat_history, redact_bad_prompts_in_history, clean_conversation_history
from utils.response_utils import extract_bot_reply, parse_agent_response, merge_cart_and_cora
import logging
import aiohttp
from concurrent.futures import ThreadPoolExecutor

# Import modularized utilities and services
from utils.env_utils import load_env_vars, validate_env_vars
from utils.message_utils import (
    IMAGE_UPLOAD_MESSAGES, IMAGE_CREATE_MESSAGES, IMAGE_ANALYSIS_MESSAGES,
    VIDEO_UPLOAD_MESSAGES, VIDEO_ANALYSIS_MESSAGES,
    get_rotating_message
)

scenario = os.path.basename(__file__)
tracer = trace.get_tracer(__name__)

env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
else:
    # Try loading from current directory as fallback
    load_dotenv(override=True)

#from app.agents.singleAgentExample import generate_response
from app.agents.mcpToolAgentExample import generate_response_using_tools
from app.servers.mcp_inventory_server import mcp as inventory_mcp
# Configure structured logging
logging.basicConfig(
    level=logging.INFO if os.getenv('DEBUG') else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global thread pool executor for CPU-bound operations
thread_pool = ThreadPoolExecutor(max_workers=4)


scenario = os.path.basename(__file__)


# Timing utility function with structured logging
def log_timing(operation_name: str, start_time: float, additional_info: str = ""):
    """Log timing information for operations using structured logging."""
    elapsed_time = time.time() - start_time
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_message = f"[TIMING] {timestamp} - {operation_name}: {elapsed_time:.3f}s"
    if additional_info:
        log_message += f" | {additional_info}"
    logger.info(log_message)
    return elapsed_time

async def get_cached_image_description(image_url: str, image_cache: dict) -> str:
    """Get image description with caching. If not in cache, fetch and store it."""
    if image_url in image_cache:
        logger.debug("Using cached image description", extra={"url": image_url[:50], "cache_size": len(image_cache)})
        return image_cache[image_url]
    
    logger.debug("Fetching new image description", extra={"url": image_url[:50]})
    try:
        # Use thread pool executor for CPU-bound operations
        loop = asyncio.get_event_loop()
        description = await loop.run_in_executor(thread_pool, get_image_description, image_url)
        image_cache[image_url] = description
        logger.debug("Cached image description", extra={"url": image_url[:50]})
        return description
    except Exception as e:
        logger.error("Failed to get image description", extra={"url": image_url[:50], "error": str(e)})
        return ""

async def pre_fetch_image_description(image_url: str, image_cache: dict):
    """Pre-fetch image description asynchronously without blocking."""
    if image_url and image_url not in image_cache:
        logger.debug("Pre-fetching image description", extra={"url": image_url[:50]})
        try:
            loop = asyncio.get_event_loop()
            description = await loop.run_in_executor(thread_pool, get_image_description, image_url)
            image_cache[image_url] = description
            logger.debug("Pre-fetched and cached image description", extra={"url": image_url[:50]})
        except Exception as e:
            logger.error("Failed to pre-fetch image description", extra={"url": image_url[:50], "error": str(e)})

def log_cache_status(image_cache: dict, current_url: str = ""):
    """Log the current status of the image cache using structured logging."""
    cache_size = len(image_cache)
    cache_keys = list(image_cache.keys())
    logger.debug("Image cache status", extra={
        "cache_size": cache_size,
        "cache_keys": [url[:30] + '...' for url in cache_keys],
        "current_url_in_cache": current_url in image_cache if current_url else None
    })

def extract_product_names_from_response(response_data) -> str:
    """Extract product names from response data and format them."""
    try:
        # Handle string response data
        if isinstance(response_data, str):
            try:
                response_data = orjson.loads(response_data)
            except (orjson.JSONDecodeError, TypeError):
                return ""
        
        # Handle dictionary response
        if isinstance(response_data, dict):
            products = response_data.get("products")
            if products:
                # Handle products as string (JSON)
                if isinstance(products, str):
                    try:
                        products_list = orjson.loads(products)
                    except (orjson.JSONDecodeError, TypeError):
                        return ""
                # Handle products as list
                elif isinstance(products, list):
                    products_list = products
                else:
                    return ""
                
                # Extract names from products
                if products_list and isinstance(products_list, list):
                    names = []
                    for product in products_list:
                        if isinstance(product, dict) and "name" in product:
                            names.append(product["name"])
                    if names:
                        return f" [Products Mentioned: {', '.join(names)}]"
        
        return ""
    except Exception:
        return ""

def format_chat_history(chat_history: Deque[Tuple[str, str]]) -> str:
    """Format chat history for the handoff prompt."""
    return "\n".join([
        f"user: {msg}" if role == "user" else f"bot: {msg}"
        for role, msg in chat_history
    ])

# Optimized JSON serialization function
def fast_json_dumps(obj, **kwargs):
    """Use orjson for faster JSON serialization."""
    return orjson.dumps(obj, **kwargs).decode('utf-8')

# Optimized string template for user message formatting
def format_user_message_with_products(image_url: str, image_data: str, video_summary: str, 
                                   formatted_history: str, products) -> str:
    """Optimized string formatting for user messages with products."""
    parts = [
        f'"image_url": "{image_url or ""}",',
        f'"image_description": "{image_data or ""}",',
        f'"video_description": "{video_summary or ""}",',
        f'"conversation_history": "{formatted_history}",',
        f'"products_available": {fast_json_dumps(products)}'
    ]
    return "{" + ", ".join(parts) + "}"

# Safe operation wrapper for better error handling
async def safe_operation(operation, fallback_value=None, operation_name="Unknown"):
    """Safely execute an operation with proper error handling."""
    try:
        return await operation()
    except (ValueError, TypeError) as e:
        logger.warning(f"{operation_name} failed: {e}")
        return fallback_value
    except Exception as e:
        logger.error(f"Unexpected error in {operation_name}: {e}", exc_info=True)
        return fallback_value


app = FastAPI()


inventory_mcp_app = inventory_mcp.sse_app()
app.mount("/mcp-inventory", inventory_mcp_app)

load_dotenv()
env_vars = load_env_vars()
validated_env_vars = validate_env_vars(env_vars)

project_endpoint = os.environ.get("AZURE_AI_AGENT_ENDPOINT")
if not project_endpoint:
    raise ValueError("AZURE_AI_AGENT_ENDPOINT environment variable is required")
project_client = AIProjectClient(
    endpoint=project_endpoint,
    credential=DefaultAzureCredential(),
)



llm_client = AzureOpenAI(
    azure_endpoint=validated_env_vars['AZURE_OPENAI_ENDPOINT'],
    api_key=validated_env_vars['AZURE_OPENAI_KEY'],
    api_version=validated_env_vars['AZURE_OPENAI_API_VERSION'],
)

@app.get("/")
async def get():
    chat_html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chat.html')
    with open(chat_html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/health")
async def health_check():
    """Health check endpoint for Azure Web App."""
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "environment_vars_configured": {
            "phi_4_endpoint": bool(validated_env_vars.get('phi_4_endpoint')),
            "phi_4_api_key": bool(validated_env_vars.get('phi_4_api_key')),
            "azure_openai_endpoint": bool(validated_env_vars.get('AZURE_OPENAI_ENDPOINT')),
            "azure_openai_key": bool(validated_env_vars.get('AZURE_OPENAI_KEY')),
            "azure_ai_agent_endpoint": bool(os.environ.get("AZURE_AI_AGENT_ENDPOINT"))
        }
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    session_start_time = time.time()
    logger.info("WebSocket Session Started")
    
    await websocket.accept()
    thread = project_client.agents.threads.create()
    chat_history: Deque[Tuple[str, str]] = deque(maxlen=5)
 

    
    # Session-level variable to track persistent image URL
    persistent_image_url = ""

    # Session-level variable to track persistent cart state
    persistent_cart = []

    # Dictionary to cache image URLs and their descriptions
    image_cache = {}

    # Track bad prompts (those that triggered content filter)
    bad_prompts = set()

    # Use deque with maxlen for raw_io_history to prevent unbounded growth
    raw_io_history = deque(maxlen=100)

    try:
        while True:
            message_start_time = time.time()
            try:
                data = await websocket.receive_text()
                parsed = orjson.loads(data)  # Use orjson for faster parsing
                user_message = parsed.get("message", "")
                has_image = parsed.get("has_image", False)
                image_url = parsed.get("image_url", "")
                conversation_history = parsed.get("conversation_history", "")
                has_video = parsed.get("has_video", False)
                video_url = parsed.get("video_url", "")
                cart = parsed.get("cart", [])
                
                # # Update persistent image URL if a new one is provided
                if image_url:
                    persistent_image_url = image_url
                    logger.debug("Persistent image URL updated", extra={"url": persistent_image_url})
                    log_cache_status(image_cache, image_url)
                    # Pre-fetch the image description asynchronously
                    asyncio.create_task(pre_fetch_image_description(image_url, image_cache))
                
                # Append user message to raw_io_history
                raw_io_history.append({"input": user_message, "cart": persistent_cart})
                log_timing("Message Parsing", message_start_time, f"Message length: {len(user_message)} chars")
            except WebSocketDisconnect:
                logger.info("WebSocket connection terminated - client disconnected from endpoint")
                break
            except Exception as e:
                logger.error("Error parsing message", exc_info=True)
                user_message = data if 'data' in locals() else ''
                image_data = None
                has_image = False
                image_url = None
                has_video = False
                video_url = None
                conversation_history = ""
            
            # Parse conversation history from string format
            history_start_time = time.time()
            try:
                if conversation_history:
                    # Clear existing chat history
                    chat_history.clear()
                    # Parse the string format: "user: message\nbot: message"
                    lines = conversation_history.strip().split('\n')
                    for i, line in enumerate(lines):
                        if line.startswith('user: '):
                            user_msg = line[6:]  # Remove "user: " prefix
                            chat_history.append(("user", user_msg))
                        elif line.startswith('bot: '):
                            bot_msg = line[5:]   # Remove "bot: " prefix
                            # Clean bot messages to remove large JSON data
                            try:
                                parsed_bot = orjson.loads(bot_msg)  # Use orjson
                                # Handle list format (new agent response format)
                                if isinstance(parsed_bot, list) and len(parsed_bot) > 0:
                                    first_item = parsed_bot[0]
                                    if isinstance(first_item, dict) and "answer" in first_item:
                                        bot_msg = first_item["answer"]
                                # Handle dict format (old format)
                                elif isinstance(parsed_bot, dict) and "answer" in parsed_bot:
                                    bot_msg = parsed_bot["answer"]
                            except (orjson.JSONDecodeError, TypeError):
                                pass
                            chat_history.append(("bot", bot_msg))
                    # Add the current user message to the history
                    chat_history.append(("user", user_message))
                else:
                    chat_history.append(("user", user_message))
                log_timing("History Parsing", history_start_time, f"History entries: {len(chat_history)}")
            except Exception as e:
                logger.error("Error parsing conversation history", exc_info=True)
                chat_history.append(("user", user_message))
            
            #await websocket.send_text(fast_json_dumps({"answer": "This application is not yet ready to serve results. Please check back later.", "agent": None, "cart": persistent_cart}))

            # # Single-agent example
            # print("Generating single-agent response")
            # try:
            #     response = generate_response(user_message)
            #     await websocket.send_text(fast_json_dumps({"answer": response, "agent": "single", "cart": persistent_cart}))
            # except Exception as e:
            #     logger.error("Error during single-agent response generation", exc_info=True)
            #     await websocket.send_text(fast_json_dumps({"answer": "Error during single-agent response generation", "error": str(e), "cart": persistent_cart}))

        
            print("Generating MCP Tools agent response")
            try:
                response = await generate_response_using_tools(user_message)
                await websocket.send_text(fast_json_dumps({"answer": response, "agent": "tools", "cart": persistent_cart}))
            except Exception as e:
                logger.error("Error during mcp-tools-agent response generation", exc_info=True)
                await websocket.send_text(fast_json_dumps({"answer": "Error during mcp-tools-agent response generation", "error": str(e), "cart": persistent_cart}))

    except Exception as e:
        logger.error("WebSocket session error", exc_info=True)
        try:
            await websocket.send_text(fast_json_dumps({"answer": "Internal server error", "error": str(e), "cart": persistent_cart}))
        except Exception:
            pass
    finally:
        session_duration = time.time() - session_start_time
        logger.info(f"WebSocket Session Ended - Duration: {session_duration:.3f}s")

if __name__ == "__main__":
    import datetime
    import atexit
    
    # Register cleanup function
    def cleanup():
        """Cleanup function to close thread pool on shutdown."""
        logger.info("Shutting down thread pool executor")
        thread_pool.shutdown(wait=True)
    
    atexit.register(cleanup)
    
    now = datetime.datetime.now()
    # Format date as '19th June 4.51PM'
    day = now.day
    suffix = 'th' if 11 <= day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
    formatted_date = now.strftime(f"%d{suffix} %B %I.%M%p")
    connection_message = f"Connection Established - Zava Chat App - {formatted_date}"
    with tracer.start_as_current_span(connection_message):
        import uvicorn
        port = int(os.environ.get("PORT", 8000))
        uvicorn.run("chat_app:app", host="0.0.0.0", port=port)

        
