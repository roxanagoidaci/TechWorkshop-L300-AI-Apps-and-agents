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
import uuid

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



env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
else:
    # Try loading from current directory as fallback
    load_dotenv(override=True)

env_vars = load_env_vars()
validated_env_vars = validate_env_vars(env_vars)

#from app.agents.singleAgentExample import generate_response
from services.agent_service import get_or_create_agent_processor
from services.handoff_service import HandoffService
from services.fallback_service import call_fallback, cora_fallback
from app.agents.mcpToolAgentExample import generate_response_using_tools
from app.servers.mcp_inventory_server import mcp as inventory_mcp
from app.tools.aiSearchTools import product_recommendations
from app.tools.understandImage import get_image_description
from app.tools.imageCreationTool import create_image

# Configure structured logging
logging.basicConfig(
    level=logging.INFO if os.getenv('DEBUG') else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global thread pool executor for CPU-bound operations
thread_pool = ThreadPoolExecutor(max_workers=4)

application_insights_connection_string = os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"]
configure_azure_monitor(connection_string=application_insights_connection_string)
# OpenAIInstrumentor().instrument()

scenario = os.path.basename(__file__)
tracer = trace.get_tracer(__name__)

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
        f'"image_url": "{image_url or ""}"',
        f'"image_description": "{image_data or ""}"',
        f'"video_description": "{video_summary or ""}"',
        f'"conversation_history": "{formatted_history}"',
        f'"products_available": {fast_json_dumps(products)}'
    ]
    return "{" + ", ".join(parts) + "}"

async def get_video_summary(video_url: str) -> str:
    """Get video summary (placeholder for actual video analysis)."""
    # TODO: Implement actual video analysis
    # For now, return a placeholder indicating video was received
    return f"Video received from URL: {video_url}. Video content analysis pending."

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

@tracer.start_as_current_span("assess_claims_with_context")
def select_agent(handoff_reply: str, env_vars: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    """Select agent and agent_name based on handoff reply."""
    start_time = time.time()
    reply = handoff_reply.lower()
    if "cora" in reply:
        result = env_vars.get('cora'), "cora"
    elif "interior_designer_create_image" in reply:
        result = env_vars.get('interior_designer'), "interior_designer_create_image"
    elif "interior_designer" in reply:
        result = env_vars.get('interior_designer'), "interior_designer"
    elif "inventory_agent" in reply:
        result = env_vars.get('inventory_agent'), "inventory_agent"
    elif "customer_loyalty" in reply:
        result = env_vars.get('customer_loyalty'), "customer_loyalty"
    else:
        result = None, None
    
    log_timing("Agent Selection", start_time, f"Selected: {result[1] if result[1] else 'None'}")
    return result

def call_handoff(handoff_client: ChatCompletionsClient, handoff_prompt: str, formatted_history: str, phi_4_deployment: str) -> str:
    """Call the handoff model and return its reply. Handles content filter errors."""
    start_time = time.time()
    with tracer.start_as_current_span("custom_function") as span:
        span.set_attribute("custom_attribute", "value")    
        try:
            handoff_response = handoff_client.complete(
                messages=[
                    SystemMessage(content=handoff_prompt),
                    UserMessage(content=formatted_history),
                ],
                max_tokens=2048,
                temperature=0.8,
                top_p=0.1,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                model=phi_4_deployment
            )
            result = handoff_response.choices[0].message.content
            log_timing("Handoff Call", start_time, f"Model: {phi_4_deployment}")
            return result
        except Exception as e:
            # Check for content filter error
            err_str = str(e)
            if "content_filter" in err_str or "ResponsibleAIPolicyViolation" in err_str:
                # Return a special marker string so the caller can handle it 
                result = "__CONTENT_FILTER_ERROR__" + err_str
                log_timing("Handoff Call (Content Filter Error)", start_time, f"Error: {err_str[:50]}...")
                return result
            # Otherwise, re-raise
            log_timing("Handoff Call (Exception)", start_time, f"Exception: {str(e)[:50]}...")
            raise

def call_fallback(llm_client, fallback_prompt: str, gpt_deployment = "gpt-4.1"):
    """Call the fallback model and return its reply."""
    start_time = time.time()
    
    chat_prompt = [    
        {
            "role": "system",      
            "content": 
            [           
                {               
                    "type": "text",               
                    "text": fallback_prompt           
                }       
            ]   
        }]

    messages = chat_prompt
    completion = llm_client.chat.completions.create(
        model=gpt_deployment,
        messages=messages,
        temperature=0.7,
        stream=False)
    result = completion.choices[0].message.content
    log_timing("Fallback Call", start_time, f"Model: {gpt_deployment}")
    return result

def cora_fallback(llm_client, fallback_prompt: str, gpt_deployment = "Phi-4"):
    """Call the fallback model for cora and return its reply."""
    start_time = time.time()
    
    chat_prompt = [    
        {
            "role": "system",      
            "content": 
            [           
                {               
                    "type": "text",               
                    "text": fallback_prompt           
                }       
            ]   
        }]

    messages = chat_prompt
    completion = llm_client.chat.completions.create(
        model=gpt_deployment,
        messages=messages,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False)
    result = completion.choices[0].message.content
    log_timing("Cora Fallback Call", start_time, f"Model: {gpt_deployment}")
    return result

def cart_update(llm_client, cart_update_prompt: str):
    """Call the cart update model and return its reply."""
    start_time = time.time()
    
    chat_prompt = [    
        {
            "role": "system",      
            "content": 
            [           
                {               
                    "type": "text",               
                    "text": cart_update_prompt           
                }       
            ]   
        }]

    gpt_deployment = validated_env_vars['gpt_deployment']
    messages = chat_prompt
    completion = llm_client.chat.completions.create(
        model=gpt_deployment,
        messages=messages,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False)
    result = completion.choices[0].message.content
    log_timing("Cart Update Call", start_time, f"Model: {gpt_deployment}")
    return result

app = FastAPI()


#set up MCP inventory server as a mounted app
inventory_mcp_app = inventory_mcp.sse_app()
app.mount("/mcp-inventory/", inventory_mcp_app)


project_endpoint = os.environ.get("AZURE_AI_AGENT_ENDPOINT")
if not project_endpoint:
    raise ValueError("AZURE_AI_AGENT_ENDPOINT environment variable is required")
project_client = AIProjectClient(
    endpoint=project_endpoint,
    credential=DefaultAzureCredential(),
)



HANDOFF_PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompts', 'handoffPrompt.txt')
FALLBACK_PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompts', 'fallBackPrompt.txt')
CORA_FALLBACK_PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompts', 'CoraPrompt.txt')
CART_UPDATE_PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompts', 'addToCartPrompt.txt')

with open(HANDOFF_PROMPT_PATH, 'r') as file:
    HANDOFF_PROMPT = file.read()

with open(FALLBACK_PROMPT_PATH, 'r') as file:
    FALLBACK_PROMPT = file.read()

with open(CORA_FALLBACK_PROMPT_PATH, 'r') as file:
    CORA_FALLBACK_PROMPT = file.read()

with open(CART_UPDATE_PROMPT_PATH, 'r') as file:
    CART_UPDATE_PROMPT = file.read()

handoff_client = ChatCompletionsClient(
    endpoint=validated_env_vars['phi_4_endpoint'],
    credential=AzureKeyCredential(validated_env_vars['phi_4_api_key']),
    api_version=validated_env_vars['phi_4_api_version']
)

llm_client = AzureOpenAI(
    azure_endpoint=validated_env_vars['AZURE_OPENAI_ENDPOINT'],
    api_key=validated_env_vars['AZURE_OPENAI_KEY'],
    api_version=validated_env_vars['AZURE_OPENAI_API_VERSION'],
)

# Initialize HandoffService with structured intent classification
handoff_service = HandoffService(
    azure_openai_client=llm_client,
    deployment_name=validated_env_vars['gpt_deployment'],
    default_domain="cora",
    lazy_classification=True
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
            "azure_ai_agent_endpoint": bool(os.environ.get("AZURE_AI_AGENT_ENDPOINT")),
            "MCP_SERVER_URL": bool(os.environ.get("MCP_SERVER_URL")),
        }
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint implementing a Multi-Agent System with Intelligent Handoff.
    
    This endpoint orchestrates multiple specialized AI agents that handle different aspects
    of the shopping experience:
    
    MULTI-AGENT ARCHITECTURE:
    ========================
    1. Cora (General Shopping Assistant) - Product browsing, general questions
    2. Interior Designer - Design recommendations, color schemes, image creation
    3. Cart Manager - Shopping cart operations (add/remove items, checkout)
    4. Inventory Agent - Stock availability and inventory checks
    5. Customer Loyalty - Discount calculations and loyalty programs
    
    INTELLIGENT HANDOFF SYSTEM:
    ==========================
    The HandoffService uses LLM-based intent classification to intelligently route
    user queries to the most appropriate specialized agent:
    
    - Analyzes user message content and conversation context
    - Classifies intent into one of the agent domains
    - Provides confidence scores and reasoning for routing decisions
    - Maintains session-aware domain tracking to reduce unnecessary handoffs
    - Supports lazy classification to optimize performance
    
    UNIFIED AGENT EXECUTION PATTERN:
    ===============================
    All agents are executed through a consistent AgentProcessor pattern:
    
    1. Intent Classification: HandoffService determines the target agent
    2. Context Preparation: Enriches user message with relevant data (images, products, history)
    3. Agent Execution: AgentProcessor handles the conversation with streaming responses
    4. Response Processing: Parses agent output, updates session state, sends to user
    
    SESSION STATE MANAGEMENT:
    ========================
    - persistent_cart: Shopping cart maintained across the session
    - session_discount_percentage: Customer loyalty discount (calculated once)
    - persistent_image_url: Last uploaded image for context
    - chat_history: Recent conversation for context (5 messages)
    - raw_io_history: Full input/output history for cart management
    
    MULTIMODAL SUPPORT:
    ==================
    - Image upload and analysis (room photos, product images)
    - Video upload and summarization
    - Product recommendations based on visual content
    
    OBSERVABILITY:
    =============
    - OpenTelemetry distributed tracing for all agent calls
    - Performance timing logs for each operation
    - Structured logging for debugging and monitoring
    """
    session_start_time = time.time()
    session_id = str(uuid.uuid4())
    logger.info("WebSocket Session Started")
    
    await websocket.accept()
    
    # Create dedicated threads for agent conversations
    thread = project_client.agents.threads.create()  # Main conversation thread
    customer_loyalty_thread = project_client.agents.threads.create()  # Separate thread for loyalty calculations
    
    # Conversation history management (limited to 5 recent messages for context)
    chat_history: Deque[Tuple[str, str]] = deque(maxlen=5)
    customer_loyalty_thread = project_client.agents.threads.create()

    # =============================================================================
    # SESSION STATE VARIABLES
    # =============================================================================
    
    # Customer Loyalty Agent State
    # ---------------------------
    customer_loyalty_executed = False  # Ensures loyalty calculation runs only once per session
    session_discount_percentage = ""   # Stores customer's discount rate for the session
    session_loyalty_response = None    # Full loyalty agent response (sent after cart operations)
    loyalty_response_sent = False      # Prevents duplicate loyalty responses
    
    # Multimodal Content State
    # -----------------------
    persistent_image_url = ""  # Last uploaded image URL for context in multi-turn conversations
    image_cache = {}           # Cache image descriptions to avoid redundant AI vision calls
    
    # Shopping Cart State
    # ------------------
    persistent_cart = []  # Shopping cart maintained across all agent interactions
    raw_io_history = deque(maxlen=100)  # Complete I/O history for cart state management
    
    # Conversation Management
    # ----------------------
    bad_prompts = set()  # Track prompts that triggered content filters for history redaction

    async def run_customer_loyalty_task(customer_id):
        """
        Background task: Calculate customer discount using Customer Loyalty Agent.
        
        This runs asynchronously at session start to:
        1. Determine customer's loyalty tier and discount percentage
        2. Store discount for application to cart operations
        3. Prepare loyalty message for display after cart interactions
        
        The loyalty response is intentionally NOT sent immediately - it's stored
        and sent after the first cart operation to avoid overwhelming the user.
        """
        start_time = time.time()
        with tracer.start_as_current_span("Run Customer Loyalty Thread"):
            nonlocal session_discount_percentage, session_loyalty_response
            message = f"Calculate discount for the customer with id {customer_id}"
            customer_loyalty_id = validated_env_vars.get('customer_loyalty')
            if not customer_loyalty_id:
                session_loyalty_response = {"answer": "Customer loyalty agent not configured.", "agent": "customer_loyalty"}
                log_timing("Customer Loyalty Task", start_time, "Agent not configured")
                return
                
            processor = get_or_create_agent_processor(
                agent_id=customer_loyalty_id,
                agent_type="customer_loyalty",
                thread_id=customer_loyalty_thread.id,
                project_client=project_client
            )
            bot_reply = ""
            async for msg in processor.run_conversation_with_text_stream(input_message=message):
                bot_reply = extract_bot_reply(msg)
            parsed_response = parse_agent_response(bot_reply)
            parsed_response["agent"] = "customer_loyalty"  # Override agent field
            
            # Store the discount_percentage for the session
            if parsed_response.get("discount_percentage"):
                session_discount_percentage = parsed_response["discount_percentage"]
            session_loyalty_response = parsed_response  # Store the full response for later
            # Do NOT send the response here!
            log_timing("Customer Loyalty Task", start_time, f"Discount: {session_discount_percentage}")

    # Run customer loyalty task only once when session starts
    customer_id = "CUST001"
    if not customer_loyalty_executed:
        asyncio.create_task(run_customer_loyalty_task(customer_id))
        customer_loyalty_executed = True


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

            # # Step 1: Single-agent example
            # print("Generating single-agent response")
            # try:
            #     response = generate_response(user_message)
            #     await websocket.send_text(fast_json_dumps({"answer": response, "agent": "single", "cart": persistent_cart}))
            # except Exception as e:
            #     logger.error("Error during single-agent response generation", exc_info=True)
            #     await websocket.send_text(fast_json_dumps({"answer": "Error during single-agent response generation", "error": str(e), "cart": persistent_cart}))

        
            # # Step 2: MCP Tools agent example
            # # Question: Can you recommend products for redecorating a small bathroom?
            # print("Generating MCP Tools agent response")
            # try:
            #     response = await generate_response_using_tools(user_message)
            #     await websocket.send_text(fast_json_dumps({"answer": response, "agent": "single", "cart": persistent_cart}))
            # except Exception as e:
            #     logger.error("Error during mcp-tools-agent response generation", exc_info=True)
            #     await websocket.send_text(fast_json_dumps({"answer": "Error during mcp-tools-agent response generation", "error": str(e), "cart": persistent_cart}))

            # =============================================================================
            # INTELLIGENT HANDOFF: Intent Classification & Agent Selection
            # =============================================================================
            # The HandoffService analyzes the user's message and conversation context to
            # determine which specialized agent should handle the request. This replaces
            # keyword-based routing with LLM-powered intent understanding.
            #
            # Process:
            # 1. Format conversation history (redact filtered content)
            # 2. Call HandoffService.classify_intent() with message + context
            # 3. Receive structured classification with domain, confidence, reasoning
            # 4. Route to appropriate agent based on classification
            #
            # Supported Domains:
            # - cora: General shopping assistance
            # - interior_designer: Design recommendations and image creation
            # - cart_manager: Shopping cart operations
            # - inventory_agent: Stock availability checks
            # - customer_loyalty: Discount and promotion queries
            # =============================================================================
            try:
                handoff_start_time = time.time()
                formatted_history = format_chat_history(redact_bad_prompts_in_history(chat_history, bad_prompts))
                logger.debug("Handoff agent execution initiated - commencing agent selection protocol")
                
                with tracer.start_as_current_span("Handoff Intent Classification"):
                    # Intent classification using structured outputs for reliable routing
                    intent_result = handoff_service.classify_intent(
                        user_message=user_message,
                        session_id=session_id,
                        chat_history=formatted_history
                    )
                
                # Extract agent information from classification result
                agent_name = intent_result["agent_id"]  # e.g., "cora", "cart_manager"
                agent_selected = validated_env_vars.get(agent_name)  # Get agent ID from environment
                
                logger.debug(f"Intent classification: domain={intent_result['domain']}, "
                           f"confidence={intent_result['confidence']:.2f}, "
                           f"reasoning={intent_result['reasoning']}")
                log_timing("Handoff Processing", handoff_start_time, 
                          f"Selected: {agent_name} (confidence: {intent_result['confidence']:.2f})")
                
                # Check if agent selection failed
                if not agent_selected or not agent_name:
                    await websocket.send_text(fast_json_dumps({
                        "answer": "Sorry, I could not determine the right agent.",
                        "agent": None,
                        "cart": persistent_cart
                    }))
                    continue
                    
            except Exception as e:
                logger.error("Error during handoff classification", exc_info=True)
                await websocket.send_text(fast_json_dumps({
                    "answer": "Error during handoff classification",
                    "error": str(e),
                    "cart": persistent_cart
                }))
                continue
            
            # =============================================================================
            # UNIFIED AGENT EXECUTION: Context Enrichment & Agent Processing
            # =============================================================================
            # All agents now follow a consistent execution pattern:
            # 1. Context Enrichment: Add multimodal data (images, videos, products)
            # 2. Agent-Specific Preparation: Format context based on agent needs
            # 3. Agent Execution: Use AgentProcessor for streaming responses
            # 4. Response Handling: Parse output, update state, send to user
            #
            # No more if-else branching - all agents use the same processor pattern!
            # =============================================================================
            try:
                agent_execution_start_time = time.time()
                logger.debug(f"{agent_name} agent execution initiated")
                
                # Initialize context enrichment variables
                enriched_message = user_message  # Base message
                image_data = None                # Image description from vision analysis
                video_summary = None             # Video summary from analysis
                products = None                  # Product recommendations from AI Search
                
                # =============================================================================
                # MULTIMODAL CONTENT PROCESSING: Enrich context with visual data
                # =============================================================================
                # Process images and videos to add visual understanding to the agent's context.
                # This enables contextually aware recommendations based on what the user shares.
                # =============================================================================
                
                # Process multimodal inputs if present
                if image_url or video_url:
                    if image_url:
                        # IMAGE ANALYSIS: Extract visual information from uploaded images
                        # Uses phi-4 vision model with caching to avoid re-analyzing same image
                        # Results are cached for the session and shared across agents
                        image_start_time = time.time()
                        log_cache_status(image_cache, image_url)
                        image_data = await get_cached_image_description(image_url, image_cache)
                        log_timing("Image Analysis", image_start_time, f"URL: {image_url[:50]}...")
                        
                        # Send analysis message to user (provides feedback during processing)
                        analysis_msg = get_rotating_message(IMAGE_ANALYSIS_MESSAGES)
                        await websocket.send_text(fast_json_dumps({
                            "answer": analysis_msg,
                            "agent": agent_name,
                            "cart": persistent_cart
                        }))
                        logger.debug("Image analysis completed")
                    
                    if video_url:
                        # VIDEO ANALYSIS: Summarize video content using vision model
                        # Extracts key frames and provides comprehensive summary
                        # Useful for room tours, product demonstrations, etc.
                        video_start_time = time.time()
                        logger.debug("Video analysis initiated")
                        video_summary = await get_video_summary(video_url)
                        
                        # Send upload confirmation (acknowledges receipt)
                        thank_you_msg = get_rotating_message(VIDEO_UPLOAD_MESSAGES)
                        await websocket.send_text(fast_json_dumps({
                            "answer": thank_you_msg,
                            "agent": agent_name,
                            "cart": persistent_cart
                        }))
                        
                        log_timing("Video Analysis", video_start_time, f"URL: {video_url[:50]}...")
                        
                        # Send analysis message (indicates processing in progress)
                        analysis_msg = get_rotating_message(VIDEO_ANALYSIS_MESSAGES)
                        await websocket.send_text(fast_json_dumps({
                            "answer": analysis_msg,
                            "agent": agent_name,
                            "cart": persistent_cart
                        }))
                        logger.debug("Video analysis completed")
                
                # =============================================================================
                # PRODUCT RECOMMENDATIONS: AI Search integration for contextual products
                # =============================================================================
                # For agents that make product recommendations, query AI Search with enriched
                # context (user message + visual analysis). This provides the agent with
                # relevant product options to suggest based on user needs and visual context.
                # =============================================================================
                
                # Get product recommendations for relevant agents
                if agent_name in ["interior_designer", "interior_designer_create_image", "cora"]:
                    product_start_time = time.time()
                    # Build search query from all available context
                    search_query = user_message
                    if image_data:
                        # Add visual context to search (e.g., "blue living room" → search for blue paint)
                        search_query += f" {image_data} paint accessories, paint sprayers, drop cloths, painters tape"
                    if video_summary:
                        # Add video context to search (e.g., room tour → search for room-specific products)
                        search_query += f" {video_summary} paint accessories, paint sprayers, drop cloths, painters tape"
                    
                    products = product_recommendations(search_query)
                    log_timing("Product Recommendations", product_start_time, f"Found: {len(products) if products else 0}")
                    logger.debug("Product recommendations completed")
                
                # =============================================================================
                # CONTEXT ENRICHMENT: Build complete message with all available context
                # =============================================================================
                # Combine user message with multimodal analysis and product data to create
                # a comprehensive context for the agent. This enables more accurate and
                # contextually relevant responses.
                # =============================================================================
                
                # Build enriched message with all context
                if image_data or video_summary or products:
                    context_parts = []
                    if image_data:
                        context_parts.append(f"Image description: {image_data}")
                    if video_summary:
                        context_parts.append(f"Video description: {video_summary}")
                    if products:
                        context_parts.append(f"Available products: {fast_json_dumps(products)}")
                    
                    # Prepend user message, append all enriched context
                    enriched_message = f"{user_message}\n\n" + "\n".join(context_parts)
                
                # =============================================================================
                # AGENT EXECUTION: Unified pattern for all agents (no if-else branching!)
                # =============================================================================
                # All agents now follow the same execution flow:
                # 1. Prepare agent-specific context (raw_io_history, conversation, enriched message)
                # 2. Get or create AgentProcessor for the selected agent
                # 3. Stream response chunks from the agent
                # 4. Parse structured response and update session state
                #
                # SPECIAL CASE: interior_designer_create_image uses DALL-E instead of agent
                # =============================================================================
                
                # Execute agent based on type - unified agent processor pattern
                bot_reply = ""
                
                with tracer.start_as_current_span(f"{agent_name.title()} Agent Call"):
                    # =================================================================
                    # SPECIAL CASE: Image Creation (uses DALL-E, not agent processor)
                    # =================================================================
                    # This is the only case that doesn't use the unified agent pattern
                    # because it generates images via DALL-E API rather than conversing
                    if agent_name == "interior_designer_create_image":
                        # Acknowledge image creation request
                        thank_you_msg = get_rotating_message(IMAGE_CREATE_MESSAGES)
                        await websocket.send_text(fast_json_dumps({
                            "answer": thank_you_msg,
                            "agent": "interior_designer",
                            "cart": persistent_cart
                        }))
                        
                        # Use persistent image URL for context (e.g., "make this room blue")
                        if persistent_image_url:
                            image_data = await get_cached_image_description(persistent_image_url, image_cache)
                            enriched_message = f"{user_message} {image_data}"
                        
                        # Create image using DALL-E
                        image = create_image(text=enriched_message, image_url=persistent_image_url)
                        
                        # Build response with generated image URL
                        response_data = {
                            "answer": "Here is the requested image",
                            "products": "",
                            "discount_percentage": session_discount_percentage or "",
                            "image_url": image,
                            "video_url": "",
                            "additional_data": "",
                            "cart": persistent_cart
                        }
                        
                        # Send response
                        response_json = fast_json_dumps(response_data)
                        raw_io_history.append({"output": response_json, "cart": persistent_cart})
                        
                        bot_answer = response_data.get("answer", "")
                        product_names = extract_product_names_from_response(response_data)
                        chat_history.append(("bot", bot_answer + product_names))
                        
                        await websocket.send_text(response_json)
                        log_timing("Agent Execution", agent_execution_start_time, f"Agent: {agent_name}")
                        continue  # Skip common response handling
                    
                    # =================================================================
                    # AGENT-SPECIFIC CONTEXT PREPARATION
                    # =================================================================
                    # Each agent type receives context tailored to its needs:
                    # - cart_manager: Full raw I/O history for tracking cart state changes
                    # - cora: Formatted conversation history for contextual responses
                    # - Others: Enriched message with multimodal + product data
                    # =================================================================
                    
                    # Prepare context based on agent type
                    agent_context = enriched_message  # Default: enriched message
                    
                    # Cart manager needs full raw_io_history for state management
                    if agent_name == "cart_manager":
                        # Provide complete interaction history so cart_manager can track
                        # all add/remove operations and maintain accurate cart state
                        agent_context = f"{enriched_message}\n\nRAW_IO_HISTORY:\n{fast_json_dumps(list(raw_io_history), option=orjson.OPT_INDENT_2)}"
                    
                    # Cora needs conversation history for contextual dialogue
                    elif agent_name == "cora":
                        # Provide formatted chat history so cora can reference previous
                        # conversation turns and maintain coherent multi-turn dialogue
                        agent_context = f"{formatted_history}\n\nUser: {enriched_message}"
                    
                    # =================================================================
                    # UNIFIED AGENT PROCESSOR EXECUTION (All agents use this pattern!)
                    # =================================================================
                    # Get or create an AgentProcessor instance for the selected agent.
                    # The processor manages the agent's execution lifecycle and streams
                    # responses back token-by-token for a better user experience.
                    #
                    # This replaces the old if-else branching with a single unified flow.
                    # =================================================================
                    
                    # All agents use unified agent processor pattern
                    processor = get_or_create_agent_processor(
                        agent_id=agent_selected,     # Agent ID from environment variables
                        agent_type=agent_name,       # Agent type (cora, cart_manager, etc.)
                        thread_id=thread.id,         # Conversation thread for stateful agents
                        project_client=project_client  # Azure AI client for agent execution
                    )
                    
                    # Stream response from agent (yields chunks as they're generated)
                    async for msg in processor.run_conversation_with_text_stream(input_message=agent_context):
                        bot_reply = extract_bot_reply(msg)  # Extract text from streaming message
                
                logger.debug(f"{agent_name} agent execution completed")
                
                log_timing("Agent Execution", agent_execution_start_time, f"Agent: {agent_name}")
                
                # =============================================================================
                # RESPONSE PROCESSING: Parse structured output and update session state
                # =============================================================================
                # Agents return structured JSON responses with fields like answer, products,
                # discount_percentage, image_url, etc. We parse this, update session state,
                # and send formatted response to the user.
                # =============================================================================
                
                # Parse the response first to get structured fields (answer, products, etc.)
                parsed_response = parse_agent_response(bot_reply)
                parsed_response["agent"] = agent_name  # Override agent field to show which agent responded
                
                # =============================================================================
                # CART STATE UPDATE: Persist cart changes from cart_manager agent
                # =============================================================================
                # The cart_manager agent returns an updated cart array in its response.
                # We persist this to the session so all subsequent messages see the updated cart.
                # =============================================================================
                
                # Update persistent_cart if cart_manager returned a cart
                if agent_name == "cart_manager" and "cart" in parsed_response:
                    if isinstance(parsed_response.get("cart"), list):
                        persistent_cart = parsed_response["cart"]
                        logger.debug(f"Cart updated by cart_manager: {len(persistent_cart)} items")
                
                # =============================================================================
                # CONVERSATION HISTORY UPDATE: Maintain chat context for multi-turn dialogue
                # =============================================================================
                # Add the bot's response to chat history so future messages have context.
                # Clean the history to remove large product data (keep only product names).
                # =============================================================================
                
                # Add the bot reply to chat history with products if available
                bot_answer = parsed_response.get("answer", bot_reply or "")
                product_names = extract_product_names_from_response(parsed_response)
                chat_history.append(("bot", bot_answer + product_names))
                print(f"Chat history after bot reply: {chat_history}")
                
                # Clean the conversation history to remove large product data (keep compact)
                chat_history = clean_conversation_history(chat_history)
                print(f"Chat history after bot reply: {chat_history}")
                
                # =============================================================================
                # DISCOUNT PERSISTENCE: Maintain customer loyalty tier across session
                # =============================================================================
                # Once the customer_loyalty agent calculates a discount, persist it across
                # all subsequent responses so the user sees consistent discount information.
                # =============================================================================
                
                # Update session discount_percentage if a new one is received
                if parsed_response.get("discount_percentage"):
                    session_discount_percentage = parsed_response["discount_percentage"]
                
                # Include session discount_percentage in all responses if available
                if session_discount_percentage and not parsed_response.get("discount_percentage"):
                    parsed_response["discount_percentage"] = session_discount_percentage
                
                # =============================================================================
                # RESPONSE TRANSMISSION: Send structured response to user
                # =============================================================================
                # Send the final response with all fields (answer, products, cart, discount, etc.)
                # Also append to raw_io_history for cart_manager's state tracking.
                # =============================================================================
                
                # When sending any other response, also append to raw_io_history
                response_json = fast_json_dumps({**parsed_response, "cart": persistent_cart})
                raw_io_history.append({"output": response_json, "cart": persistent_cart})
                await websocket.send_text(response_json)
                
                # =============================================================================
                # DELAYED LOYALTY RESPONSE: Send loyalty message after cart operations
                # =============================================================================
                # The customer_loyalty agent runs in background at session start.
                # Its response is delayed until after the first cart_manager operation,
                # ensuring users see their loyalty tier after interacting with the cart.
                # =============================================================================
                
                # After cart_manager response, send loyalty response if available (only once per session)
                if agent_name == "cart_manager" and session_loyalty_response and not loyalty_response_sent:
                    loyalty_response_with_cart = {**session_loyalty_response, "cart": persistent_cart}
                    await websocket.send_text(fast_json_dumps(loyalty_response_with_cart))
                    loyalty_response_sent = True
                    
            # =============================================================================
            # ERROR HANDLING: Failure during agent execution
            # =============================================================================
            # If agent execution fails, catch the exception, log it for debugging,
            # and send a user-friendly error message with the current cart state.
            # =============================================================================
            except Exception as e:
                logger.error("Error in agent execution", exc_info=True)
                try:
                    await websocket.send_text(fast_json_dumps({
                        "answer": "Internal server error",
                        "error": str(e),
                        "cart": persistent_cart
                    }))
                except Exception:
                    pass  # If even error sending fails, silently continue
    
    # =============================================================================
    # SESSION-LEVEL ERROR HANDLING: Catch WebSocket disconnects and errors
    # =============================================================================
    # Handle normal disconnections (user closes tab) and unexpected session errors.
    # Log all errors for monitoring and debugging.
    # =============================================================================
    except WebSocketDisconnect:
        pass  # Normal disconnection, no action needed
    except Exception as e:
        logger.error("WebSocket session error", exc_info=True)
        try:
            await websocket.send_text(fast_json_dumps({
                "answer": "Internal server error",
                "error": str(e),
                "cart": persistent_cart
            }))
        except Exception:
            pass  # If sending error fails, give up gracefully
    
    # =============================================================================
    # SESSION CLEANUP: Log session duration and cleanup resources
    # =============================================================================
    # When the WebSocket connection closes (user disconnects, network error, etc.),
    # log the total session duration for monitoring and performance analysis.
    # =============================================================================
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

        
