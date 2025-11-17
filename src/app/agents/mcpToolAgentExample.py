import asyncio
import json
import os
import base64
from typing import List, Dict, Any
from openai import AzureOpenAI
from dotenv import load_dotenv
import numpy as np
import time

from ..servers.mcp_inventory_client import get_mcp_client

# Load environment variables (Azure endpoint, deployment, keys, etc.)
load_dotenv()

endpoint = os.getenv("gpt_endpoint")
deployment = os.getenv("gpt_deployment")
api_key = os.getenv("gpt_api_key")
api_version = os.getenv("gpt_api_version")
mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp_inventory/sse")


# Initialize Azure OpenAI client for GPT-4.1 model
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version=api_version,
)


async def generate_response_using_tools(text_input):
    start_time = time.time()
    """
    Input:
        text_input (str): The user's chat input.

    Output:
        response (str): A Markdown-formatted response from the agent.
    """

    mcp_client = await get_mcp_client(mcp_server_url)

    tools = await mcp_client.get_mcp_tools_llm()
    
    # Prepare the full chat prompt with system and user messages
    chat_prompt = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": """Interior Design Agent Guidelines
========================================
- You are a Interior Designer sales person working for Zava and help customers who need help in DIY Projects and other interior design queries
- Your main tasks are the following: recommending and upselling products, creating images
- You will get input in the form of a json, having:
[
    {
        "Conversation_history":the Conversation thats going on,
        "image_url": Image based on which you need to recreate some image
        "image_description": If there is an image attached, the description or it will be empty
        "video_description": description of video if attached
        "products_available": A list of products, from where you can give recommendations
        "user_last_query": The last query from user
    }
]
- You will always recommend product from the products_available.
- You will keep asking questions to the user and keep recommending.
- When you get video or image, reply saying "I see you uploaded..."
- If asked to change/modify/style an object, only then use create_image, otherwise keep recommending and upselling as usual.
- In your answer do not mention e.g. word instead use Example, such as or like based on the sentence.

Return response in following json format

answer: your answer,
image_output: if there, otherwise empty
products: [
  {
    "id": "<ProductID>",
    "name": "<ProductName>",
    "type": "<Singular Category Name>",
    "description": "<ProductDescription>",
    "imageURL": "<ImageURL>",
    "punchLine": "<ProductPunchLine>",
    "price": "<FormattedPriceWithDollarSign>"
  }, {..}
  ...
]


Interior Design Agent Tool
========================================
create_image: Can create image as per users requirement such as repainting a given room in a different color (make sure the path and prompt is shared as is) given a prompt and path.

Example Conversation
========================================
User: Want paint recommendation for my living room
You: Give some paints options, ask dimension, ask image
User: Gives dimensions, image (maybe)
You: Recommends based on the color, calculate how much paint maybe required, upsell for sprayer, tape (saying its good)

Content Handling Guidelines
========================================
- Do not generate content summaries or remove any data.

---
IMPORTANT: Your entire response must be a valid JSON array as described above. Do not include any other text or formatting.
                    """
                }
            ]
        },
        {"role": "user", "content": text_input}
    ]

    # Call Azure OpenAI chat API
    completion = client.chat.completions.create(
        model=deployment,
        messages=chat_prompt,
        max_completion_tokens=10000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False,
        tools=tools,
        tool_choice="auto"
    )
    
    # Get assistant's response
    assistant_message = completion.choices[0].message

    # Initialize conversation with user query and assistant response
    messages = [
        {"role": "user", "content": text_input},
        assistant_message,
    ]

    try:
        # Handle tool calls if present
        if assistant_message.tool_calls:
            # Process each tool call
            for tool_call in assistant_message.tool_calls:
                # Execute tool call
                result = await mcp_client.call_tool(
                    tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments),
                )

                # Ensure result is a string for the tool response
                result_str = json.dumps(result) if isinstance(result, (dict, list)) else str(result)

                # Add tool response to conversation
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_str,
                    }
                )

                # Get final response from OpenAI with tool results
                final_response = client.chat.completions.create(
                    model=deployment,
                    messages=messages,
                    tools=tools,
                    tool_choice="none",  # Don't allow more tool calls
                )
                end_sum = time.time()
                print(f"MCP Tools Demonstrative agent :generate_response_using_tools: Execution Time: {end_sum - start_time} seconds")

                return final_response.choices[0].message.content

        # No tool calls, just return the direct response
        return assistant_message.content
    except Exception as e:
        print(f"Error during tool execution: {str(e)}")
        raise e
    
    # Return response content
    return completion.choices[0].message.content