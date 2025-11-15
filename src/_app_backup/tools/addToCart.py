# # This tool has to be invoked when the user wants to add an item to the cart.
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from .env file
load_dotenv()

# Retrieve credentials and model deployment info
ENDPOINT = os.environ.get("gpt_endpoint")
DEPLOYMENT = os.environ.get("gpt_deployment")
API_KEY = os.environ.get("gpt_api_key")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=ENDPOINT,
    api_key=API_KEY,
    api_version="2024-12-01-preview",
)

# Load the prompt for cart addition task from file
SR_PROMPT_TARGET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'prompts', 'addToCartPrompt.txt')
with open(SR_PROMPT_TARGET, 'r') as file:
    PROMPT = file.read()

def add_products_to_cart(question, product_list):
    """
    Determines which product(s) from a list the user wants to add to their cart,
    based on a natural language query.

    Inputs:
        question (str): User's request.
        product_list (list): List of product objects (dicts) with info like ProductID, ProductName, etc.

    Output:
        str: Response from GPT model indicating which product(s) to add to the cart.
             Typically returned as JSON or structured text with the matching product details.
    """

    # Step 1: Prepare chat prompt for GPT with system prompt, user question, and product list
    chat_prompt = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": PROMPT  # Loaded from addToCartPrompt.txt
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question  # The user's cart-related request
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(product_list)  # Product options for the assistant to choose from
                }
            ]
        }
    ]

    # Step 2: Generate response using Azure OpenAI GPT model
    completion = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=chat_prompt,
        max_tokens=5686,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False
    )

    # Step 3: Return only the assistant's message content
    return completion.choices[0].message.content

# # previously used for testing
# question = " I want 2 gallons of Effervescent Jade paint, add it to  cart."
# product_list = [
#             {
#                 "id": "OM-403",
#                 "name": "Effervescent Jade, Interior Wall Paint, 1 gallon bucket",
#                 "additionaldetails": "Interior Wall Paint, 1-gallon bucket",
#                 "type": "Paint Shade",
#                 "description": "A sparkling, uplifting jade green for spaces brimming with vitality.",
#                 "imageURL": "https://staidemodev.blob.core.windows.net/hero-demos-hardcoded-chat-images/figmaimages/jade-color-interior-design.png",
#                 "punchLine": "Chill out in classic blue",
#                 "price": "$47.99"
#             },
#             {
#                 "id": "OM-403",
#                 "name": "Effervescent Jade, Interior Wall Paint, 1 gallon bucket",
#                 "additionaldetails": "Interior Wall Paint, 1-gallon bucket",
#                 "type": "Paint Shade",
#                 "description": "A sparkling, uplifting jade green for spaces brimming with vitality.",
#                 "imageURL": "https://staidemodev.blob.core.windows.net/hero-demos-hardcoded-chat-images/figmaimages/jade-color-interior-design.png",
#                 "punchLine": "Chill out in classic blue",
#                 "price": "$47.99"
#             },
#             {
#                 "id": "PS-401",
#                 "name": "EZ-Coat Paint Sprayer",
#                 "type": "Paint Sprayer",
#                 "description": "Go cordless and conquer any project with this ultra-portable airless paint sprayer. Delivers smooth, even coverage on walls, decks, and fences\u2014anywhere freedom is needed.",
#                 "imageURL": "https://staidemodev.blob.core.windows.net/hero-demos-hardcoded-chat-images/figmaimages/EZ-Coat%20Paint%20Sprayer.png",
#                 "punchLine": "Spray without limits, anywhere you go!",
#                 "price": "$40.00"
#             },
#             {
#                 "id": "DC-401",
#                 "name": "Drop Cloth",
#                 "type": "Paint Accessories",
#                 "description": "Heavy-duty, reusable drop cloth designed to protect floors and furniture during painting, staining, or remodeling projects. Ideal for both professional painters and DIY enthusiasts seeking reliable surface coverage.",
#                 "imageURL": "https://staidemodev.blob.core.windows.net/hero-demos-hardcoded-chat-images/figmaimages/Paint%20Drop%20Cloth.png",
#                 "punchLine": "Shield your space, paint with confidence.",
#                 "price": "$10.00"
#             }
#         ]


# cart=add_products_to_cart(question, product_list)
# print(f"Cart Response: {cart}")
