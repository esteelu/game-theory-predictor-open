import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file, where we should define the key
load_dotenv()

client = OpenAI()

def get_agent_client():
    """
    Initializes and returns the OpenAI client configured for the gpt-4.0-mini API.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in environment variables.")

    client = OpenAI(api_key=api_key)
    return client

# Model and temperature settings
MODEL_NAME = "gpt-5"
TEMPERATURE = 1
