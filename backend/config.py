"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# Provider registry — keyed by model prefix (e.g., "openai" in "openai/gpt-5.1")
PROVIDERS = {
    "openai": {
        "base_url": "https://api.openai.com/v1/chat/completions",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "format": "openai",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1/messages",
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "format": "anthropic",
    },
    "google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "format": "openai",
    },
    "x-ai": {
        "base_url": "https://api.x.ai/v1/chat/completions",
        "api_key": os.getenv("XAI_API_KEY"),
        "format": "openai",
    },
}

# Council members - list of model identifiers (provider/model)
COUNCIL_MODELS = [
    "openai/gpt-5.1",
    "google/gemini-3-pro-preview",
    "anthropic/claude-sonnet-4-5",
    "x-ai/grok-4",
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "google/gemini-3-pro-preview"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
