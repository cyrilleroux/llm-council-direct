"""Multi-provider API client for making LLM requests."""

import httpx
from typing import List, Dict, Any, Optional
from .config import PROVIDERS


def _get_provider_config(model: str) -> tuple[str, str, dict]:
    """
    Split a model identifier into provider config and model name.

    Args:
        model: Model identifier like "openai/gpt-5.1"

    Returns:
        Tuple of (model_name, provider_key, provider_config)

    Raises:
        ValueError: If provider prefix not found in PROVIDERS
    """
    prefix, model_name = model.split("/", 1)
    config = PROVIDERS.get(prefix)
    if not config:
        raise ValueError(f"Unknown provider prefix: {prefix}")
    return model_name, prefix, config


async def _query_openai_compatible(
    model_name: str,
    messages: List[Dict[str, str]],
    config: dict,
    timeout: float,
) -> Optional[Dict[str, Any]]:
    """Query an OpenAI-compatible endpoint (OpenAI, Google, xAI)."""
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name,
        "messages": messages,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            config["base_url"],
            headers=headers,
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        message = data["choices"][0]["message"]

        return {
            "content": message.get("content"),
            "reasoning_details": message.get("reasoning_details"),
        }


async def _query_anthropic(
    model_name: str,
    messages: List[Dict[str, str]],
    config: dict,
    timeout: float,
) -> Optional[Dict[str, Any]]:
    """Query the Anthropic Messages API."""
    headers = {
        "x-api-key": config["api_key"],
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    # Separate system message from the rest
    system_text = None
    non_system = []
    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"]
        else:
            non_system.append(msg)

    payload = {
        "model": model_name,
        "max_tokens": 8192,
        "messages": non_system,
    }
    if system_text:
        payload["system"] = system_text

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            config["base_url"],
            headers=headers,
            json=payload,
        )
        response.raise_for_status()

        data = response.json()

        # Anthropic returns content blocks: [{"type": "text", "text": "..."}]
        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")

        return {
            "content": content,
            "reasoning_details": None,
        }


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0,
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via its native API.

    Args:
        model: Model identifier (e.g., "openai/gpt-5.1")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    try:
        model_name, _prefix, config = _get_provider_config(model)

        if config["format"] == "anthropic":
            return await _query_anthropic(model_name, messages, config, timeout)
        else:
            return await _query_openai_compatible(model_name, messages, config, timeout)

    except Exception as e:
        print(f"Error querying model {model}: {e}")
        return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]],
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    import asyncio

    tasks = [query_model(model, messages) for model in models]
    responses = await asyncio.gather(*tasks)

    return {model: response for model, response in zip(models, responses)}
