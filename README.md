# LLM Council (Direct API Variant)

![llmcouncil](header.jpg)

This is a fork of [karpathy/llm-council](https://github.com/karpathy/llm-council) that **bypasses OpenRouter and calls AI providers directly**.

## Why bypass OpenRouter?

The original repo routes all queries through [OpenRouter](https://openrouter.ai/), a paid proxy that sits between you and the AI providers. The problem: OpenRouter categorises every input using a hosted classifier model and uses that data to track and share user metrics — including on its public [Rankings page](https://openrouter.ai/rankings) — even when you disable logging in your account settings.

This variant removes OpenRouter entirely. Your queries go straight from your machine to OpenAI, Anthropic, Google, and xAI with no middleman. You pay each provider directly at their standard API rates.

## What changed

Three backend files were modified. Zero frontend changes.

| File | Change |
|------|--------|
| `.env` | Replaced single `OPENROUTER_API_KEY` with 4 provider keys |
| `backend/config.py` | Provider registry mapping each prefix to its native API endpoint |
| `backend/openrouter.py` | Routes requests to the correct provider API based on model prefix |

## How it works

The app groups multiple LLMs into a "Council" that collaboratively answers your questions in 3 stages:

1. **Stage 1 — First opinions**: Your query goes to all council models in parallel. You can inspect each response individually.
2. **Stage 2 — Peer review**: Each model receives the other models' responses (anonymized to prevent favoritism) and ranks them by accuracy and insight.
3. **Stage 3 — Final answer**: A designated Chairman model synthesizes everything into a single comprehensive response.

## Setup

### 1. Install dependencies

Requires [uv](https://docs.astral.sh/uv/) for Python and npm for the frontend.

```bash
uv sync
cd frontend && npm install && cd ..
```

### 2. Configure API keys

Create a `.env` file in the project root with your provider keys:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
XAI_API_KEY=xai-...
```

### 3. Configure models (optional)

Edit `backend/config.py` to customize the council. Model identifiers use the format `provider/model-name`:

```python
COUNCIL_MODELS = [
    "openai/gpt-5.1",
    "google/gemini-3-pro-preview",
    "anthropic/claude-sonnet-4-5",
    "x-ai/grok-4",
]

CHAIRMAN_MODEL = "google/gemini-3-pro-preview"
```

Supported providers: `openai`, `anthropic`, `google`, `x-ai`. OpenAI, Google, and xAI all use OpenAI-compatible endpoints. Anthropic is handled separately via its native Messages API.

## Running

```bash
./start.sh
```

Then open http://localhost:5173.

## Tech stack

- **Backend:** FastAPI, async httpx, direct provider APIs
- **Frontend:** React + Vite, react-markdown
- **Storage:** JSON files in `data/conversations/`

## Credit

Original project by [Andrej Karpathy](https://github.com/karpathy/llm-council). This variant modifies only the API layer.
