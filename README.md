# HChat SDK for Python

A robust Python 3.12+ SDK for HChat, mirroring the functionality of the Node.js SDK.

## Features
- **Multi-Provider Support**: Seamlessly switch between OpenAI, Anthropic, Google (Gemini), and HChat native models.
- **Unified Interface**: Consistent `complete()` and `stream()` methods across all providers.
- **Strict Typing**: Full Pydantic V2 models for requests, responses, and stream events.
- **Automatic Mapping**: Handles provider-specific API differences (e.g., Azure-style endpoints for OpenAI) automatically.

## Installation

```bash
pip install hchat-sdk-python
# or with uv
uv add hchat-sdk-python
```

## Configuration

Set your API key in the environment variables:

```bash
export API_KEY="your_api_key_here"
# or
export HCHAT_API_KEY="your_api_key_here"
```

## Usage

### Basic Chat

```python
import asyncio
import os
from hchat_sdk import HChat

async def main():
    client = HChat(
        model="gpt-4o", 
        api_key=os.getenv("API_KEY")
    )
    
    response = await client.complete("Hello, world!")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    asyncio.run(main())
```

### Streaming with Thinking

```python
import asyncio
from hchat_sdk import HChat

async def main():
    client = HChat(model="gpt-5-mini", api_key="...")
    
    async for chunk in client.stream("Explain quantum physics", reasoning=True):
        if chunk.type == "stream_delta":
            if chunk.content.type == "thinking_delta":
                print(f"[Thinking] {chunk.content.thinking}", end="")
            elif chunk.content.type == "text_delta":
                print(chunk.content.text, end="")

if __name__ == "__main__":
    asyncio.run(main())
```

## Review & Verification

The SDK includes a comprehensive test suite to validate model support and functionality.

1. **Validation Tests**: Checks strict model support policies.
   ```bash
   uv run pytest tests/test_validation.py
   ```

2. **Verification Tests**: Functional smoke tests against the live API (requires API_KEY).
   ```bash
   uv run pytest tests/test_verification.py
   ```

## Supported Models

- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-5-mini`, etc.
- **Anthropic**: `claude-sonnet-4-5`, `claude-3-5-sonnet-v2`, etc.
- **Google**: `gemini-2.0-flash`, `gemini-2.5-pro`, etc.
