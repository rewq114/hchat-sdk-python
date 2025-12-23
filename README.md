# HChat SDK for Python

A robust Python 3.12+ SDK for HChat, mirroring the functionality of the Node.js SDK with full parity for all major providers.

## Features

- **Multi-Provider Support**: Seamlessly switch between OpenAI (Azure), Anthropic (Claude), and Google (Gemini).
- **Unified Interface**: Use the same `complete()` and `stream()` methods regardless of the backend.
- **Advanced Capabilities**:
  - **Streaming**: Real-time response handling with support for text, tool calls, and thinking.
  - **Thinking (Reasoning)**: Native support for reasoning-enabled models like Claude 3.7 and Gemini Thinking.
  - **Tool Use (Function Calling)**: Simple interface for multi-tool integration.
  - **Multimodal (Vision)**: Support for image analysis via Base64 or URL.
- **Strict Typing**: Built with Pydantic V2 for robust validation and IDE support.

## Installation

```bash
pip install hchat-sdk-python
# or with uv
uv add hchat-sdk-python
```

## Configuration

Set your API key as an environment variable:

```bash
export HCHAT_API_KEY="your_api_key_here"
```

## Usage

### Basic Chat

```python
import asyncio
from hchat_sdk import HChat

async def main():
    client = HChat(model="gpt-4o")
    
    response = await client.complete("Hello! How can you help me today?")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    asyncio.run(main())
```

### Streaming with Thinking

Perfect for models like `gpt-5-mini` or `claude-sonnet-4-5` that support reasoning.

```python
async for chunk in client.stream("Explain quantum entanglement", reasoning=True):
    if chunk.type == "stream_delta":
        content = chunk.content
        if content.type == "thinking_delta":
            print(f"[Thinking] {content.thinking}", end="")
        elif content.type == "text_delta":
            print(content.text, end="")
```

### Vision (Image Input)

```python
from hchat_sdk.types.request import ContentBlock

image_block = ContentBlock(
    type="image",
    source={
        "type": "base64",
        "media_type": "image/jpeg",
        "data": "..." # base64 string
    }
)

response = await client.complete(messages=[
    {"role": "user", "content": [
        {"type": "text", "text": "What is in this image?"},
        image_block
    ]}
])
```

## Supported Models

| Provider | Key Models | Features |
| :--- | :--- | :--- |
| **Azure (OpenAI)** | `gpt-4o`, `gpt-5-mini` | Vision, Tools, Reasoning |
| **Anthropic** | `claude-sonnet-4-5`, `claude-3-5-sonnet-v2` | Vision, Tools, Thinking |
| **Google** | `gemini-2.0-flash`, `gemini-2.5-pro` | Vision, Tools, Thinking |

## Testing

The SDK uses `pytest` for verification. Set `HCHAT_API_KEY` before running.

```bash
# Run all message tests
uv run pytest tests/test_messages.py -v

# Run model listing tests
uv run pytest tests/test_models.py -v
```
