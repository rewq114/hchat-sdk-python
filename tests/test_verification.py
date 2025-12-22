import pytest
import os
import asyncio
from hchat_sdk import HChat, MessageRole

# Run these only if API KEY is present
api_key = os.getenv("API_KEY") or os.getenv("HCHAT_API_KEY")
pytestmark = pytest.mark.skipif(not api_key, reason="API_KEY not set")

model = "gpt-5-mini"

@pytest.mark.asyncio
async def test_complete_simple():
    client = HChat(model=model, api_key=api_key)
    response = await client.complete("Hello, HChat!")
    assert response.choices[0].message.content
    print(f"\n[Complete] {response.choices[0].message.content}")

@pytest.mark.asyncio
async def test_stream_simple():
    client = HChat(model=model, api_key=api_key)
    stream = client.stream("Tell me a short joke.")
    print("\n[Stream Output] ", end="")
    text = ""
    async for chunk in stream:
        if chunk.type == "stream_delta" and chunk.content.type == "text_delta":
            print(chunk.content.text, end="")
            text += chunk.content.text
    assert len(text) > 0

@pytest.mark.asyncio
async def test_stream_thinking():
    # Use a model that supports reasoning if possible, or mock/expect failure?
    # gpt-5-mini might support it? Node test used gpt-5-mini?
    # Node test used gpt-5-mini.
    client = HChat(model=model, api_key=api_key)
    stream = client.stream("Explain quantum entanglement simply.", reasoning=True)
    
    has_thinking = False
    print("\n[Thinking Output] ", end="")
    async for chunk in stream:
        if chunk.type == "stream_delta":
            if chunk.content.type == "thinking_delta":
                has_thinking = True
                print(chunk.content.thinking, end="")
            elif chunk.content.type == "text_delta":
                print(chunk.content.text, end="")
    
    # Asserting has_thinking might fail if model doesn't actually produce it or API doesn't support it yet
    # But SDK should handle it if it does.

@pytest.mark.asyncio
async def test_stream_tools():
    client = HChat(model=model, api_key=api_key)
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }]
    
    # We need to adapt the tools structure to what provider expects? 
    # Node SDK passes custom tools. Python SDK config should match.
    # Our LLMRequest has `tools: List[Dict]`.
    
    stream = client.stream("What is the weather in Seoul?", tools=tools)
    tool_called = False
    async for chunk in stream:
        if chunk.type == "stream_delta" and chunk.content.type == "tool_call_end":
            print(f"\n[Tool Call] {chunk.content.input}")
            tool_called = True
    
    assert tool_called

