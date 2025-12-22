import pytest
import respx
from httpx import Response
from hchat_sdk import HChat, MessageRole

@pytest.mark.asyncio
async def test_hchat_openai_complete():
    api_key = "test-key"
    client = HChat(model="gpt-4o", api_key=api_key)
    
    async with respx.mock:
        route = respx.post("https://h-chat-api.autoever.com/v2/api/openai/deployments/gpt-4o/chat/completions").mock(
            return_value=Response(200, json={
                "id": "test-id",
                "model": "gpt-4o",
                "created": 1234567890,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello World"},
                    "finish_reason": "stop"
                }],
                "usage": {"promptTokens": 1, "completionTokens": 1, "totalTokens": 2}
            })
        )
        
        response = await client.complete("Hi")
        assert response.choices[0].message.content == "Hello World"
        assert route.called

@pytest.mark.asyncio
async def test_hchat_anthropic_complete():
    api_key = "test-key"
    client = HChat(model="claude-sonnet-4-5", api_key=api_key)
    
    async with respx.mock:
        route = respx.post("https://h-chat-api.autoever.com/v2/api/claude/messages").mock(
            return_value=Response(200, json={
                "id": "test-id",
                "model": "claude-sonnet-4-5",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello Claude"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 5}
            })
        )
        
        response = await client.complete("Hi Claude")
        assert response.choices[0].message.content == "Hello Claude"
        assert response.model == "claude-sonnet-4-5"
        assert route.called
