from typing import AsyncGenerator, Dict, Any
import httpx
import json

from .base import BaseProvider
from ..types.request import LLMRequest, MessageRole
from ..types.response import LLMResponse, ResponseChunk

class AnthropicProvider(BaseProvider):
    async def complete(self, request: LLMRequest) -> LLMResponse:
        url = f"{request.api_base.rstrip('/')}/claude/messages"
        payload = self._create_payload(request, stream=False)
        
        async with httpx.AsyncClient() as client:
            headers = self._get_headers(request)
            headers['anthropic-version'] = '2023-06-01' # Ensure version
            
            response = await client.post(url, headers=headers, json=payload, timeout=60.0)
            response.raise_for_status()
            return self._map_response(response.json(), request)

    def _map_response(self, data: Dict[str, Any], request: LLMRequest) -> LLMResponse:
        content_full = ""
        for block in data.get('content', []):
            if block.get('type') == 'text':
                content_full += block.get('text', "")
        
        return LLMResponse(
            id=data.get('id', 'unknown'),
            model=data.get('model', request.model),
            created=1234567890, 
            usage={
                "promptTokens": data.get('usage', {}).get('input_tokens', 0),
                "completionTokens": data.get('usage', {}).get('output_tokens', 0),
                "totalTokens": data.get('usage', {}).get('input_tokens', 0) + data.get('usage', {}).get('output_tokens', 0)
            },
            choices=[{
                "index": 0,
                "message": {
                    "role": MessageRole.ASSISTANT,
                    "content": content_full
                },
                "finishReason": data.get('stop_reason', 'unknown')
            }]
        )

    async def stream(self, request: LLMRequest) -> AsyncGenerator[ResponseChunk, None]:
        url = f"{request.api_base.rstrip('/')}/claude/messages"
        payload = self._create_payload(request, stream=True)
        headers = self._get_headers(request)
        headers['anthropic-version'] = '2023-06-01'

        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, headers=headers, json=payload, timeout=60.0) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            # Robust mapping needed here
                            # For now returning raw or minimal mapping
                            # Anthropic sends event types: message_start, content_block_delta, etc.
                            yield self._map_event(data)
                        except:
                            continue

    def _create_payload(self, request: LLMRequest, stream: bool) -> Dict[str, Any]:
        messages = [m.model_dump(exclude_none=True) for m in request.messages if m.role != 'system']
        
        payload = {
            "model": request.model,
            "messages": messages,
            "stream": stream,
            "max_tokens": request.max_tokens or 4096,
            "temperature": request.temperature,
            "system": request.system,
        }
        if request.tools:
            payload["tools"] = request.tools
            
        return {k: v for k, v in payload.items() if v is not None}

    def _map_event(self, event: Dict[str, Any]) -> ResponseChunk:
        # Simplified mapping
        event_type = event.get('type')
        if event_type == 'content_block_delta':
            delta = event.get('delta', {})
            if delta.get('type') == 'text_delta':
                return {
                    "type": "stream_delta",
                    "content": {
                        "type": "text_delta",
                        "text": delta.get('text')
                    }
                }
        return { "type": "stream_delta", "content": {"type": "text_delta", "text": ""} } # Placeholder
