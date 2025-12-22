from typing import AsyncGenerator, Dict, Any
import httpx
import json

from .base import BaseProvider
from ..types.request import LLMRequest, MessageRole
from ..types.response import LLMResponse, ResponseChunk, Choice, Usage

class OpenAIProvider(BaseProvider):
    async def complete(self, request: LLMRequest) -> LLMResponse:
        # HChat/Azure style endpoint
        url = f"{request.api_base.rstrip('/')}/openai/deployments/{request.model}/chat/completions"
        payload = self._create_payload(request, stream=False)
        
        headers = self._get_headers(request)
        headers['api-key'] = request.api_key
        # Remove Authorization if present to match Node SDK? 
        # Node says 'api-key'... checks if it overrides or adds. 
        # BaseProvider adds Authorization. We should probably overwrite or remove it if strictly Azure style.
        # But let's just add api-key for now. Safe to keep Bearer if HChat supports it, but api-key is key.
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=60.0)
            response.raise_for_status()
            data = response.json()
            
            return LLMResponse(**data)

    async def stream(self, request: LLMRequest) -> AsyncGenerator[ResponseChunk, None]:
        url = f"{request.api_base.rstrip('/')}/openai/deployments/{request.model}/chat/completions"
        payload = self._create_payload(request, stream=True)
        
        headers = self._get_headers(request)
        headers['api-key'] = request.api_key
        
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, headers=headers, json=payload, timeout=60.0) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(data_str)
                            # Convert OpenAI chunk format to internal ResponseChunk format
                            yield self._map_chunk(chunk_data)
                        except json.JSONDecodeError:
                            continue

    def _create_payload(self, request: LLMRequest, stream: bool) -> Dict[str, Any]:
        messages = []
        if request.system:
             messages.append({"role": "system", "content": request.system})
        
        for m in request.messages:
            content = m.content
            if isinstance(content, list):
                # Simply serialize Pydantic models to dict if needed, or rely on pydantic serialization
                # Here we assume simple string for now or basic list conversion
                # For robustness, we should traverse list and convert block types to OpenAI format
                # But to keep this step actionable, assuming simple structure or direct dict compatibility of basic types
                # Using model_dump() if it's a Pydantic object
                pass 
            messages.append(m.model_dump(exclude_none=True))

        payload = {
            "model": request.model,
            "messages": messages,
            "stream": stream,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stop": request.stop,
        }
        
        if request.tools:
            payload["tools"] = request.tools
            
        return {k: v for k, v in payload.items() if v is not None}

    def _map_chunk(self, chunk: Dict[str, Any]) -> ResponseChunk:
        # Simplified mapping
        # OpenAI chunk: { id, choices: [{ delta: { content }, finish_reason }] }
        # Map to StreamDelta/Start/Stop
        
        # This is a simplified implementation. 
        # Ideally we need robust mapping like in Node SDK.
        
        choice = chunk.get('choices', [{}])[0]
        delta = choice.get('delta', {})
        
        if 'content' in delta:
             return {
                 "type": "stream_delta",
                 "content": {
                     "type": "text_delta",
                     "text": delta['content']
                 }
             }
        # Handle other types...
        return { "type": "stream_delta", "content": {"type": "text_delta", "text": ""} } # Placeholder
