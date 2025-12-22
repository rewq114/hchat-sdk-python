from typing import AsyncGenerator, Dict, Any, List
import httpx
import json

from .base import BaseProvider
from ..types.request import LLMRequest, MessageRole
from ..types.response import LLMResponse, ResponseChunk

class GoogleProvider(BaseProvider):
    async def complete(self, request: LLMRequest) -> LLMResponse:
        url = self._get_url(request, stream=False)
        payload = self._create_payload(request)
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers={'Content-Type': 'application/json'}, json=payload, timeout=60.0)
            response.raise_for_status()
            data = response.json()
            # Need to map Google response to LLMResponse
            return self._map_response(data, request)

    async def stream(self, request: LLMRequest) -> AsyncGenerator[ResponseChunk, None]:
        url = self._get_url(request, stream=True)
        payload = self._create_payload(request)
        
        async with httpx.AsyncClient() as client:
            # Google SSE format might differ slightly or use standard SSE
            # Node SDK adds '&alt=sse'
            async with client.stream("POST", url, headers={'Content-Type': 'application/json'}, json=payload, timeout=60.0) as response:
                 response.raise_for_status()
                 async for line in response.aiter_lines():
                     if line.startswith("data: "):
                         try:
                             chunk = json.loads(line[6:])
                             yield self._map_chunk(chunk)
                         except:
                             continue

    def _get_url(self, request: LLMRequest, stream: bool) -> str:
        method = 'streamGenerateContent' if stream else 'generateContent'
        base = request.api_base
        if 'models' not in base:
            base = f"{base.rstrip('/')}/models/"
        
        url = f"{base}{request.model}:{method}?key={request.api_key}"
        if stream:
            url += "&alt=sse"
        return url

    def _create_payload(self, request: LLMRequest) -> Dict[str, Any]:
        contents = []
        for m in request.messages:
            role = 'user' if m.role == MessageRole.USER else 'model'
            parts = []
            if isinstance(m.content, str):
                parts.append({"text": m.content})
            elif isinstance(m.content, list):
                # Simplified block handling
                for block in m.content:
                    if block.type == 'text':
                        parts.append({"text": block.text})
                    # Add other types as needed
            contents.append({"role": role, "parts": parts})

        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": request.max_tokens,
                "temperature": request.temperature,
                "topP": request.top_p,
                "topK": request.top_k,
                "stopSequences": request.stop
            }
        }
        
        if request.system:
            payload["systemInstruction"] = { "parts": [{ "text": request.system }] }
            
        return payload

    def _map_response(self, data: Dict[str, Any], request: LLMRequest) -> LLMResponse:
        content_text = ""
        candidates = data.get('candidates', [])
        if candidates:
            parts = candidates[0].get('content', {}).get('parts', [])
            for p in parts:
                if 'text' in p:
                    content_text += p['text']
        
        # Simplified mapping
        return LLMResponse(
            id="resp_google_placeholder",
            model=request.model,
            created=0,
            usage={"promptTokens": 0, "completionTokens": 0, "totalTokens": 0},
            choices=[{
                "index": 0, 
                "message": {"role": MessageRole.ASSISTANT, "content": content_text},
                "finishReason": "stop"
            }]
        )

    def _map_chunk(self, chunk: Dict[str, Any]) -> ResponseChunk:
        candidates = chunk.get('candidates', [])
        text = ""
        if candidates:
            parts = candidates[0].get('content', {}).get('parts', [])
            for p in parts:
                if 'text' in p:
                    text += p['text']
        
        return {
            "type": "stream_delta",
            "content": {
                "type": "text_delta",
                "text": text
            }
        }
