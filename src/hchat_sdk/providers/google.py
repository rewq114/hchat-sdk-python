from typing import AsyncGenerator, Dict, Any, List, Optional
import httpx
import json
import uuid

from .base import BaseProvider
from ..types.request import LLMRequest, MessageRole, InputMessage
from ..types.response import (
    LLMResponse, ResponseChunk, StreamStart, StreamDelta, StreamStop,
    TextStart, TextDelta, TextEnd, ThinkingDelta,
    ToolCallStart, ToolCallDelta, ToolCallEnd, Usage, Choice
)

class GoogleProvider(BaseProvider):
    async def complete(self, request: LLMRequest) -> LLMResponse:
        url = self._get_url(request, stream=False)
        payload = self._convert_request(request)
        headers = self._get_headers(request)
        headers['Content-Type'] = 'application/json'

        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=60.0)
            response.raise_for_status()
            data = response.json()
            return self._map_complete_response(data, request)

    async def stream(self, request: LLMRequest) -> AsyncGenerator[ResponseChunk, None]:
        url = self._get_url(request, stream=True)
        payload = self._convert_request(request)
        headers = self._get_headers(request)
        headers['Content-Type'] = 'application/json'

        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, headers=headers, json=payload, timeout=60.0) as response:
                response.raise_for_status()
                
                is_first_chunk = True
                current_block_type = None # 'text', 'thinking', 'tool_call'
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    if line.startswith("data:"):
                        data_str = line[len("data:"):].strip()
                        if not data_str:
                            continue
                        
                        try:
                            raw_chunk = json.loads(data_str)
                            
                            if is_first_chunk:
                                yield StreamStart(
                                    type="stream_start",
                                    data={
                                        "model": raw_chunk.get("modelVersion", request.model),
                                        "responseId": raw_chunk.get("responseId")
                                    }
                                )
                                is_first_chunk = False

                            if "candidates" in raw_chunk:
                                for candidate in raw_chunk["candidates"]:
                                    if "content" in candidate and "parts" in candidate["content"]:
                                        for part in candidate["content"]["parts"]:
                                            # 1. Text
                                            if "text" in part and not part.get("thought"):
                                                if current_block_type != "text":
                                                    if current_block_type: yield self._create_end_event(current_block_type)
                                                    yield StreamDelta(type="stream_delta", content=TextStart(type="text_start"))
                                                    current_block_type = "text"
                                                yield StreamDelta(type="stream_delta", content=TextDelta(type="text_delta", text=part["text"]))
                                            
                                            # 2. Thinking
                                                if current_block_type != "thinking":
                                                    if current_block_type: yield self._create_end_event(current_block_type)
                                                    yield StreamDelta(type="stream_delta", content=ThinkingStart(type="thinking_start"))
                                                    current_block_type = "thinking"
                                                yield StreamDelta(type="stream_delta", content=ThinkingDelta(type="thinking_delta", thinking=part["text"]))
                                            
                                            # 3. Tool Call
                                            elif "functionCall" in part:
                                                if current_block_type: yield self._create_end_event(current_block_type)
                                                
                                                call_id = f"call_{uuid.uuid4()}"
                                                fn = part["functionCall"]
                                                yield StreamDelta(type="stream_delta", content=ToolCallStart(
                                                    type="tool_call_start",
                                                    toolCallId=call_id,
                                                    name=fn.get("name")
                                                ))
                                                args_str = json.dumps(fn.get("args", {}))
                                                yield StreamDelta(type="stream_delta", content=ToolCallDelta(
                                                    type="tool_call_delta",
                                                    args=args_str
                                                ))
                                                yield StreamDelta(type="stream_delta", content=ToolCallEnd(
                                                    type="tool_call_end",
                                                    input=fn.get("args", {})
                                                ))
                                                current_block_type = None # Reset after tool call as Gemini typically sends full call
                            
                            # Update usage if present
                            if "usageMetadata" in raw_chunk:
                                # We might want to yield StreamStop here or wait for the end
                                pass

                        except:
                            continue
                
                if current_block_type:
                    yield self._create_end_event(current_block_type)
                
                # Final usage stop chunk would be nice, but need to capture usageMetadata from last chunk
                yield StreamStop(
                    type="stream_stop",
                    data={
                        "finishReason": "stop",
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    }
                )

    def _get_url(self, request: LLMRequest, stream: bool) -> str:
        method = 'streamGenerateContent' if stream else 'generateContent'
        base = request.api_base.rstrip('/')
        if 'models' not in base:
            base = f"{base}/models"
        
        url = f"{base}/{request.model}:{method}?key={request.api_key}"
        if stream:
            url += "&alt=sse"
        return url

    def _convert_request(self, request: LLMRequest) -> Dict[str, Any]:
        contents = self._convert_messages(request.messages)
        
        generation_config = {
            "maxOutputTokens": request.max_tokens,
            "temperature": request.temperature,
            "topP": request.top_p,
            "topK": request.top_k,
            "stopSequences": request.stop,
        }
        
        if request.reasoning:
            generation_config["thinkingConfig"] = {
                "includeThoughts": True,
                "thinkingBudget": request.reasoning_budget or 1024
            }

        payload = {
            "contents": contents,
            "generationConfig": generation_config
        }
        
        if request.system:
            payload["systemInstruction"] = { "parts": [{ "text": request.system }] }
            
        if request.tools:
            # Google expects functional declarations inside a tools list
            fn_declarations = []
            for t in request.tools:
                if t.get("type") in ["function", "custom"]:
                    if "function" in t:
                        fn = t["function"]
                        fn_declarations.append({
                            "name": fn.get("name"),
                            "description": fn.get("description"),
                            "parameters": fn.get("parameters")
                        })
                    else:
                        fn_declarations.append({
                            "name": t.get("name"),
                            "description": t.get("description"),
                            "parameters": t.get("parameters")
                        })
            if fn_declarations:
                payload["tools"] = [{"functionDeclarations": fn_declarations}]

        return payload

    def _convert_messages(self, messages: List[InputMessage]) -> List[Dict[str, Any]]:
        contents = []
        for m in messages:
            if m.role == 'system':
                continue
            
            role = "user" if m.role in [MessageRole.USER, "user"] else "model"
            parts = []
            
            if isinstance(m.content, str):
                parts.append({"text": m.content})
            elif isinstance(m.content, list):
                for block in m.content:
                    if block.type == 'text':
                        parts.append({"text": block.text})
                    elif block.type == 'image':
                        source = block.source
                        if source.type == 'base64':
                            parts.append({
                                "inlineData": {
                                    "mimeType": source.media_type or "image/jpeg",
                                    "data": source.data
                                }
                            })
                        elif source.type == 'url':
                            # Gemini handles fileUri for GCS or similar, 
                            # but for public URLs we might need inlineData or similar if backend supports it.
                            pass
                    elif block.type == 'tool_use':
                        parts.append({
                            "functionCall": {
                                "name": block.name,
                                "args": block.input
                            }
                        })
                    elif block.type == 'tool_result':
                        parts.append({
                            "functionResponse": {
                                "name": block.tool_use_id,
                                "response": { "result": block.content }
                            }
                        })
            
            if parts:
                contents.append({"role": role, "parts": parts})
        return contents

    def _map_complete_response(self, data: Dict[str, Any], request: LLMRequest) -> LLMResponse:
        content_blocks = []
        candidates = data.get('candidates', [])
        finish_reason = "stop"
        
        if candidates:
            candidate = candidates[0]
            finish_reason = candidate.get('finishReason', 'stop')
            parts = candidate.get('content', {}).get('parts', [])
            for p in parts:
                if "text" in p and not p.get("thought"):
                    content_blocks.append({"type": "text", "text": p["text"]})
                elif p.get("thought"):
                    content_blocks.append({"type": "thinking", "thinking": p["text"]})
                elif "functionCall" in p:
                    fn = p["functionCall"]
                    content_blocks.append({
                        "type": "tool_use",
                        "id": f"call_{uuid.uuid4()}",
                        "name": fn.get("name"),
                        "input": fn.get("args")
                    })
        
        usage_md = data.get('usageMetadata', {})
        usage = Usage(
            prompt_tokens=usage_md.get('promptTokenCount', 0),
            completion_tokens=usage_md.get('candidatesTokenCount', 0),
            total_tokens=usage_md.get('totalTokenCount', 0),
            reasoning_tokens=usage_md.get('thoughtsTokenCount', 0)
        )

        return LLMResponse(
            id=data.get('responseId', 'unknown'),
            model=data.get('modelVersion', request.model),
            created=0,
            usage=usage,
            choices=[Choice(
                index=0,
                message=InputMessage(
                    role=MessageRole.ASSISTANT,
                    content=content_blocks
                ),
                finish_reason=finish_reason
            )]
        )

    def _create_end_event(self, block_type: str) -> StreamDelta:
        if block_type == "text":
            return StreamDelta(type="stream_delta", content=TextEnd(type="text_end"))
        elif block_type == "thinking":
            return StreamDelta(type="stream_delta", content=ThinkingEnd(type="thinking_end"))
        return StreamDelta(type="stream_delta", content=TextEnd(type="text_end"))
