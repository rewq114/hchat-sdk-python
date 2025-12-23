from typing import AsyncGenerator, Dict, Any, List, Optional
import httpx
import json
import uuid

from .base import BaseProvider
from ..types.request import LLMRequest, MessageRole, InputMessage
from ..types.response import (
    LLMResponse, ResponseChunk, StreamStart, StreamDelta, StreamStop,
    TextStart, TextDelta, TextEnd, ThinkingStart, ThinkingDelta, ThinkingEnd,
    ToolCallStart, ToolCallDelta, ToolCallEnd, Usage, Choice
)

class AnthropicProvider(BaseProvider):
    async def complete(self, request: LLMRequest) -> LLMResponse:
        url = self._build_url(request)
        payload = self._convert_request(request, stream=False)
        headers = self._get_headers(request)
        headers['anthropic-version'] = '2023-06-01'

        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=60.0)
            response.raise_for_status()
            data = response.json()
            return self._map_complete_response(data, request)

    async def stream(self, request: LLMRequest) -> AsyncGenerator[ResponseChunk, None]:
        url = self._build_url(request)
        payload = self._convert_request(request, stream=True)
        headers = self._get_headers(request)
        headers['anthropic-version'] = '2023-06-01'

        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, headers=headers, json=payload, timeout=60.0) as response:
                response.raise_for_status()
                
                usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
                current_block_type = None
                current_tool_args = ""
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    if line.startswith("data:"):
                        data_str = line[len("data:"):].strip()
                        if not data_str:
                            continue
                        
                        try:
                            raw_chunk = json.loads(data_str)
                            event_type = raw_chunk.get("type")
                            
                            if event_type == "message_start":
                                msg = raw_chunk.get("message", {})
                                usage.prompt_tokens = msg.get("usage", {}).get("input_tokens", 0)
                                yield StreamStart(
                                    type="stream_start",
                                    data={
                                        "model": msg.get("model", request.model),
                                        "responseId": msg.get("id")
                                    }
                                )
                            
                            elif event_type == "content_block_start":
                                block = raw_chunk.get("content_block", {})
                                current_block_type = block.get("type")
                                
                                if current_block_type == "text":
                                    yield StreamDelta(type="stream_delta", content=TextStart(type="text_start"))
                                    if block.get("text"):
                                        yield StreamDelta(type="stream_delta", content=TextDelta(type="text_delta", text=block["text"]))
                                
                                elif current_block_type == "thinking":
                                    yield StreamDelta(type="stream_delta", content=ThinkingStart(type="thinking_start"))
                                    if block.get("thinking"):
                                        yield StreamDelta(type="stream_delta", content=ThinkingDelta(
                                            type="thinking_delta", 
                                            thinking=block["thinking"],
                                            signature=block.get("signature")
                                        ))
                                        
                                elif current_block_type == "tool_use":
                                    yield StreamDelta(type="stream_delta", content=ToolCallStart(
                                        type="tool_call_start",
                                        toolCallId=block.get("id"),
                                        name=block.get("name")
                                    ))
                            
                            elif event_type == "content_block_delta":
                                delta = raw_chunk.get("delta", {})
                                delta_type = delta.get("type")
                                
                                if delta_type == "text_delta":
                                    yield StreamDelta(type="stream_delta", content=TextDelta(type="text_delta", text=delta.get("text", "")))
                                
                                elif delta_type == "thinking_delta":
                                    yield StreamDelta(type="stream_delta", content=ThinkingDelta(
                                        type="thinking_delta",
                                        thinking=delta.get("thinking", ""),
                                        signature=delta.get("signature")
                                    ))
                                
                                elif delta_type == "input_json_delta":
                                    partial_json = delta.get("partial_json", "")
                                    current_tool_args += partial_json
                                    yield StreamDelta(type="stream_delta", content=ToolCallDelta(
                                        type="tool_call_delta",
                                        args=partial_json
                                    ))
                            
                            elif event_type == "content_block_stop":
                                if current_block_type == "text":
                                    yield StreamDelta(type="stream_delta", content=TextEnd(type="text_end"))
                                elif current_block_type == "thinking":
                                    yield StreamDelta(type="stream_delta", content=ThinkingEnd(type="thinking_end"))
                                elif current_block_type == "tool_use":
                                    tool_input = {}
                                    try:
                                        if current_tool_args:
                                            tool_input = json.loads(current_tool_args)
                                    except:
                                        pass
                                    yield StreamDelta(type="stream_delta", content=ToolCallEnd(
                                        type="tool_call_end",
                                        input=tool_input
                                    ))
                                    current_tool_args = ""
                                current_block_type = None
                            
                            elif event_type == "message_delta":
                                u = raw_chunk.get("usage", {})
                                usage.completion_tokens = u.get("output_tokens", 0)
                                usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
                            
                            elif event_type == "message_stop":
                                yield StreamStop(
                                    type="stream_stop",
                                    data={
                                        "finishReason": "stop",
                                        "usage": usage.model_dump()
                                    }
                                )
                                
                        except:
                            continue

    def _build_url(self, request: LLMRequest) -> str:
        return f"{request.api_base.rstrip('/')}/claude/messages"

    def _convert_request(self, request: LLMRequest, stream: bool) -> Dict[str, Any]:
        messages = self._convert_messages(request.messages)
        
        # Default max_tokens
        max_tokens = request.max_tokens or 4096

        # Thinking logic
        thinking = None
        if request.reasoning:
            budget = request.reasoning_budget or 1024
            if budget >= max_tokens:
                budget = max_tokens // 2
            thinking = {
                "type": "enabled",
                "budget_tokens": budget
            }

        payload = {
            "model": request.model,
            "messages": messages,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": request.temperature if not request.reasoning else None,
            "top_p": request.top_p if not request.reasoning else None,
            "top_k": request.top_k,
            "stop_sequences": request.stop,
            "system": request.system,
            "thinking": thinking
        }

        if request.tools:
            payload["tools"] = self._convert_tools(request.tools)
        
        # Anthropic doesn't allow both temperature and thinking
        if request.reasoning:
            payload.pop("temperature", None)
            payload.pop("top_p", None)

        return {k: v for k, v in payload.items() if v is not None}

    def _convert_messages(self, messages: List[InputMessage]) -> List[Dict[str, Any]]:
        result = []
        for m in messages:
            if m.role == 'system':
                continue
            
            content_blocks = []
            if isinstance(m.content, str):
                content_blocks.append({"type": "text", "text": m.content})
            elif isinstance(m.content, list):
                for block in m.content:
                    if block.type == 'text':
                        content_blocks.append({"type": "text", "text": block.text})
                    elif block.type == 'image':
                        source = block.source
                        if source.type == 'base64':
                            content_blocks.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": source.media_type or "image/jpeg",
                                    "data": source.data
                                }
                            })
                        elif source.type == 'url':
                            # Anthropic doesn't support URLs directly in messages, 
                            # would need to fetch it. Node SDK skips this or assumes base64.
                            pass
                    elif block.type == 'tool_use':
                        content_blocks.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input
                        })
                    elif block.type == 'tool_result':
                        # Handle recursive content if needed, but for now:
                        content_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": block.tool_use_id,
                            "content": block.content
                        })
            
            result.append({
                "role": "user" if m.role == MessageRole.USER else "assistant",
                "content": content_blocks
            })
        return result

    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        mapped = []
        for t in tools:
            if t.get("type") in ["function", "custom"]:
                # Anthropic expects: name, description, input_schema
                if "function" in t:
                    fn = t["function"]
                    mapped.append({
                        "name": fn.get("name"),
                        "description": fn.get("description"),
                        "input_schema": fn.get("parameters")
                    })
                else:
                    mapped.append({
                        "name": t.get("name"),
                        "description": t.get("description"),
                        "input_schema": t.get("parameters")
                    })
        return mapped

    def _map_complete_response(self, data: Dict[str, Any], request: LLMRequest) -> LLMResponse:
        content_blocks = []
        for block in data.get('content', []):
            if block.get('type') == 'text':
                content_blocks.append({"type": "text", "text": block.get("text", "")})
            elif block.get('type') == 'tool_use':
                content_blocks.append({
                    "type": "tool_use",
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": block.get("input")
                })
        
        usage = data.get('usage', {})
        input_tokens = usage.get('input_tokens', 0)
        output_tokens = usage.get('output_tokens', 0)

        return LLMResponse(
            id=data.get('id', 'unknown'),
            model=data.get('model', request.model),
            created=0,
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens
            ),
            choices=[Choice(
                index=0,
                message=InputMessage(
                    role=MessageRole.ASSISTANT,
                    content=content_blocks
                ),
                finish_reason=data.get('stop_reason', 'stop')
            )]
        )
