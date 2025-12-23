import json
import uuid
from typing import AsyncGenerator, Dict, Any, List, Optional, Union

import httpx

from .base import BaseProvider
from ..types.request import LLMRequest, ContentBlock, InputMessage, MessageRole
from ..types.response import (
    LLMResponse, ResponseChunk, StreamDelta, StreamStart, StreamStop,
    TextStart, TextDelta, TextEnd, ToolCallStart, ToolCallDelta, ToolCallEnd,
    Choice, Usage, ThinkingDelta
)


class AzureProvider(BaseProvider):
    """
    Azure Provider (HChat wrapped OpenAI)
    - Supports deployments/{model}/chat/completions endpoint
    - Stateful stream parsing for tool calls and text
    - Maps max_tokens to max_completion_tokens
    """

    async def complete(self, request: LLMRequest) -> LLMResponse:
        url = self._build_url(request)
        payload = self._convert_request(request, stream=False)
        headers = self._get_headers(request)
        headers["api-key"] = request.api_key

        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=60.0)
            if not response.is_success:
                 # Replicate server error handling logic if needed, but for now raise
                 response.raise_for_status()
            
            data = response.json()
            return self._map_complete_response(data)

    async def stream(self, request: LLMRequest) -> AsyncGenerator[ResponseChunk, None]:
        url = self._build_url(request)
        print(f"DEBUG AZURE URL: {url}")
        payload = self._convert_request(request, stream=True)
        headers = self._get_headers(request)
        headers["api-key"] = request.api_key
        headers["Accept"] = "text/event-stream"

        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, headers=headers, json=payload, timeout=60.0) as response:
                print(f"DEBUG RESPONSE STATUS: {response.status_code}")
                if not response.is_success:
                    err_body = await response.aread()
                    print(f"\n[Azure Stream Error Body] {err_body.decode()}")
                response.raise_for_status()

                is_first_chunk = True
                current_block_type = None  # 'text' or 'tool_call'
                current_tool_index = -1
                current_tool_args_buffer = ""
                current_tool_id = ""
                current_tool_name = ""
                final_usage = None
                final_finish_reason = "unknown"

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    if line.startswith("data:"):
                        data_str = line[len("data:"):].strip()
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            raw_chunk = json.loads(data_str)
                            
                            choices = raw_chunk.get("choices", [])
                            if is_first_chunk and choices:
                                choice = choices[0]
                                delta = choice.get("delta", {})
                                if delta.get("role"):
                                    yield StreamStart(
                                        type="stream_start",
                                        data={
                                            "model": raw_chunk.get("model", request.model),
                                            "responseId": raw_chunk.get("id")
                                        }
                                    )
                                    is_first_chunk = False

                            if not choices:
                                if "usage" in raw_chunk:
                                    u = raw_chunk["usage"]
                                    details = u.get("completion_tokens_details", {})
                                    final_usage = Usage(
                                        prompt_tokens=u.get("prompt_tokens", 0),
                                        completion_tokens=u.get("completion_tokens", 0),
                                        total_tokens=u.get("total_tokens", 0),
                                        reasoning_tokens=details.get("reasoning_tokens", 0)
                                    )
                                continue

                            choice = choices[0]
                            delta = choice.get("delta", {})
                            
                            # 1. Text Content
                            if "content" in delta and delta["content"]:
                                content = delta["content"]
                                if current_block_type != "text":
                                    if current_block_type == "tool_call":
                                        yield self._create_tool_end_event(current_tool_args_buffer)
                                    
                                    yield StreamDelta(type="stream_delta", content=TextStart(type="text_start"))
                                    current_block_type = "text"
                                
                                yield StreamDelta(type="stream_delta", content=TextDelta(type="text_delta", text=content))

                            # 2. Tool Calls
                            if "tool_calls" in delta:
                                for tc in delta["tool_calls"]:
                                    index = tc.get("index", 0)
                                    
                                    if current_block_type != "tool_call" or index != current_tool_index:
                                        if current_block_type == "tool_call":
                                            yield self._create_tool_end_event(current_tool_args_buffer)
                                        
                                        current_block_type = "tool_call"
                                        current_tool_index = index
                                        current_tool_args_buffer = ""
                                        current_tool_id = tc.get("id") or f"call_{uuid.uuid4()}"
                                        current_tool_name = tc.get("function", {}).get("name", "")
                                        
                                        yield StreamDelta(
                                            type="stream_delta",
                                            content=ToolCallStart(
                                                type="tool_call_start",
                                                toolCallId=current_tool_id,
                                                name=current_tool_name
                                            )
                                        )
                                    else:
                                        if "id" in tc and not current_tool_id:
                                            current_tool_id = tc["id"]
                                        if "function" in tc and "name" in tc["function"] and not current_tool_name:
                                            current_tool_name = tc["function"]["name"]

                                    if "function" in tc and "arguments" in tc["function"]:
                                        args_delta = tc["function"]["arguments"]
                                        current_tool_args_buffer += args_delta
                                        yield StreamDelta(
                                            type="stream_delta",
                                            content=ToolCallDelta(type="tool_call_delta", args=args_delta)
                                        )

                            # 3. Reasoning (Thinking)
                            # Handle reasoning_content if present (O1 models)
                            reasoning = delta.get("reasoning_content")
                            if reasoning:
                                yield StreamDelta(type="stream_delta", content=ThinkingDelta(type="thinking_delta", thinking=reasoning))

                            if choice.get("finish_reason"):
                                final_finish_reason = choice["finish_reason"]

                        except json.JSONDecodeError:
                            continue

                # Cleanup
                if current_block_type == "text":
                    yield StreamDelta(type="stream_delta", content=TextEnd(type="text_end"))
                elif current_block_type == "tool_call":
                    yield self._create_tool_end_event(current_tool_args_buffer)

                yield StreamStop(
                    type="stream_stop",
                    data={
                        "finishReason": final_finish_reason,
                        "usage": final_usage.model_dump() if final_usage else {"promptTokens": 0, "completionTokens": 0, "totalTokens": 0}
                    }
                )

    def _build_url(self, request: LLMRequest) -> str:
        api_base = request.api_base.rstrip("/") + "/"
        if "openai" not in api_base.lower():
            api_base += "openai/"
        return f"{api_base}deployments/{request.model}/chat/completions"

    def _create_tool_end_event(self, args_buffer: str) -> StreamDelta:
        input_data = {}
        try:
            if args_buffer:
                input_data = json.loads(args_buffer)
        except Exception:
            pass
        return StreamDelta(type="stream_delta", content=ToolCallEnd(type="tool_call_end", input=input_data))

    def _convert_request(self, request: LLMRequest, stream: bool) -> Dict[str, Any]:
        messages = self._convert_messages(request.messages)
        
        # System message handling (unshift)
        if request.system:
            messages.insert(0, {"role": "system", "content": request.system})

        is_o1 = request.model.startswith("o1")
        is_gpt5 = request.model.startswith("gpt-5")
        
        temperature = request.temperature
        if is_o1 or is_gpt5:
            temperature = 1.0  # Azure SDK logic parity
            
        payload = {
            "model": request.model,
            "messages": messages,
            "stream": stream,
            "max_completion_tokens": request.max_tokens,
            "temperature": temperature,
            "top_p": request.top_p,
            "stop": request.stop,
        }

        # Tools
        if request.tools:
            mapped_tools = []
            for t in request.tools:
                # Expecting standard OpenAI format or simplified
                if t.get("type") in ["function", "custom"]:
                    if "function" in t:
                        mapped_tools.append({
                            "type": "function",
                            "function": t["function"]
                        })
                    else:
                        mapped_tools.append({
                            "type": "function",
                            "function": {
                                "name": t.get("name"),
                                "description": t.get("description"),
                                "parameters": t.get("parameters"),
                                "strict": False
                            }
                        })
            if mapped_tools:
                payload["tools"] = mapped_tools

        # Reasoning effort
        if request.reasoning:
            payload["reasoning_effort"] = "high"
        else:
            payload["reasoning_effort"] = "minimal"

        payload = {k: v for k, v in payload.items() if v is not None}
        return payload

    def _convert_messages(self, messages: List[InputMessage]) -> List[Dict[str, Any]]:
        result = []
        for msg in messages:
            content = msg.content
            if isinstance(content, list):
                # Multimodal content
                parts = []
                for part in content:
                    p_dict = part.model_dump() if hasattr(part, 'model_dump') else part
                    if p_dict.get("type") == "text":
                        parts.append({"type": "text", "text": p_dict["text"]})
                    elif p_dict.get("type") == "image":
                        source = p_dict.get("source", {})
                        if source.get("type") == "base64":
                            url = f"data:{source.get('media_type', 'image/jpeg')};base64,{source['data']}"
                            parts.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
                        elif source.get("type") == "url":
                            parts.append({"type": "image_url", "image_url": {"url": source["url"], "detail": "high"}})
                result.append({"role": msg.role, "content": parts})
            else:
                result.append({"role": msg.role, "content": content})
        return result

    def _map_complete_response(self, data: Dict[str, Any]) -> LLMResponse:
        usage_data = data.get("usage", {})
        details = usage_data.get("completion_tokens_details", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
            reasoning_tokens=details.get("reasoning_tokens", 0)
        )

        choices = []
        for i, c in enumerate(data.get("choices", [])):
            msg_data = c.get("message", {})
            content = msg_data.get("content") or ""
            
            # If tool calls, merge into blocks
            tool_calls = msg_data.get("tool_calls")
            if tool_calls:
                blocks = []
                if content:
                    blocks.append(ContentBlock(type="text", text=content))
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except:
                        args = {}
                    blocks.append(ContentBlock(
                        type="tool_use",
                        id=tc.get("id"),
                        name=fn.get("name"),
                        input=args
                    ))
                content = blocks

            choices.append(Choice(
                index=c.get("index", i),
                message=InputMessage(role=MessageRole.ASSISTANT, content=content),
                finishReason=c.get("finish_reason")
            ))

        return LLMResponse(
            id=data.get("id", ""),
            model=data.get("model", ""),
            created=data.get("created", 0),
            usage=usage,
            choices=choices
        )
