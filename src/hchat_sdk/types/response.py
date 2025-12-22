from typing import List, Optional, Union, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict
from .request import InputMessage

class Usage(BaseModel):
    promptTokens: int = Field(alias="prompt_tokens")
    completionTokens: int = Field(alias="completion_tokens")
    totalTokens: int = Field(alias="total_tokens")
    reasoningTokens: Optional[int] = Field(None, alias="reasoning_tokens")
    
    model_config = ConfigDict(populate_by_name=True)

class Choice(BaseModel):
    index: int
    message: InputMessage
    finishReason: str = Field(alias="finish_reason")
    
    model_config = ConfigDict(populate_by_name=True)

class LLMResponse(BaseModel):
    id: str
    model: str
    created: int
    usage: Usage
    choices: List[Choice]
    
    model_config = ConfigDict(populate_by_name=True)

# ====================
# STREAM EVENTS
# ====================

# Content Events
class TextDelta(BaseModel):
    type: Literal['text_delta']
    text: str

class TextStart(BaseModel):
    type: Literal['text_start']

class TextEnd(BaseModel):
    type: Literal['text_end']

class ThinkingDelta(BaseModel):
    type: Literal['thinking_delta']
    thinking: str
    signature: Optional[str] = None

class ToolCallStart(BaseModel):
    type: Literal['tool_call_start']
    name: str
    toolCallId: Optional[str] = None

class ToolCallDelta(BaseModel):
    type: Literal['tool_call_delta']
    args: str

class ToolCallEnd(BaseModel):
    type: Literal['tool_call_end']
    input: Dict[str, Any]

ContentEvent = Union[
    TextStart, TextDelta, TextEnd,
    ThinkingDelta,
    ToolCallStart, ToolCallDelta, ToolCallEnd
]

# Stream Chunks
class StreamStart(BaseModel):
    type: Literal['stream_start']
    data: Dict[str, Any]

class StreamDelta(BaseModel):
    type: Literal['stream_delta']
    content: ContentEvent

class StreamStop(BaseModel):
    type: Literal['stream_stop']
    data: Dict[str, Any]

class StreamError(BaseModel):
    type: Literal['error']
    data: Dict[str, Any]

ResponseChunk = Union[StreamStart, StreamDelta, StreamStop, StreamError]
