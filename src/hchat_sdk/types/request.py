from enum import Enum
from typing import List, Optional, Union, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict
from .content import ContentBlock

class MessageRole(str, Enum):
    USER = 'user'
    ASSISTANT = 'assistant'
    SYSTEM = 'system'
    TOOL = 'tool'

class InputMessage(BaseModel):
    role: MessageRole
    content: Union[str, List[ContentBlock]]

# Minimal configuration override model (subset of LLMRequest)
class HChatConfig(BaseModel):
    temperature: Optional[float] = None
    max_tokens: Optional[int] = Field(None, alias="maxTokens")
    top_p: Optional[float] = Field(None, alias="topP")
    top_k: Optional[int] = Field(None, alias="topK")
    stop: Optional[List[str]] = None
    tools: Optional[List[Dict[str, Any]]] = None # Simplified tool definition
    stream: Optional[bool] = False
    system: Optional[str] = None
    
    model_config = ConfigDict(populate_by_name=True, extra="allow")

class LLMRequest(BaseModel):
    # API Info
    api_key: str = Field(alias="apiKey")
    api_base: str = Field(alias="apiBase")

    # LLM Info
    provider: str
    model: str

    # Request Info
    messages: List[InputMessage]

    # Model Parameters
    max_tokens: Optional[int] = Field(None, alias="maxTokens")
    temperature: Optional[float] = None
    top_p: Optional[float] = Field(None, alias="topP")
    top_k: Optional[int] = Field(None, alias="topK")
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False
    system: Optional[str] = None
    reasoning: Optional[bool] = None
    reasoning_budget: Optional[int] = Field(None, alias="reasoningBudget")
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(None, alias="toolChoice")
    extra_headers: Optional[Dict[str, str]] = Field(None, alias="extraHeaders")
    response_format: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(populate_by_name=True)

