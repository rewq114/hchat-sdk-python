from typing import Union, List, Optional, AsyncGenerator, Dict, Any
import os

from .types.request import InputMessage
from .types.response import LLMResponse, ResponseChunk
from .resources.messages import Messages
from .resources.models import Models

class HChat:
    DEFAULT_API_BASE = 'https://h-chat-api.autoever.com/v2/api'

    def __init__(self, api_key: str, api_base: Optional[str] = None):
        self.api_key = api_key
        self.api_base = api_base or self.DEFAULT_API_BASE
        
        self.messages = Messages(self.api_key, self.api_base)
        self.models = Models(self.api_key, self.api_base)

    async def complete(self, model: str, input: Union[str, List[InputMessage]], **config) -> LLMResponse:
        """
        Deprecated: Use client.messages.complete() instead.
        """
        return await self.messages.complete(model, input, **config)

    async def stream(self, model: str, input: Union[str, List[InputMessage]], **config) -> AsyncGenerator[ResponseChunk, None]:
        """
        Deprecated: Use client.messages.stream() instead.
        """
        async for chunk in self.messages.stream(model, input, **config):
            yield chunk
