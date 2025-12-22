from abc import ABC, abstractmethod
from typing import AsyncGenerator
import httpx

from ..types.request import LLMRequest
from ..types.response import LLMResponse, ResponseChunk

class BaseProvider(ABC):
    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        pass

    @abstractmethod
    async def stream(self, request: LLMRequest) -> AsyncGenerator[ResponseChunk, None]:
        pass

    def _get_headers(self, request: LLMRequest) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {request.api_key}",
            **(request.extra_headers or {})
        }
