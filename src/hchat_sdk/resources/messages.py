from typing import Union, List, Optional, AsyncGenerator, Dict, Any
from ..types.request import InputMessage, LLMRequest, HChatConfig, MessageRole
from ..types.response import LLMResponse, ResponseChunk
from ..capabilities import get_provider_for_model
from ..providers.base import BaseProvider
from ..providers.openai import OpenAIProvider
from ..providers.anthropic import AnthropicProvider
from ..providers.google import GoogleProvider
from ..providers.azure import AzureProvider

class Messages:
    def __init__(self, api_key: str, api_base: str):
        self.api_key = api_key
        self.api_base = api_base
        self._providers: Dict[str, BaseProvider] = {}

    def _get_provider_instance(self, provider_name: str) -> BaseProvider:
        if provider_name in self._providers:
            return self._providers[provider_name]
            
        if provider_name == 'openai':
            instance = OpenAIProvider()
        elif provider_name == 'anthropic':
            instance = AnthropicProvider()
        elif provider_name == 'google':
            instance = GoogleProvider()
        elif provider_name == 'azure':
            instance = AzureProvider()
        elif provider_name == 'hchat':
            # Mapping hchat provider to Azure logic (deployment endpoint)
            instance = AzureProvider()
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")
            
        self._providers[provider_name] = instance
        return instance

    def _normalize_input(self, input_data: Union[str, List[InputMessage]]) -> List[InputMessage]:
        if isinstance(input_data, str):
            from ..types.request import InputMessage, MessageRole # Import here to avoid circular or just use available
            return [InputMessage(role=MessageRole.USER, content=input_data)]
        return input_data

    async def complete(self, model: str, input: Union[str, List[InputMessage]], **config) -> LLMResponse:
        messages = self._normalize_input(input)
        provider_name = get_provider_for_model(model)
        provider = self._get_provider_instance(provider_name)
        
        cfg = HChatConfig(**config)
        
        request = LLMRequest(
            api_key=self.api_key,
            api_base=self.api_base,
            provider=provider_name,
            model=model,
            messages=messages,
            stream=False,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            stop=cfg.stop,
            tools=cfg.tools,
            system=cfg.system
        )
        
        return await provider.complete(request)

    async def stream(self, model: str, input: Union[str, List[InputMessage]], **config) -> AsyncGenerator[ResponseChunk, None]:
        messages = self._normalize_input(input)
        provider_name = get_provider_for_model(model)
        provider = self._get_provider_instance(provider_name)
        
        cfg = HChatConfig(**config)
        
        request = LLMRequest(
            api_key=self.api_key,
            api_base=self.api_base,
            provider=provider_name,
            model=model,
            messages=messages,
            stream=True,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            stop=cfg.stop,
            tools=cfg.tools,
            system=cfg.system
        )
        
        async for chunk in provider.stream(request):
            yield chunk
