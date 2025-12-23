from typing import List, Dict, Optional
from pydantic import BaseModel

class ModelCapability(BaseModel):
    model: str
    provider: str
    max_tokens: int

# Simple registry based on the Node SDK
MODEL_CAPABILITIES = [
    # OpenAI (Mapped to azure provider for HChat deployment logic)
    # ModelCapability(model='gpt-5', provider='azure', max_tokens=16384),
    ModelCapability(model='gpt-5-mini', provider='azure', max_tokens=16384),
    ModelCapability(model='gpt-4o', provider='azure', max_tokens=4096),
    ModelCapability(model='gpt-4o-mini', provider='azure', max_tokens=16384),
    ModelCapability(model='gpt-4.1', provider='azure', max_tokens=16384),
    ModelCapability(model='gpt-4.1-mini', provider='azure', max_tokens=16384),
    
    # Anthropic
    ModelCapability(model='claude-sonnet-4', provider='anthropic', max_tokens=8192),
    ModelCapability(model='claude-sonnet-4-5', provider='anthropic', max_tokens=8192),
    ModelCapability(model='claude-haiku-4-5', provider='anthropic', max_tokens=4096),
    ModelCapability(model='claude-3-7-sonnet', provider='anthropic', max_tokens=8192),
    ModelCapability(model='claude-3-5-sonnet-v2', provider='anthropic', max_tokens=8192),
    
    # Google
    ModelCapability(model='gemini-2.5-pro', provider='google', max_tokens=8192),
    ModelCapability(model='gemini-2.5-flash', provider='google', max_tokens=8192),
    ModelCapability(model='gemini-2.5-flash-image', provider='google', max_tokens=4096),
    ModelCapability(model='gemini-2.0-flash', provider='google', max_tokens=8192),

    # HChat (Provider: hchat)
    ModelCapability(model='gpt-5-mini', provider='hchat', max_tokens=16384),
    ModelCapability(model='gpt-4.1', provider='hchat', max_tokens=16384),
    ModelCapability(model='gpt-4.1-mini', provider='hchat', max_tokens=16384),
    ModelCapability(model='gpt-4o', provider='hchat', max_tokens=4096),
    ModelCapability(model='gpt-4o-mini', provider='hchat', max_tokens=16384),
    ModelCapability(model='claude-sonnet-4-5', provider='hchat', max_tokens=8192),
    ModelCapability(model='claude-haiku-4-5', provider='hchat', max_tokens=4096),
    ModelCapability(model='claude-sonnet-4', provider='hchat', max_tokens=8192),
    ModelCapability(model='claude-3-7-sonnet', provider='hchat', max_tokens=8192),
    ModelCapability(model='claude-3-5-sonnet-v2', provider='hchat', max_tokens=8192),
    ModelCapability(model='gemini-2.5-pro', provider='hchat', max_tokens=8192),
    ModelCapability(model='gemini-2.5-flash', provider='hchat', max_tokens=8192),
    ModelCapability(model='gemini-2.5-flash-image', provider='hchat', max_tokens=4096),
    ModelCapability(model='gemini-2.0-flash', provider='hchat', max_tokens=8192),
]


def get_provider_for_model(model: str) -> str:
    """Find the provider for a given model name."""
    for cap in MODEL_CAPABILITIES:
        if cap.model == model:
            return cap.provider
            
    # Fallback heuristics REMOVED for strict validation matching Node SDK
    # if model.startswith('gpt'):
    #     return 'openai'
    # if model.startswith('claude'):
    #     return 'anthropic'
    # if model.startswith('gemini'):
    #     return 'google'
        
    raise ValueError(f"Unsupported model: {model}. Please check HChat Guide for supported models.")
