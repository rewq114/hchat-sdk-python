from typing import List, Dict, Optional
from pydantic import BaseModel

class ModelCapability(BaseModel):
    model: str
    provider: str
    
    # We can add more fields (reasoning, tools, etc) later if needed for validation
    # For now, we just need model -> provider mapping

# Simple registry based on the Node SDK
MODEL_CAPABILITIES = [
    # OpenAI
    # ModelCapability(model='gpt-5', provider='openai'), # Commented out to match Node validation test
    ModelCapability(model='gpt-5-mini', provider='openai'),
    ModelCapability(model='gpt-4o', provider='openai'),
    ModelCapability(model='gpt-4o-mini', provider='openai'),
    ModelCapability(model='gpt-4.1', provider='openai'),
    ModelCapability(model='gpt-4.1-mini', provider='openai'),
    
    # Anthropic
    ModelCapability(model='claude-sonnet-4', provider='anthropic'),
    ModelCapability(model='claude-sonnet-4-5', provider='anthropic'),
    ModelCapability(model='claude-haiku-4-5', provider='anthropic'),
    ModelCapability(model='claude-3-7-sonnet', provider='anthropic'),
    ModelCapability(model='claude-3-5-sonnet-v2', provider='anthropic'),
    
    # Google
    ModelCapability(model='gemini-2.5-pro', provider='google'),
    ModelCapability(model='gemini-2.5-flash', provider='google'),
    ModelCapability(model='gemini-2.5-flash-image', provider='google'),
    ModelCapability(model='gemini-2.0-flash', provider='google'),

    # HChat (Provider: hchat) - mirrors OpenAI for now usually, but strict mapping
    ModelCapability(model='gpt-5-mini', provider='hchat'),
    ModelCapability(model='gpt-4.1', provider='hchat'),
    ModelCapability(model='gpt-4.1-mini', provider='hchat'),
    ModelCapability(model='gpt-4o', provider='hchat'),
    ModelCapability(model='gpt-4o-mini', provider='hchat'),
    ModelCapability(model='claude-sonnet-4-5', provider='hchat'),
    ModelCapability(model='claude-haiku-4-5', provider='hchat'),
    ModelCapability(model='claude-sonnet-4', provider='hchat'),
    ModelCapability(model='claude-3-7-sonnet', provider='hchat'),
    ModelCapability(model='claude-3-5-sonnet-v2', provider='hchat'),
    ModelCapability(model='gemini-2.5-pro', provider='hchat'),
    ModelCapability(model='gemini-2.5-flash', provider='hchat'),
    ModelCapability(model='gemini-2.5-flash-image', provider='hchat'),
    ModelCapability(model='gemini-2.0-flash', provider='hchat'),
]


def get_provider_for_model(model: str) -> str:
    """Find the provider for a given model name."""
    # First exact match
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
