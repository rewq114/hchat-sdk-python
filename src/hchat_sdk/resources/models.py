from typing import List, Optional
from pydantic import BaseModel

from ..capabilities import MODEL_CAPABILITIES

class Model(BaseModel):
    model: str
    name: str
    maxToken: int

class Models:
    def __init__(self, api_key: str, api_base: str):
        self.api_key = api_key
        self.api_base = api_base

    async def list(self) -> List[Model]:
        """List available models based on capabilities."""
        models: List[Model] = []
        for cap in MODEL_CAPABILITIES:
            # Using model ID as name for now, similar to Node SDK
            models.append(Model(
                model=cap.model,
                name=cap.model,
                maxToken=cap.max_tokens
            ))
        return models

    async def retrieve(self, model_id: str) -> Model:
        """Retrieve a specific model by ID."""
        for cap in MODEL_CAPABILITIES:
            if cap.model == model_id:
                return Model(
                    model=cap.model,
                    name=cap.model,
                    maxToken=cap.max_tokens
                )
        raise ValueError(f"Model not found: {model_id}")
