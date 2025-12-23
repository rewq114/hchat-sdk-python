import pytest
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

from hchat_sdk import HChat

api_key = os.getenv("API_KEY") or os.getenv("HCHAT_API_KEY") or "test-key"

@pytest.mark.asyncio
async def test_models_list():
    client = HChat(api_key=api_key)
    models = await client.models.list()
    assert len(models) > 0
    print(f"\n[List] Found {len(models)} models")
    print(models[0])

@pytest.mark.asyncio
async def test_models_retrieve():
    client = HChat(api_key=api_key)
    model_id = "gpt-4o"
    model = await client.models.retrieve(model_id)
    assert model.model == model_id
    assert model.maxToken > 0
    print(f"\n[Retrieve] {model}")
