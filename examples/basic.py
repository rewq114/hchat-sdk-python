import asyncio
import os
from dotenv import load_dotenv
from hchat_sdk import HChat

load_dotenv()

async def main():
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("Please set API_KEY in .env")
        return

    # OpenAI Family
    client = HChat(model="gpt-4o", api_key=api_key)
    print(f"--- Testing {client.model} ---")
    try:
        response = await client.complete("Hello, explain quantum physics in one sentence.")
        print("Response:", response.choices[0].message.content)
    except Exception as e:
        print("Error:", e)

    # Claude Family
    client_claude = HChat(model="claude-sonnet-4-5", api_key=api_key)
    print(f"\n--- Testing {client_claude.model} ---")
    try:
        response = await client_claude.complete("Write a haiku about TypeScript.")
        print("Response:", response.choices[0].message.content)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())
