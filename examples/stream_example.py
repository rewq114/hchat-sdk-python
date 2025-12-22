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

    client = HChat(model="gpt-4o", api_key=api_key)
    print(f"--- Streaming {client.model} ---")
    
    try:
        stream = client.stream("Tell me a short story about a space cat.")
        async for chunk in stream:
            if chunk.type == "stream_delta":
                if chunk.content.type == "text_delta":
                    print(chunk.content.text, end="", flush=True)
        print("\n--- Stream Finished ---")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())
