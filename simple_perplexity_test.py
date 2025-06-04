#!/usr/bin/env python3
"""
Standalone Perplexity test to avoid circular imports
"""

import asyncio
import json
import os

async def test_perplexity():
    try:
        from litellm import acompletion

        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            print("❌ PERPLEXITY_API_KEY not set")
            return

        response = await acompletion(
            model="perplexity/sonar-pro",
            messages=[{"role": "user", "content": "Latest AI trends 2025"}],
            api_key=api_key,
            base_url="https://api.perplexity.ai",
        )

        print("=== Perplexity Response Structure ===")
        print(f"Response type: {type(response)}")
        print(f"Has model_extra: {hasattr(response, 'model_extra')}")

        if hasattr(response, 'model_extra') and response.model_extra:
            print(f"model_extra keys: {list(response.model_extra.keys())}")
            print(f"model_extra content: {response.model_extra}")

            if 'citations' in response.model_extra:
                citations = response.model_extra['citations']
                print(f"\n=== Citations Structure ===")
                print(f"Citations type: {type(citations)}")
                print(f"Citations content: {json.dumps(citations, indent=2)}")

        print(f"\nResponse content (first 200 chars): {response.choices[0].message.content[:200]}...")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_perplexity())