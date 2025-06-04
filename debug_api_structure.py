#!/usr/bin/env python3
"""
Quick debug script to capture API response structures for Phase 3.
Run this and copy the output to provide the examples needed.
"""

import asyncio
import json
import os

async def debug_asknews():
    """Get AskNews response structure"""
    try:
        from forecasting_tools.forecast_helpers.asknews_searcher import AskNewsSearcher
        from asknews_sdk import AsyncAskNewsSDK

        searcher = AskNewsSearcher()

        async with AsyncAskNewsSDK(
            client_id=searcher.client_id,
            client_secret=searcher.client_secret,
            scopes={"news"},
        ) as ask:
            response = await ask.news.search_news(
                query="AI developments 2025",
                n_articles=2,
                return_type="both",
                strategy="latest news",
            )

            articles = response.as_dicts
            if articles:
                article = articles[0]
                print("=== AskNews Article Structure ===")
                print(json.dumps(article, indent=2, default=str))
                return article
    except Exception as e:
        print(f"AskNews error: {e}")
    return None

async def debug_perplexity():
    """Get Perplexity response with citations"""
    try:
        from litellm import acompletion

        response = await acompletion(
            model="perplexity/sonar-pro",
            messages=[{"role": "user", "content": "Latest AI trends 2025"}],
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            base_url="https://api.perplexity.ai",
        )

        print("\n=== Perplexity Response Structure ===")
        print(f"Response type: {type(response)}")
        print(f"Has model_extra: {hasattr(response, 'model_extra')}")

        if hasattr(response, 'model_extra') and response.model_extra:
            print(f"model_extra: {response.model_extra}")
            if 'citations' in response.model_extra:
                citations = response.model_extra['citations']
                print(f"Citations: {citations}")
                return {"citations": citations, "text": response.choices[0].message.content}

        print(f"Content preview: {response.choices[0].message.content[:200]}...")
        return {"text": response.choices[0].message.content}

    except Exception as e:
        print(f"Perplexity error: {e}")
    return None

async def main():
    print("Capturing API structures...")

    asknews_result = await debug_asknews()
    perplexity_result = await debug_perplexity()

    if asknews_result:
        print(f"\n✅ AskNews sample captured")
    if perplexity_result:
        print(f"✅ Perplexity sample captured")

if __name__ == "__main__":
    asyncio.run(main())