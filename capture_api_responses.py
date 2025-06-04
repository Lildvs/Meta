#!/usr/bin/env python3
"""
Script to capture real API response examples for Phase 3 development.

This script will:
1. Make a call to AskNews API and capture the raw JSON response structure
2. Make a call to Perplexity API and capture the response with citations
3. Output structured examples showing the fields needed for ResearchSnippet objects

Run this script to get the API response examples needed for Phase 3.
"""

import asyncio
import json
import os
from typing import Any, Dict

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.forecast_helpers.asknews_searcher import AskNewsSearcher


async def capture_asknews_response() -> Dict[str, Any]:
    """Capture an AskNews response and extract relevant fields."""

    print("🔍 Capturing AskNews API response...")

    if not os.getenv("ASKNEWS_CLIENT_ID") or not os.getenv("ASKNEWS_SECRET"):
        print("❌ ASKNEWS_CLIENT_ID or ASKNEWS_SECRET not set")
        return {}

    try:
        # Create searcher instance
        searcher = AskNewsSearcher()

        # Make a test query
        query = "Will artificial intelligence replace human jobs in 2025?"

        # Get raw SDK response to examine structure
        from asknews_sdk import AsyncAskNewsSDK

        async with AsyncAskNewsSDK(
            client_id=searcher.client_id,
            client_secret=searcher.client_secret,
            scopes={"news"},
        ) as ask:
            response = await ask.news.search_news(
                query=query,
                n_articles=3,  # Small number for cleaner example
                return_type="both",
                strategy="latest news",
            )

            # Get articles as dicts to see structure
            articles = response.as_dicts

            print("✅ AskNews API call successful")
            print(f"📊 Found {len(articles)} articles")

            # Show structure of first article
            if articles:
                sample_article = articles[0]
                print("\n📋 Sample AskNews Article Structure:")
                print(json.dumps(sample_article, indent=2, default=str))

                # Extract key fields we care about
                relevant_fields = {
                    "title": getattr(sample_article, 'eng_title', None),
                    "snippet": getattr(sample_article, 'summary', None),
                    "url": getattr(sample_article, 'article_url', None),
                    "date": getattr(sample_article, 'pub_date', None),
                    "source": getattr(sample_article, 'source_id', None),
                }

                print("\n🎯 Relevant fields for ResearchSnippet:")
                print(json.dumps(relevant_fields, indent=2, default=str))

                return {
                    "raw_response": articles,
                    "sample_article": sample_article,
                    "relevant_fields": relevant_fields
                }

    except Exception as e:
        print(f"❌ AskNews API error: {e}")
        return {}

    return {}


async def capture_perplexity_response() -> Dict[str, Any]:
    """Capture a Perplexity response and examine citations structure."""

    print("\n🔍 Capturing Perplexity API response...")

    if not os.getenv("PERPLEXITY_API_KEY"):
        print("❌ PERPLEXITY_API_KEY not set")
        return {}

    try:
        # Create model with citations enabled
        model = GeneralLlm(
            model="perplexity/sonar-pro",
            populate_citations=True,
            temperature=0
        )

        query = "What are the latest developments in artificial intelligence for 2025?"

        # Make raw API call to examine response structure
        from litellm import acompletion

        response = await acompletion(
            model="perplexity/sonar-pro",
            messages=[{"role": "user", "content": query}],
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            base_url="https://api.perplexity.ai",
            extra_headers={"Content-Type": "application/json"},
        )

        print("✅ Perplexity API call successful")

        # Examine response structure
        print(f"\n📋 Perplexity Response Type: {type(response)}")

        # Check for citations in model_extra
        if hasattr(response, 'model_extra') and response.model_extra:
            print(f"🔗 Found model_extra: {response.model_extra}")

            if 'citations' in response.model_extra:
                citations = response.model_extra['citations']
                print(f"📚 Citations found: {citations}")
                print(f"📊 Citations type: {type(citations)}")

                if citations:
                    print("\n🎯 Sample Citation Structure:")
                    print(json.dumps(citations[0] if isinstance(citations, list) else citations, indent=2))

                    return {
                        "citations": citations,
                        "response_text": response.choices[0].message.content,
                        "citation_structure": citations[0] if isinstance(citations, list) and citations else None
                    }

        # If no citations in model_extra, check response structure
        print(f"📝 Response content: {response.choices[0].message.content[:200]}...")

        # Check all attributes of response
        print("\n🔍 Full response attributes:")
        for attr in dir(response):
            if not attr.startswith('_'):
                try:
                    value = getattr(response, attr)
                    if not callable(value):
                        print(f"  {attr}: {type(value)} = {str(value)[:100]}")
                except:
                    pass

        return {
            "response_text": response.choices[0].message.content,
            "full_response": response
        }

    except Exception as e:
        print(f"❌ Perplexity API error: {e}")
        import traceback
        print(traceback.format_exc())
        return {}


async def main():
    """Main function to capture both API responses."""

    print("🚀 Starting API response capture for Phase 3 development")
    print("=" * 60)

    # Capture AskNews response
    asknews_data = await capture_asknews_response()

    # Capture Perplexity response
    perplexity_data = await capture_perplexity_response()

    print("\n" + "=" * 60)
    print("📝 SUMMARY FOR PHASE 3 DEVELOPMENT")
    print("=" * 60)

    if asknews_data:
        print("\n✅ AskNews Response Structure Captured:")
        print("   Fields available for ResearchSnippet:")
        if 'relevant_fields' in asknews_data:
            for field, value in asknews_data['relevant_fields'].items():
                print(f"   - {field}: {type(value).__name__}")

    if perplexity_data:
        print("\n✅ Perplexity Response Structure Captured:")
        if 'citations' in perplexity_data:
            print(f"   - Citations found: {len(perplexity_data['citations'])} items")
            print(f"   - Citation type: {type(perplexity_data['citations'][0]) if perplexity_data['citations'] else 'N/A'}")
        else:
            print("   - No citations found in model_extra")

    print("\n🎯 Next Steps for Phase 3:")
    print("1. Use the captured structures to implement get_snippets_async()")
    print("2. Extract citation parsing logic for perplexity_pro_search wrapper")
    print("3. Update orchestrator to merge list-results and single-text results")
    print("4. Write unit test with dummy snippets through CoD summariser")

    # Save to file for reference
    output_data = {
        "asknews": asknews_data,
        "perplexity": perplexity_data,
        "timestamp": str(asyncio.get_event_loop().time())
    }

    with open("api_response_examples.json", "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\n💾 Full response data saved to: api_response_examples.json")


if __name__ == "__main__":
    asyncio.run(main())