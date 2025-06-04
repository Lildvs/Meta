#!/usr/bin/env python3
"""
Standalone AskNews test to avoid circular imports
"""

import asyncio
import json
import os

async def test_asknews():
    try:
        from asknews_sdk import AsyncAskNewsSDK

        client_id = os.getenv("ASKNEWS_CLIENT_ID")
        client_secret = os.getenv("ASKNEWS_SECRET")

        if not client_id or not client_secret:
            print("❌ ASKNEWS_CLIENT_ID or ASKNEWS_SECRET not set")
            return

        async with AsyncAskNewsSDK(
            client_id=client_id,
            client_secret=client_secret,
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

                print("\n=== Key Fields ===")
                print(f"Title: {getattr(article, 'eng_title', 'N/A')}")
                print(f"Summary: {getattr(article, 'summary', 'N/A')[:100]}...")
                print(f"URL: {getattr(article, 'article_url', 'N/A')}")
                print(f"Date: {getattr(article, 'pub_date', 'N/A')}")
                print(f"Source: {getattr(article, 'source_id', 'N/A')}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_asknews())