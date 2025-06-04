#!/usr/bin/env python3
"""
Standalone AskNews API test script
No dependencies on forecasting_tools package - can be run independently

Usage:
1. Set environment variables:
   export ASKNEWS_CLIENT_ID="your_client_id"
   export ASKNEWS_SECRET="your_secret"

2. Install required package:
   pip install asknews

3. Run:
   python3 standalone_asknews_test.py
"""

import asyncio
import json
import os
import sys

async def test_asknews_api():
    """Test AskNews API and capture response structure"""

    print("🔍 Testing AskNews API...")

    # Check for API keys
    client_id = "5649cfb5-b99c-4559-8cb6-1d957557362e"
    client_secret = "NASL44tDAIospAxYS3BwGAF.iY"

    if not client_id or not client_secret:
        print("❌ Missing API keys!")
        print("Please set environment variables:")
        print("  export ASKNEWS_CLIENT_ID='your_client_id'")
        print("  export ASKNEWS_SECRET='your_secret'")
        return None

    try:
        # Import AskNews SDK
        try:
            from asknews_sdk import AsyncAskNewsSDK
            print("✅ AskNews SDK imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import AskNews SDK: {e}")
            print("Install with: pip install asknews")
            return None

        # Make API call
        async with AsyncAskNewsSDK(
            client_id=client_id,
            client_secret=client_secret,
            scopes={"news"},
        ) as ask:

            print("🔄 Making API call...")
            response = await ask.news.search_news(
                query="AI developments 2025",
                n_articles=3,
                return_type="both",
                strategy="latest news",
            )

            print("✅ API call successful!")

            # Get articles
            articles = response.as_dicts
            print(f"📊 Found {len(articles)} articles")

            if not articles:
                print("⚠️ No articles returned")
                return None

            # Examine first article structure
            article = articles[0]

            print("\n" + "="*60)
            print("📋 ASKNEWS ARTICLE STRUCTURE")
            print("="*60)

            # Show full JSON structure
            print("\n🔍 Full Article JSON:")
            print(json.dumps(article, indent=2, default=str))

            # Extract key fields
            print("\n🎯 Key Fields for ResearchSnippet:")
            fields = {
                "title": getattr(article, 'eng_title', None),
                "snippet": getattr(article, 'summary', None),
                "url": getattr(article, 'article_url', None),
                "date": getattr(article, 'pub_date', None),
                "source": getattr(article, 'source_id', None),
            }

            for field, value in fields.items():
                if value is not None:
                    if field == "snippet" and len(str(value)) > 100:
                        print(f"  {field}: {str(value)[:100]}...")
                    else:
                        print(f"  {field}: {value}")
                else:
                    print(f"  {field}: None")

            print("\n🔧 Field Types:")
            for field, value in fields.items():
                print(f"  {field}: {type(value).__name__}")

            # Show all available attributes
            print("\n📝 All Article Attributes:")
            for attr in dir(article):
                if not attr.startswith('_'):
                    try:
                        value = getattr(article, attr)
                        if not callable(value):
                            print(f"  {attr}: {type(value).__name__}")
                    except:
                        pass

            return {
                "articles": articles,
                "sample_fields": fields,
                "success": True
            }

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function"""
    print("🚀 AskNews API Structure Test")
    print("="*40)

    # Run the test
    result = asyncio.run(test_asknews_api())

    if result and result.get("success"):
        print("\n✅ Test completed successfully!")
        print("\n📋 Summary:")
        print("  - API call worked")
        print(f"  - Found {len(result['articles'])} articles")
        print("  - Captured field structures")

        # Save results
        with open("asknews_response.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        print("  - Saved full response to asknews_response.json")
    else:
        print("\n❌ Test failed - see errors above")
        sys.exit(1)

if __name__ == "__main__":
    main()