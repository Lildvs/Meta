#!/usr/bin/env python3
"""
Standalone Perplexity API test script
No dependencies on forecasting_tools package - can be run independently

Usage:
1. Set environment variable:
   export PERPLEXITY_API_KEY="your_api_key"

2. Install required package:
   pip install litellm

3. Run:
   python3 standalone_perplexity_test.py
"""

import asyncio
import json
import os
import sys

async def test_perplexity_api():
    """Test Perplexity API and capture response structure with citations"""

    print("🔍 Testing Perplexity API...")

    # Check for API key
    api_key = "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"

    if not api_key:
        print("❌ Missing API key!")
        print("Please set environment variable:")
        print("  export PERPLEXITY_API_KEY='your_api_key'")
        return None

    try:
        # Import litellm
        try:
            from litellm import acompletion
            print("✅ LiteLLM imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import litellm: {e}")
            print("Install with: pip install litellm")
            return None

        print("🔄 Making API call...")

        # Make API call
        response = await acompletion(
            model="perplexity/sonar-pro",
            messages=[{"role": "user", "content": "What are the latest AI developments in 2025?"}],
            api_key=api_key,
            base_url="https://api.perplexity.ai",
            extra_headers={"Content-Type": "application/json"},
        )

        print("✅ API call successful!")

        print("\n" + "="*60)
        print("📋 PERPLEXITY RESPONSE STRUCTURE")
        print("="*60)

        # Basic response info
        print(f"\n🔍 Response Type: {type(response)}")
        print(f"📝 Response Content (first 200 chars): {response.choices[0].message.content[:200]}...")

        # Check for model_extra (where citations usually are)
        print(f"\n🔗 Has model_extra: {hasattr(response, 'model_extra')}")

        citations_found = False
        citations_data = None

        if hasattr(response, 'model_extra') and response.model_extra:
            print(f"📊 model_extra keys: {list(response.model_extra.keys())}")
            print(f"🔧 model_extra content: {response.model_extra}")

            if 'citations' in response.model_extra:
                citations = response.model_extra['citations']
                citations_found = True
                citations_data = citations

                print(f"\n🎯 CITATIONS STRUCTURE:")
                print(f"  Type: {type(citations)}")
                print(f"  Content: {json.dumps(citations, indent=2)}")

                if isinstance(citations, list) and citations:
                    print(f"  Number of citations: {len(citations)}")
                    print(f"  First citation type: {type(citations[0])}")
                    print(f"  First citation: {citations[0]}")

        # Check all response attributes
        print(f"\n📝 All Response Attributes:")
        for attr in dir(response):
            if not attr.startswith('_'):
                try:
                    value = getattr(response, attr)
                    if not callable(value):
                        attr_type = type(value).__name__
                        attr_preview = str(value)[:50] if len(str(value)) > 50 else str(value)
                        print(f"  {attr}: {attr_type} = {attr_preview}")
                except:
                    pass

        # Try to find citations in other places
        if not citations_found:
            print(f"\n🔍 Searching for citations in other locations...")

            # Check if citations are in the response text
            content = response.choices[0].message.content
            if '[' in content and ']' in content:
                print("  ✅ Found bracket notation in content - citations may be embedded")
            else:
                print("  ❌ No bracket notation found in content")

            # Check response object structure more deeply
            try:
                response_dict = response.model_dump() if hasattr(response, 'model_dump') else dict(response)
                print(f"  📋 Response as dict keys: {list(response_dict.keys())}")
            except:
                print("  ❌ Could not convert response to dict")

        return {
            "response_text": response.choices[0].message.content,
            "citations_found": citations_found,
            "citations_data": citations_data,
            "model_extra": response.model_extra if hasattr(response, 'model_extra') else None,
            "success": True
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function"""
    print("🚀 Perplexity API Structure Test")
    print("="*40)

    # Run the test
    result = asyncio.run(test_perplexity_api())

    if result and result.get("success"):
        print("\n✅ Test completed successfully!")
        print("\n📋 Summary:")
        print("  - API call worked")
        print(f"  - Citations found: {result['citations_found']}")
        if result['citations_found']:
            print(f"  - Citation type: {type(result['citations_data'])}")
        print("  - Captured response structures")

        # Save results
        with open("perplexity_response.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        print("  - Saved full response to perplexity_response.json")
    else:
        print("\n❌ Test failed - see errors above")
        sys.exit(1)

if __name__ == "__main__":
    main()