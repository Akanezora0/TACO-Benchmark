#!/usr/bin/env python3
"""
Test whether the API is working correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sql_filling'))

from openai import OpenAI
import yaml

# Load configuration
def load_config():
    config_file = os.path.join(os.path.dirname(__file__), '..', 'sql_filling', 'config.yaml')
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('llm', {})
    else:
        return {
            "api_url": os.environ.get("TACO_API_URL", "https://api.openai.com/v1"),
            "api_key": os.environ.get("TACO_API_KEY", "your-api-key-here"),
            "model": os.environ.get("TACO_MODEL", "gpt-4o"),
            "temperature": 0.1,
            "max_tokens": 8000,
        }

print("=" * 70)
print("Testing API connection")
print("=" * 70)

# Load configuration
config = load_config()
print(f"\nAPI configuration:")
print(f"  URL: {config.get('api_url')}")
print(f"  Key: {config.get('api_key')[:20]}...")
print(f"  Model: {config.get('model', 'gpt-4o')}")

# Create client
try:
    api_url = config["api_url"].rstrip('/')  # Remove trailing slash; OpenAI SDK adds it automatically
    
    print(f"\nActual base_url in use: {api_url}")
    print(f"Expected full path: {api_url}/chat/completions")
    
    client = OpenAI(
        base_url=api_url,
        api_key=config["api_key"]
    )
    print("\n✅ Client created successfully")
except Exception as e:
    print(f"\n❌ Failed to create client: {e}")
    sys.exit(1)

# Test a simple call
print("\nTesting API call...")
print("Note: a quota error means the API config is correct but quota is exhausted")
print("Other errors may indicate a URL configuration problem\n")

try:
    response = client.chat.completions.create(
        model=config.get("model", "gpt-4o"),
        messages=[
            {"role": "system", "content": "你是一个SQL专家。"},
            {"role": "user", "content": "请回答：1+1等于几？"}
        ],
        temperature=0.1,
        max_tokens=100,
        timeout=30
    )
    
    result = response.choices[0].message.content.strip()
    print(f"✅ API call succeeded")
    print(f"Response: {result}")
    
except Exception as e:
    error_str = str(e)
    print(f"❌ API call failed")
    print(f"Error type: {type(e).__name__}")
    
    # Check error type
    if "quota" in error_str.lower() or "429" in error_str or "insufficient_quota" in error_str:
        print("\n⚠️  This is a quota error, which means the API URL configuration is correct!")
        print("   Issue: API quota exhausted; wait for quota reset or use a different API key")
    elif "404" in error_str or "not found" in error_str.lower():
        print("\n⚠️  This is a URL path error; the API endpoint may be misconfigured")
    elif "401" in error_str or "unauthorized" in error_str.lower():
        print("\n⚠️  This is an authentication error; the API key may be incorrect")
    else:
        print(f"\nFull error details:")
        import traceback
        traceback.print_exc()
    
    sys.exit(1)

print("\n" + "=" * 70)
print("API test complete!")
print("=" * 70)
