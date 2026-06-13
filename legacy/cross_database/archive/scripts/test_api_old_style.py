#!/usr/bin/env python3
"""
Test API using the legacy openai library style (consistent with previously working code).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sql_filling'))

import openai
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
            "model": os.environ.get("TACO_MODEL", "gpt-3.5-turbo"),
            "temperature": 0.1,
            "max_tokens": 8000,
        }

print("=" * 70)
print("Testing API connection - legacy openai library style")
print("=" * 70)

# Load configuration
config = load_config()
print(f"\nAPI configuration:")
print(f"  URL: {config.get('api_url')}")
print(f"  Key: {config.get('api_key')[:20]}...")
print(f"  Model: {config.get('model', 'gpt-3.5-turbo')}")

# Use legacy openai library style (consistent with previously working code)
try:
    openai.api_key = config["api_key"]
    openai.api_base = config["api_url"]
    print("\n✅ Configuration succeeded (legacy style)")
except Exception as e:
    print(f"\n❌ Configuration failed: {e}")
    sys.exit(1)

# Test a simple call
print("\nTesting API call (using openai.ChatCompletion.create)...")
try:
    response = openai.ChatCompletion.create(
        model=config.get("model", "gpt-3.5-turbo"),
        messages=[
            {"role": "system", "content": "你是一个SQL专家。"},
            {"role": "user", "content": "请回答：1+1等于几？"}
        ],
        temperature=0.1,
        max_tokens=100
    )
    
    result = response.choices[0].message.content.strip()
    print(f"✅ API call succeeded")
    print(f"Response: {result}")
    
except Exception as e:
    error_str = str(e)
    print(f"❌ API call failed")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {error_str[:200]}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("API test complete!")
print("=" * 70)
