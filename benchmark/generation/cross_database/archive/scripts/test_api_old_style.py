#!/usr/bin/env python3
"""
测试API - 使用旧版openai库的方式（与之前成功的代码一致）
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sql_filling'))

import openai
import yaml

# 加载配置
def load_config():
    config_file = os.path.join(os.path.dirname(__file__), '..', 'sql_filling', 'config.yaml')
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('llm', {})
    else:
        return {
            "api_url": "https://35.aigcbest.top/v1",
            "api_key": "sk-SeJvPPUTe9rGLtPP182bD0320779480a9705C39d25Be0215",
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "max_tokens": 8000
        }

print("=" * 70)
print("测试API连接 - 使用旧版openai库方式")
print("=" * 70)

# 加载配置
config = load_config()
print(f"\nAPI配置:")
print(f"  URL: {config.get('api_url')}")
print(f"  Key: {config.get('api_key')[:20]}...")
print(f"  Model: {config.get('model', 'gpt-3.5-turbo')}")

# 使用旧版openai库的方式（与之前成功的代码一致）
try:
    openai.api_key = config["api_key"]
    openai.api_base = config["api_url"]
    print("\n✅ 配置成功（旧版方式）")
except Exception as e:
    print(f"\n❌ 配置失败: {e}")
    sys.exit(1)

# 测试简单调用
print("\n测试API调用（使用 openai.ChatCompletion.create）...")
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
    print(f"✅ API调用成功")
    print(f"响应: {result}")
    
except Exception as e:
    error_str = str(e)
    print(f"❌ API调用失败")
    print(f"错误类型: {type(e).__name__}")
    print(f"错误信息: {error_str[:200]}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("API测试完成！")
print("=" * 70)

