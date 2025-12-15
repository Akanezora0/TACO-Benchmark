#!/usr/bin/env python3
"""
测试API是否正常工作
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sql_filling'))

from openai import OpenAI
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
            "model": "gpt-4o",
            "temperature": 0.1,
            "max_tokens": 8000
        }

print("=" * 70)
print("测试API连接")
print("=" * 70)

# 加载配置
config = load_config()
print(f"\nAPI配置:")
print(f"  URL: {config.get('api_url')}")
print(f"  Key: {config.get('api_key')[:20]}...")
print(f"  Model: {config.get('model', 'gpt-4o')}")

# 创建客户端
try:
    api_url = config["api_url"].rstrip('/')  # 移除末尾斜杠，OpenAI SDK会自动添加
    
    print(f"\n实际使用的base_url: {api_url}")
    print(f"预期完整路径: {api_url}/chat/completions")
    
    client = OpenAI(
        base_url=api_url,
        api_key=config["api_key"]
    )
    print("\n✅ 客户端创建成功")
except Exception as e:
    print(f"\n❌ 客户端创建失败: {e}")
    sys.exit(1)

# 测试简单调用
print("\n测试API调用...")
print("注意：如果看到配额错误，说明API配置正确，只是配额用完了")
print("如果看到其他错误，可能是URL配置问题\n")

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
    print(f"✅ API调用成功")
    print(f"响应: {result}")
    
except Exception as e:
    error_str = str(e)
    print(f"❌ API调用失败")
    print(f"错误类型: {type(e).__name__}")
    
    # 检查错误类型
    if "quota" in error_str.lower() or "429" in error_str or "insufficient_quota" in error_str:
        print("\n⚠️  这是配额错误，说明API URL配置是正确的！")
        print("   问题：API配额已用完，需要等待配额恢复或更换API密钥")
    elif "404" in error_str or "not found" in error_str.lower():
        print("\n⚠️  这是URL路径错误，可能是API端点配置不正确")
    elif "401" in error_str or "unauthorized" in error_str.lower():
        print("\n⚠️  这是认证错误，可能是API密钥不正确")
    else:
        print(f"\n完整错误信息:")
        import traceback
        traceback.print_exc()
    
    sys.exit(1)

print("\n" + "=" * 70)
print("API测试完成！")
print("=" * 70)

