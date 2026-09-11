#!/usr/bin/env python3
"""
测试 Qwen-VL API 是否正常工作
"""

import sys
import os
sys.path.insert(0, '/home/tao_h/VLMnav/src')

from vlm import QwenVLClient
import numpy as np

def test_qwen_api():
    # 从环境变量读取 API Key
    api_key = os.getenv('QWEN_API_KEY')
    
    if not api_key:
        print("❌ 错误: 未设置 QWEN_API_KEY 环境变量")
        print("请运行: export QWEN_API_KEY='your_api_key'")
        return False
    
    print(f"🧪 测试 Qwen-VL API")
    print(f"📡 API Key: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        # 初始化客户端
        client = QwenVLClient(
            model="qwen-vl-max",
            api_key=api_key,
            max_image_res=1024,
            system_instruction="You are a helpful assistant."
        )
        
        # 创建测试图片（简单的彩色方块）
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[100:200, 100:200] = [255, 0, 0]  # 红色方块
        
        # 发送测试请求
        prompt = "What color is the square in this image? Respond in JSON: {'color': '...'}"
        
        print("📤 发送请求到 Qwen-VL...")
        response = client.call_chat(
            history=0,
            images=[test_image],
            text_prompt=prompt
        )
        
        if response:
            print(f"✅ 成功！响应长度: {len(response)}")
            print(f"📝 响应内容: {response}")
            return True
        else:
            print("❌ 失败: 收到空响应")
            return False
            
    except Exception as e:
        print(f"❌ 异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_qwen_api()
    sys.exit(0 if success else 1)
