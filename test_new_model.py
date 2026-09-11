#!/usr/bin/env python3
"""
快速测试新的 VLM 模型
"""

import sys
sys.path.insert(0, '/home/tao_h/VLMnav/src')

from vlm import OllamaVLM
import numpy as np
from PIL import Image

def test_new_model():
    # 配置新模型
    model_name = "llava:13b"  # 改成你想测试的模型
    
    print(f"🧪 测试模型: {model_name}")
    
    try:
        # 初始化 VLM
        vlm = OllamaVLM(
            model=model_name,
            max_image_res=512,
            system_instruction=None
        )
        
        # 创建一张测试图片（全黑）
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 发送测试请求
        prompt = "What do you see in this image? Respond in JSON format: {'description': '...'}"
        
        print("📤 发送请求...")
        response = vlm.call_chat(
            history=0,
            images=[test_image],
            text_prompt=prompt
        )
        
        print(f"✅ 成功！响应长度: {len(response)}")
        print(f"📝 响应预览: {response[:200]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_new_model()
    sys.exit(0 if success else 1)
