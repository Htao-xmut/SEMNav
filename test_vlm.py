import os
import sys
import numpy as np

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

from vlm import OllamaVLM


def test_ollama_vlm():
    """测试 Ollama LLaVA 模型的文本和图像推理能力"""
    
    print("=" * 60)
    print("🧪 测试 Ollama LLaVA VLM 接口")
    print("=" * 60)
    
    # 1. 初始化 VLM
    print("\n📦 初始化 OllamaVLM (llava:7b)...")
    try:
        vlm = OllamaVLM(
            model="llava:7b",
            system_instruction=None,  # LLaVA 不支持 system instruction
            max_image_res=512
        )
        print(f"   ✅ VLM 初始化成功: {vlm.name}")
    except Exception as e:
        print(f"   ❌ VLM 初始化失败: {e}")
        return False
    
    # 2. 测试纯文本推理
    print("\n💬 测试纯文本推理...")
    try:
        text_prompt = "What is 2+2? Answer with just the number."
        response = vlm.call(images=[], text_prompt=text_prompt)
        print(f"   Prompt: {text_prompt}")
        print(f"   Response: {response}")
        print(f"   ✅ 纯文本推理成功")
    except Exception as e:
        print(f"   ❌ 纯文本推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 测试图像推理（创建一个简单的测试图像）
    print("\n🖼️  测试图像推理...")
    try:
        # 创建一个简单的彩色图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        text_prompt = "Describe this image in one sentence."
        response = vlm.call(images=[test_image], text_prompt=text_prompt)
        print(f"   Image shape: {test_image.shape}")
        print(f"   Prompt: {text_prompt}")
        print(f"   Response: {response[:200]}..." if len(response) > 200 else f"   Response: {response}")
        print(f"   ✅ 图像推理成功")
    except Exception as e:
        print(f"   ❌ 图像推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("🎉 所有测试通过！Ollama LLaVA VLM 接口正常工作。")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_ollama_vlm()
