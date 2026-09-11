import os
import sys
import yaml

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

from env import ObjectNavEnv


def test_single_episode():
    """测试单个导航 episode"""
    
    print("=" * 60)
    print("🧪 测试 VLMnav 完整导航流程")
    print("=" * 60)
    
    # 加载配置
    print("\n📦 加载配置文件...")
    with open('config/ObjectNav.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 修改配置用于快速测试
    config['env_cfg']['num_episodes'] = 1
    config['env_cfg']['max_steps'] = 5  # 只运行 5 步
    config['env_cfg']['log_freq'] = 1
    config['env_cfg']['name'] = 'test_quick'
    
    print(f"   ✅ 配置加载成功")
    print(f"   - Episodes: {config['env_cfg']['num_episodes']}")
    print(f"   - Max steps: {config['env_cfg']['max_steps']}")
    print(f"   - VLM Model: {config['agent_cfg']['vlm_cfg']['model_kwargs']['model']}")
    
    # 初始化环境
    print("\n🌍 初始化环境...")
    try:
        env = ObjectNavEnv(cfg=config)
        print(f"   ✅ 环境初始化成功")
        print(f"   - Task: {env.task}")
        print(f"   - Total episodes: {env.num_episodes}")
    except Exception as e:
        print(f"   ❌ 环境初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 运行实验
    print("\n🚀 开始导航测试...")
    try:
        env.run_experiment()
        print(f"\n   ✅ 导航测试完成！")
    except Exception as e:
        print(f"\n   ❌ 导航测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("🎉 完整导航流程测试通过！")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_single_episode()
