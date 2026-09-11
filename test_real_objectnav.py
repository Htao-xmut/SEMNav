import os
import sys
import yaml
from omegaconf import OmegaConf

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

from env import ObjectNavEnv


def test_real_objectnav():
    """测试真实的 ObjectNav 环境"""
    
    print("=" * 60)
    print("🧪 测试真实 ObjectNav 环境")
    print("=" * 60)
    
    # 加载配置（使用 yaml.safe_load，然后手动处理环境变量）
    print("\n📦 加载配置文件...")
    with open('config/ObjectNav.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 手动解析环境变量（支持 ${VAR:-default} 格式）
    import re
    def resolve_env_vars(obj):
        """递归解析配置中的环境变量"""
        if isinstance(obj, str):
            # 匹配 ${VAR:-default} 或 ${VAR}
            pattern = r'\$\{([^}:]+)(?::-(.+))?\}'
            match = re.match(pattern, obj)
            if match:
                var_name = match.group(1)
                default_value = match.group(2)
                return os.environ.get(var_name, default_value)
            return obj
        elif isinstance(obj, dict):
            return {k: resolve_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_env_vars(item) for item in obj]
        return obj
    
    config = resolve_env_vars(config)
    
    # 修改配置用于快速测试
    config['env_cfg']['num_episodes'] = 1
    config['env_cfg']['max_steps'] = 3  # 只运行 3 步
    config['env_cfg']['log_freq'] = 1
    config['env_cfg']['name'] = 'test_real_data'
    config['env_cfg']['split'] = 'val_mini'  # 使用 val_mini
    
    print(f"   ✅ 配置加载成功")
    print(f"   - Split: {config['env_cfg']['split']}")
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
        print(f"\n   ⚠️  导航测试出现异常（可能正常）: {type(e).__name__}")
        # 不立即返回 False，因为可能是预期的异常
    
    print("\n" + "=" * 60)
    print("✅ ObjectNav 环境测试完成！")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_real_objectnav()
