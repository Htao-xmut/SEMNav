import os
import sys
import gzip
import json
import yaml

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()


def test_objectnav_dataset():
    """测试 ObjectNav 数据集加载"""
    
    print("=" * 60)
    print("🧪 测试 ObjectNav 数据集")
    print("=" * 60)
    
    # 检查数据集路径
    dataset_path = "/home/tao_h/VLMnav/data/datasets/objectnav_hm3d_v2"
    scene_path = "/home/tao_h/VLMnav/data/scene_datasets/hm3d/minival"
    
    print(f"\n📂 数据集路径: {dataset_path}")
    print(f"📂 场景路径: {scene_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集不存在: {dataset_path}")
        return False
    
    if not os.path.exists(scene_path):
        print(f"❌ 场景数据不存在: {scene_path}")
        return False
    
    print(f"✅ 数据集和场景目录存在")
    
    # 加载 val_mini 配置（主配置文件可能为空，实际数据在 content 目录）
    val_mini_config = os.path.join(dataset_path, "val_mini", "val_mini.json.gz")
    print(f"\n📄 检查配置文件: {val_mini_config}")
    
    with gzip.open(val_mini_config, 'rt') as f:
        config = json.load(f)
    
    # 从 content 目录加载所有 episode
    content_dir = os.path.join(dataset_path, "val_mini", "content")
    all_episodes = []
    
    if os.path.exists(content_dir):
        for scene_file in os.listdir(content_dir):
            if scene_file.endswith('.json.gz'):
                scene_path_full = os.path.join(content_dir, scene_file)
                with gzip.open(scene_path_full, 'rt') as f:
                    scene_data = json.load(f)
                    episodes = scene_data.get('episodes', [])
                    all_episodes.extend(episodes)
                    print(f"   - {scene_file}: {len(episodes)} episodes")
    
    print(f"\n✅ 总 Episodes 数量: {len(all_episodes)}")
    print(f"✅ 场景数量: {len(set(ep['scene_id'] for ep in all_episodes))}")
    
    # 显示第一个 episode 的详细信息
    if all_episodes:
        first_ep = all_episodes[0]
        print(f"\n📋 第一个 Episode 详情:")
        print(f"   - Episode ID: {first_ep['episode_id']}")
        print(f"   - Scene ID: {first_ep['scene_id']}")
        print(f"   - Object Category: {first_ep['object_category']}")
        print(f"   - Start Position: {first_ep['start_position']}")
        print(f"   - Goals: {len(first_ep.get('goals', []))} objects")
        
        # 检查场景文件是否存在
        scene_id_short = first_ep['scene_id'].split('-')[1] if '-' in first_ep['scene_id'] else first_ep['scene_id']
        scene_file = os.path.join(scene_path, first_ep['scene_id'], 
                                  f"{scene_id_short}.basis.glb")
        if os.path.exists(scene_file):
            print(f"   ✅ 场景文件存在: {scene_file}")
        else:
            print(f"   ⚠️  场景文件不存在: {scene_file}")
            print(f"      尝试查找其他格式...")
            # 列出该场景目录下的所有文件
            scene_dir = os.path.join(scene_path, first_ep['scene_id'])
            if os.path.exists(scene_dir):
                files = os.listdir(scene_dir)
                print(f"      可用文件: {files}")
    
    # 统计目标物体类别分布
    categories = {}
    for ep in all_episodes:
        cat = ep['object_category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\n📊 目标物体类别分布:")
    for cat, count in sorted(categories.items()):
        print(f"   - {cat}: {count} episodes")
    
    print("\n" + "=" * 60)
    print("🎉 ObjectNav 数据集验证通过！")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_objectnav_dataset()
