#!/usr/bin/env python3
"""
查找并运行寻找 sofa 的测试
"""

import gzip
import json
from pathlib import Path

def find_sofa_episodes():
    """查找所有目标为 sofa 的 episodes"""
    dataset_dir = Path('data/datasets/objectnav_hm3d_v2/val_mini/content')
    
    if not dataset_dir.exists():
        print(f"❌ 数据集目录不存在: {dataset_dir}")
        return []
    
    sofa_episodes = []
    
    for scene_file in sorted(dataset_dir.glob('*.json.gz')):
        with gzip.open(scene_file, 'rt') as f:
            data = json.load(f)
            for i, episode in enumerate(data.get('episodes', [])):
                if episode['object_category'] == 'sofa':
                    sofa_episodes.append({
                        'scene_file': scene_file.name,
                        'episode_index': i,
                        'episode_id': episode['episode_id'],
                        'scene_id': episode['scene_id'],
                        'geodesic_distance': episode['info']['geodesic_distance']
                    })
    
    print(f"=== 找到 {len(sofa_episodes)} 个 sofa episodes ===\n")
    for i, ep in enumerate(sofa_episodes):
        print(f"{i+1}. Episode ID: {ep['episode_id']}")
        print(f"   场景: {ep['scene_id']}")
        print(f"   最短路径距离: {ep['geodesic_distance']:.2f}m")
        print()
    
    return sofa_episodes

if __name__ == "__main__":
    sofa_eps = find_sofa_episodes()
