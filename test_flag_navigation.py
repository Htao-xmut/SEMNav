#!/usr/bin/env python3
"""
测试寻找美国国旗 (American flag) 的任务
使用 ObjectNav 数据集，但手动过滤出包含 flag 的 episodes
"""

import sys
import os
sys.path.insert(0, 'src')

import gzip
import json
from pathlib import Path
from collections import Counter

def find_flag_episodes():
    """查找数据集中所有包含 flag 的 episodes"""
    dataset_dir = Path('data/datasets/objectnav_hm3d_v2/val_mini/content')
    
    if not dataset_dir.exists():
        print(f"❌ 数据集目录不存在: {dataset_dir}")
        return []
    
    flag_episodes = []
    all_categories = []
    
    for scene_file in sorted(dataset_dir.glob('*.json.gz')):
        with gzip.open(scene_file, 'rt') as f:
            data = json.load(f)
            for episode in data.get('episodes', []):
                cat = episode['object_category']
                all_categories.append(cat)
                
                # 查找包含 flag 的类别
                if 'flag' in cat.lower():
                    flag_episodes.append({
                        'scene': scene_file.name,
                        'episode': episode,
                        'category': cat
                    })
    
    # 统计所有类别
    counter = Counter(all_categories)
    print("=== ObjectNav val_mini 数据集中的物体类别 ===")
    print(f"总 Episode 数: {len(all_categories)}\n")
    print("各类别数量（前20个）:")
    for cat, count in counter.most_common(20):
        print(f"  {cat}: {count}")
    
    print(f"\n=== Flag 相关 Episodes ===")
    if flag_episodes:
        print(f"找到 {len(flag_episodes)} 个包含 flag 的 episodes:\n")
        for i, ep_info in enumerate(flag_episodes[:10]):  # 只显示前10个
            print(f"{i+1}. 场景: {ep_info['scene']}")
            print(f"   类别: {ep_info['category']}")
            print(f"   场景ID: {ep_info['episode']['scene_id']}")
            print()
    else:
        print("❌ 未找到任何包含 flag 的 episodes")
        print("\n💡 建议的替代方案:")
        print("   1. 使用其他常见物体进行测试，如: chair, table, sofa, bed")
        print("   2. 或者创建一个自定义的测试场景")
    
    return flag_episodes

if __name__ == "__main__":
    flag_eps = find_flag_episodes()
