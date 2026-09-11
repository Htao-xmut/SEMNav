#!/usr/bin/env python3
"""
测试寻找 sofa 的任务（10步）
手动选择第一个 sofa episode
"""

import sys
import os
sys.path.insert(0, 'src')

import gzip
import json
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def get_sofa_episode():
    """获取第一个 sofa episode"""
    dataset_dir = Path('data/datasets/objectnav_hm3d_v2/val_mini/content')
    
    all_episodes = []
    for scene_file in sorted(dataset_dir.glob('*.json.gz')):
        with gzip.open(scene_file, 'rt') as f:
            data = json.load(f)
            for ep in data.get('episodes', []):
                all_episodes.append(ep)
    
    # 找到第一个 sofa
    for i, ep in enumerate(all_episodes):
        if ep['object_category'] == 'sofa':
            logging.info(f"✅ 找到 sofa episode:")
            logging.info(f"   索引: {i}")
            logging.info(f"   Episode ID: {ep['episode_id']}")
            logging.info(f"   场景: {ep['scene_id']}")
            logging.info(f"   距离: {ep['info']['geodesic_distance']:.2f}m")
            return i, ep
    
    raise ValueError("未找到 sofa episode")

if __name__ == "__main__":
    idx, ep = get_sofa_episode()
    print(f"\n💡 要测试这个 episode，需要修改 ObjectNavEnv._initialize_experiment 方法")
    print(f"   或者在运行时传入 episode_index={idx}")
