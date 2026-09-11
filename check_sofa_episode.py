#!/usr/bin/env python3
import gzip
import json

dataset_file = 'data/datasets/objectnav_hm3d_v2/val_mini/content/wcojb4TFT35.json.gz'

with gzip.open(dataset_file, 'rt') as f:
    data = json.load(f)
    episodes = data.get('episodes', [])
    
    # 找到第一个 sofa episode
    for ep in episodes:
        if ep['object_category'] == 'sofa':
            print("Sofa Episode keys:", list(ep.keys()))
            print("\nshortest_paths:", ep.get('shortest_paths', 'NOT FOUND'))
            print("\ninfo:", ep.get('info', {}))
            print("\ngoals:", ep.get('goals', []))
            
            # 查看 shortest_paths 的结构
            if ep.get('shortest_paths'):
                print("\nFirst shortest_path structure:")
                sp = ep['shortest_paths'][0]
                print(f"  Keys: {list(sp.keys())}")
                if 'positions' in sp:
                    print(f"  Positions count: {len(sp['positions'])}")
                    print(f"  First position: {sp['positions'][0]}")
            break
