#!/usr/bin/env python3
import gzip
import json

dataset_file = 'data/datasets/objectnav_hm3d_v2/val_mini/content/wcojb4TFT35.json.gz'

with gzip.open(dataset_file, 'rt') as f:
    data = json.load(f)
    episodes = data.get('episodes', [])
    
    if episodes:
        ep = episodes[0]
        print("Episode keys:", list(ep.keys()))
        print("\ngoals field:", ep.get('goals', 'NOT FOUND'))
        print("\ninfo field:", ep.get('info', {}))
        print("\nFull episode structure:")
        for key in ['episode_id', 'scene_id', 'start_position', 'object_category', 'goals', 'info']:
            if key in ep:
                print(f"  {key}: {str(ep[key])[:150]}")
