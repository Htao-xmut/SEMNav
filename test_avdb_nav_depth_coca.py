#!/usr/bin/env python3
"""AVDB 深度注入导航测试 — 长距离 episode (之前无法完成的例子)

episode 51: coca_cola_glass_bottle, 图距离 9.19m (之前无法完成的长距离任务)
深度 Decision Trace (可行走方向+安全距离+置信度+180°判断) 注入 CoT 提示词。
"""
import sys, os, logging

# Must be set before habitat_sim import
os.environ['HABITAT_SIM_HEADLESS'] = '1'
os.environ['EGL_PLATFORM'] = 'surfaceless'

sys.path.insert(0, '/home/tao_h/VLMnav/src')
sys.path.insert(0, '/home/tao_h/avdb_habitat_converter/scripts')

from avdb_env import AVDBEnv
from agent import ObjectNavAgent
from vlm import QwenVLClient
from simWrapper import PolarAction
from omegaconf import OmegaConf
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Build config for AVDB run with depth trace injection
config = {
    'task': 'ObjectNav',
    'agent_cls': 'ObjectNavAgent',
    'env_cls': 'AVDBEnv',
    # 顶层键: 由 avdb_env 读取 (target_category / start_episode_idx)
    'target_category': 'coca_cola_glass_bottle',
    'start_episode_idx': 5,   # 过滤后列表第5个 = episode 51 (图距离 9.19m, 之前失败)
    'agent_cfg': {
        'navigability_mode': 'depth_sensor',
        'project': False,           # no_project mode: uses memory_section
        'pivot': False,
        'context_history': 0,
        'explore_bias': 4,
        'max_action_dist': 0.8,
        'min_action_dist': 0.3,
        'clip_frac': 0.66,
        'stopping_action_dist': 1.5,
        'default_action': 0.2,
        'spacing_ratio': 360,
        'num_theta': 60,
        'image_edge_threshold': 0.04,
        'turn_around_cooldown': 3,
        'navigability_height_threshold': 0.2,
        'map_scale': 100,
        'vlm_cfg': {
            'model_cls': 'QwenVLClient',
            'model_kwargs': {
                'model': 'qwen-vl-plus',
                'api_key': os.environ.get('QWEN_API_KEY', ''),
                'max_image_res': 1024,
                'system_instruction': (
                    "You are a visual navigation assistant in a home environment. "
                    "Respond ONLY with JSON: {'reasoning': '...', 'action': <N>}. "
                    "The action numbers are defined ONLY by the 'AVAILABLE DIRECTIONS' "
                    "list in each prompt — always pick the number from that list. "
                    "If a 'DEPTH SCAN' block gives a recommended direction, the line "
                    "'Matching action for recommendation: [N] ...' tells you the exact "
                    "number to pick for it."
                ),
            }
        },
    },
    'sim_cfg': {
        'scene_id': 'avdb_home001_1',
        'scene_path': '/home/tao_h/VLMnav/data/scene_datasets/avdb_home001_1/Home_001_1.glb',
        'scene_config': '/home/tao_h/VLMnav/data/scene_datasets/avdb_home001_1/scene_dataset_config.json',
        'agent_height': 1.5,
        'agent_radius': 0.17,
        'allow_slide': True,
        'use_goal_image_agent': False,
        'sensor_cfg': {
            'height': 1.5,
            'pitch': -0.45,
            'res_factor': 2,
            'fov': 131,
        },
    },
    'env_cfg': {
        'num_episodes': 1,
        'max_steps': 50,
        'log_freq': 1,
        'split': 'val',
        'target_category': 'coca_cola_glass_bottle',
        'start_episode_idx': 5,   # 过滤后列表第5个 = episode 51 (图距离 9.19m, 之前失败)
        'success_threshold': 1.5,
        'instances': 1,
        'instance': 0,
        'parallel': False,
        'name': 'avdb_depth_ep51_coca_explore',
        'port': 5000,
    },
}

# Check API key
api_key = config['agent_cfg']['vlm_cfg']['model_kwargs']['api_key']
if not api_key:
    for var in ['DASHSCOPE_API_KEY', 'QWEN_API_KEY']:
        api_key = os.environ.get(var, '')
        if api_key:
            config['agent_cfg']['vlm_cfg']['model_kwargs']['api_key'] = api_key
            break
if not api_key:
    try:
        obj_cfg = OmegaConf.load('config/ObjectNav.yaml')
        obj_cfg = OmegaConf.to_container(obj_cfg, resolve=True)
        api_key = obj_cfg['agent_cfg']['vlm_cfg']['model_kwargs'].get('api_key', '')
        if api_key:
            config['agent_cfg']['vlm_cfg']['model_kwargs']['api_key'] = api_key
    except Exception:
        pass
if not api_key:
    print("ERROR: No API key found! Set QWEN_API_KEY or DASHSCOPE_API_KEY.")
    sys.exit(1)

print("Starting AVDB depth-injection navigation test...")
print(f"  Target: {config['env_cfg']['target_category']}")
print(f"  Episode: filtered index 3 (episode 67, graph ~5.96m — previously failed)")
print(f"  Max steps: {config['env_cfg']['max_steps']}")
print(f"  VLM: {config['agent_cfg']['vlm_cfg']['model_kwargs']['model']}")
print(f"  Depth trace: injected into CoT prompt (walkable dirs + safe dist + conf + 180°)")
print()

env = AVDBEnv(cfg=config)
env.run_experiment()

import glob
runs = sorted(glob.glob('logs/ObjectNav_avdb_depth_ep51_coca_explore/0_of_1/*/'))
if runs:
    latest = runs[-1]
    print(f"\nResults saved to: {latest}")
    steps = sorted(glob.glob(f'{latest}step*/'))
    print(f"Total step dirs: {len(steps)}")
    for s in steps:
        files = sorted(os.listdir(s))
        print(f"  {os.path.basename(s.rstrip('/'))}: {files}")
