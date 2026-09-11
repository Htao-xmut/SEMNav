#!/usr/bin/env python3
"""基准测试单组运行 — 参数: <组号> <category> <episode_idx>

配置与 test_avdb_nav_depth_coca.py 完全一致, 仅换物品/起点/日志名。
"""
import sys, os, logging

os.environ['HABITAT_SIM_HEADLESS'] = '1'
os.environ['EGL_PLATFORM'] = 'surfaceless'

sys.path.insert(0, '/home/tao_h/VLMnav/src')
sys.path.insert(0, '/home/tao_h/avdb_habitat_converter/scripts')

GRP, CAT, IDX = sys.argv[1], sys.argv[2], int(sys.argv[3])
NAME = f'bm{GRP}_{CAT.split("_")[0]}{GRP}'

from avdb_env import AVDBEnv
from vlm import QwenVLClient
from simWrapper import PolarAction
from omegaconf import OmegaConf
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

config = {
    'task': 'ObjectNav',
    'agent_cls': 'ObjectNavAgent',
    'env_cls': 'AVDBEnv',
    'target_category': CAT,
    'start_episode_idx': IDX,
    'agent_cfg': {
        'navigability_mode': 'depth_sensor',
        'project': False,
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
            },
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
        'sensor_cfg': {'height': 1.5, 'pitch': -0.45, 'res_factor': 2, 'fov': 131},
    },
    'env_cfg': {
        'num_episodes': 1,
        'max_steps': 50,
        'log_freq': 1,
        'split': 'val',
        'target_category': CAT,
        'start_episode_idx': IDX,
        'success_threshold': 1.5,
        'instances': 1,
        'instance': 0,
        'parallel': False,
        'name': NAME,
        'port': 5000,
    },
}

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
    print("ERROR: No API key found!")
    sys.exit(1)

print(f"[BM{GRP}] {CAT} ep[{IDX}] → logs/ObjectNav_{NAME}")
env = AVDBEnv(cfg=config)
env.run_experiment()
print(f"[BM{GRP}] DONE")
