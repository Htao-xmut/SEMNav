#!/usr/bin/env python3
"""AVDB navigation test runner."""
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

# Build config for AVDB run
config = {
    'task': 'ObjectNav',
    'agent_cls': 'ObjectNavAgent',
    'env_cls': 'AVDBEnv',
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
                'model': 'qwen-vl-max',
                'api_key': os.environ.get('QWEN_API_KEY', ''),
                'max_image_res': 1024,
                'system_instruction': (
                    "You are a visual navigation assistant in a home environment. "
                    "Respond ONLY with JSON: {'reasoning': '...', 'action': <0-3>}. "
                    "Actions: 0=Turn 180° behind you, 1=Turn left ~30°, 2=Move forward ~0.65m, 3=Turn right ~30°"
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
        'max_steps': 20,
        'log_freq': 1,
        'split': 'val',
        'target_category': 'crystal_hot_sauce',
        'success_threshold': 0.3,
        'instances': 1,
        'instance': 0,
        'parallel': False,
        'name': 'avdb_demo',
        'port': 5000,
    },
}

# Check API key
api_key = config['agent_cfg']['vlm_cfg']['model_kwargs']['api_key']
if not api_key:
    # Try alternate env var names
    for var in ['DASHSCOPE_API_KEY', 'QWEN_API_KEY']:
        api_key = os.environ.get(var, '')
        if api_key:
            config['agent_cfg']['vlm_cfg']['model_kwargs']['api_key'] = api_key
            break
if not api_key:
    # Try reading from ObjectNav.yaml as last resort
    try:
        from omegaconf import OmegaConf
        obj_cfg = OmegaConf.load('config/ObjectNav.yaml')
        obj_cfg = OmegaConf.to_container(obj_cfg, resolve=True)
        api_key = obj_cfg['agent_cfg']['vlm_cfg']['model_kwargs'].get('api_key', '')
        if api_key:
            config['agent_cfg']['vlm_cfg']['model_kwargs']['api_key'] = api_key
    except: pass
if not api_key:
    print("ERROR: No API key found! Set QWEN_API_KEY or DASHSCOPE_API_KEY.")
    print("  export QWEN_API_KEY='sk-...'")
    sys.exit(1)

print(f"Starting AVDB navigation test...")
print(f"  Target: {config['env_cfg']['target_category']}")
print(f"  Max steps: {config['env_cfg']['max_steps']}")
print(f"  VLM: {config['agent_cfg']['vlm_cfg']['model_kwargs']['model']}")
print()

env = AVDBEnv(cfg=config)
env.run_experiment()

# Print summary
import glob
latest = sorted(glob.glob('logs/ObjectNav_avdb_demo/0_of_1/*/'))[-1] if glob.glob('logs/ObjectNav_avdb_demo/0_of_1/*/') else None
if latest:
    print(f"\nResults saved to: {latest}")
    steps = sorted(glob.glob(f'{latest}step*/'))
    actions = []
    for s in steps:
        details = os.path.join(s, 'details.txt')
        if os.path.exists(details):
            with open(details) as f:
                content = f.read()
                for line in content.split('\n'):
                    if line.startswith('ACTION_NUMBER'):
                        continue
                # parse action from details
    print(f"Total steps: {len(steps)}")
