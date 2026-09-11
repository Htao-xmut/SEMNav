"""Batch comparison: Ollama LLaVA vs Qwen-VL-Max across target categories"""
import os
import sys
import time
import json
import gzip
import yaml
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Config
DATASET_DIR = Path('data/datasets/objectnav_hm3d_v2/val_mini/content')
MAX_STEPS = 10
TARGET_CATEGORIES = ['chair', 'bed', 'sofa', 'tv_monitor', 'plant']

MODELS = {
    'Ollama-LLaVA-7b': {
        'model_cls': 'OllamaVLM',
        'model_kwargs': {
            'model': 'llava:7b',
            'max_image_res': 512,
            'system_instruction': None,
        }
    },
    'Qwen-VL-Max': {
        'model_cls': 'QwenVLClient',
        'model_kwargs': {
            'model': 'qwen-vl-max',
            'api_key': os.environ.get('QWEN_API_KEY', ''),,
            'max_image_res': 1024,
            'system_instruction': "You are a professional visual navigation assistant. You MUST respond with ONLY a valid JSON object in the format {'action': <number>}. Do NOT include any explanation text.",
        }
    },
}

def load_config_template():
    with open('config/ObjectNav.yaml') as f:
        return yaml.safe_load(f)

def get_episode_index_for_category(target_category):
    """Find the first episode index for a given category"""
    all_episodes = []
    for scene_file in sorted(DATASET_DIR.glob('*.json.gz')):
        with gzip.open(scene_file, 'rt') as f:
            data = json.load(f)
            for ep in data.get('episodes', []):
                all_episodes.append(ep)

    for i, ep in enumerate(all_episodes):
        if ep['object_category'] == target_category:
            return i, ep['scene_id'], ep['info']['geodesic_distance']
    return None, None, None

def update_config_yaml(model_name, model_info, target_category, episode_idx):
    """Update ObjectNav.yaml with given model and target settings"""
    with open('config/ObjectNav.yaml') as f:
        content = f.read()

    import re

    # Update model_cls
    content = re.sub(r'model_cls: \w+', f"model_cls: {model_info['model_cls']}", content)

    # Update model section
    if model_name == 'Ollama-LLaVA-7b':
        content = re.sub(
            r"model_cls: OllamaVLM\n    model_kwargs:\n      model: .*?\n      max_image_res: \d+\n      system_instruction: .*",
            f"model_cls: OllamaVLM\n    model_kwargs:\n      model: llava:7b\n      max_image_res: 512\n      system_instruction: null",
            flags=re.DOTALL
        )
    else:
        content = re.sub(
            r"model_cls: QwenVLClient\n    model_kwargs:\n      model: .*?\n      api_key: .*?\n      max_image_res: \d+\n      system_instruction: .*",
            f"model_cls: QwenVLClient\n    model_kwargs:\n      model: qwen-vl-max\n      api_key: ''  # via QWEN_API_KEY / DASHSCOPE_API_KEY env (vlm.py fallback)\n      max_image_res: 1024\n      system_instruction: \"You are a professional visual navigation assistant. You MUST respond with ONLY a valid JSON object in the format {{'action': <number>}}. Do NOT include any explanation text.\"",
            flags=re.DOTALL
        )

    # Update target_category
    content = re.sub(r'target_category: \w+', f"target_category: {target_category}", content)

    # Remove start_episode_idx or set to the right value
    if 'start_episode_idx' in content:
        content = re.sub(r'start_episode_idx: \d+', f"start_episode_idx: {episode_idx}", content)
    else:
        # Add after target_category
        content = content.replace(
            f"target_category: {target_category}",
            f"target_category: {target_category}\n  start_episode_idx: {episode_idx}"
        )

    # Set max_steps and num_episodes
    content = re.sub(r'max_steps: \d+', f"max_steps: {MAX_STEPS}")
    content = re.sub(r'num_episodes: \d+', "num_episodes: 1")

    with open('config/ObjectNav.yaml', 'w') as f:
        f.write(content)

def run_experiment(model_name, target_category):
    """Run one experiment and return results"""
    episode_idx, scene_id, geo_distance = get_episode_index_for_category(target_category)
    if episode_idx is None:
        print(f"  ⚠️ No episode found for {target_category}")
        return None

    model_info = MODELS[model_name]

    print(f"  Config: model={model_name}, target={target_category}, episode_idx={episode_idx}, distance={geo_distance:.2f}m")
    update_config_yaml(model_name, model_info, target_category, episode_idx)

    # Clear cache
    os.system('find src -name "*.pyc" -delete 2>/dev/null')
    os.system('find src -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null')

    # Run
    start_time = time.time()
    ret = os.system(
        'source /home/tao_h/miniconda3/bin/activate vlm_nav && '
        'export PYTHONPATH=/home/tao_h/VLMnav/src:$PYTHONPATH && '
        'timeout 300 python scripts/main.py --config ObjectNav --num_episodes 1 --max_steps 10 2>&1 | '
        'grep -E "RUNNING EPISODE|Geodesic|Goal|finish_status|GOAL_REACHED|SPL|Model called stop|Bad action|Step [0-9]+$|RUN COMPLETE" | head -30'
    )
    elapsed = time.time() - start_time

    # Find latest log
    logs_dir = Path('logs')
    latest_log = max(logs_dir.glob('ObjectNav_default_*'), key=os.path.getctime)

    # Parse results
    result = {
        'model': model_name,
        'target': target_category,
        'scene': scene_id,
        'geodesic_distance': geo_distance,
        'elapsed_sec': elapsed,
        'log_dir': str(latest_log),
    }

    # Try to read episode result
    try:
        ep_dirs = list((latest_log / '0_of_1').glob('*'))
        if ep_dirs:
            ep_dir = ep_dirs[0]
            # Find last step
            steps = sorted(ep_dir.glob('step*'))
            if steps:
                last_step = steps[-1]
                details = last_step / 'details.txt'
                if details.exists():
                    with open(details) as f:
                        content = f.read()
                        for line in content.split('\n'):
                            line = line.strip()
                            if line == 'success':
                                result['finish_status'] = 'success'
                            elif line == 'max_steps':
                                result['finish_status'] = 'max_steps'
                            elif line == 'True':
                                if 'goal_reached_section' not in result:
                                    result['goal_reached'] = True
                            elif line == 'False':
                                if 'goal_reached_section' not in result:
                                    result['goal_reached'] = False

                result['total_steps'] = len(steps)

                # Count actions and stopping
                actions = []
                stop_count = 0
                for step in steps:
                    detail_file = step / 'details.txt'
                    if detail_file.exists():
                        with open(detail_file) as f:
                            txt = f.read()
                            if '"done": 1' in txt or "'done': 1" in txt:
                                stop_count += 1
                            # Extract action
                            for line in txt.split('\n'):
                                if line.strip() == 'ACTION_NUMBER':
                                    continue
                            import re as re_m
                            action_match = re_m.search(r'ACTION_NUMBER\s*\n\s*(-?\d+)', txt)
                            if action_match:
                                actions.append(int(action_match.group(1)))

                result['actions'] = actions
                result['unique_actions'] = len(set(a for a in actions if a >= 0))
                result['stop_count'] = stop_count

                # Check SPL
                with open(details) as f:
                    full = f.read()
                    spl_match = re_m.search(r'SPL\s*\n\s*([\d.]+)', full)
                    if spl_match:
                        result['spl'] = float(spl_match.group(1))
    except Exception as e:
        result['parse_error'] = str(e)

    return result

def main():
    results = []

    print("=" * 60)
    print("VLMnav Batch Comparison: Ollama LLaVA vs Qwen-VL-Max")
    print("=" * 60)

    for target in TARGET_CATEGORIES:
        print(f"\n{'='*40}")
        print(f"Target: {target.upper()}")
        print(f"{'='*40}")

        for model_name in ['Ollama-LLaVA-7b', 'Qwen-VL-Max']:
            print(f"\n  [{model_name}]")
            result = run_experiment(model_name, target)
            if result:
                results.append(result)
                print(f"    Result: status={result.get('finish_status', '?')}, "
                      f"steps={result.get('total_steps', '?')}, "
                      f"unique_actions={result.get('unique_actions', '?')}, "
                      f"stops={result.get('stop_count', 0)}, "
                      f"SPL={result.get('spl', 0):.3f}" if 'spl' in result else f"    Result: parse error")
            else:
                print(f"    ⚠️ Failed")

    # Save results
    with open('batch_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Results saved to batch_results.json")
    print(f"{len(results)} experiments completed")

if __name__ == '__main__':
    main()
