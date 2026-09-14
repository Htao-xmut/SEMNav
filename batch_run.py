#!/usr/bin/env python3
"""Batch runner — N episodes × 配置档位 → CSV + 汇总统计 (表 VI/VIII 数据源)

单进程单 env 顺序跑, 每 episode 终局由 AVDBEnv._collect_episode_stats
汇总 (df 尾行 + 停票门两级计数 + VLM 调用差分) → episode_stats_list。

用法:
  python -u batch_run.py --episodes 32 [--start 0] [--config full] \
      [--out logs/batch_xxx.csv] [--max-steps 50] [--log-freq 1]

配置档位 (--config, 消融记号 C=置信分档级联 / G=持久栅格中枢 /
S=米制停票门 / W=warmup 抢占):
  full  = +C+G+S+W  (默认, 全机制)
  CGS   = +C+G+S
  CG    = +C+G
  C     = +C
  base  = 纯 VLMnav 复现 (四机制全关)
自定义档位: 其余 feature_flags CLI 开关直接透传, 如:
  python -u batch_run.py --episodes 8 --no-warmup-preempt
"""
import sys, os, time, argparse, logging

os.environ.setdefault('HABITAT_SIM_HEADLESS', '1')
os.environ.setdefault('EGL_PLATFORM', 'surfaceless')

sys.path.insert(0, '/home/tao_h/VLMnav/src')
sys.path.insert(0, '/home/tao_h/avdb_habitat_converter/scripts')

import feature_flags
from feature_flags import parse_argv

# 档位 → 开关覆盖 (与 feature_flags.describe 记名一致)
TIERS = {
    'full': {},
    'CGS':  {'warmup_preempt': False},
    'CG':   {'metric_gate': False, 'warmup_preempt': False},
    'C':    {'grid_hub': False, 'metric_gate': False, 'warmup_preempt': False},
    'base': {'cascade': False, 'grid_hub': False,
             'metric_gate': False, 'warmup_preempt': False},
}


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--episodes', type=int, default=32, help='跑多少 episode')
    p.add_argument('--start', type=int, default=0, help='起始 episode 序号 (过滤后列表 0-based)')
    p.add_argument('--config', type=str, default='full',
                   choices=sorted(TIERS.keys()), help='消融档位')
    p.add_argument('--out', type=str, default='', help='CSV 输出路径 (默认 logs/batch_<tier>_s<start>_n<N>.csv)')
    p.add_argument('--max-steps', type=int, default=50)
    p.add_argument('--log-freq', type=int, default=1,
                   help='图片落盘频率 (0=不落盘, 批量长跑可关)')
    return p


def main():
    args, passthrough = build_argparser().parse_known_args()
    cli_flags, _rest = parse_argv(passthrough)   # --no-xxx / --gt-detect 透传

    # 档位开关 + CLI 透传开关合成 (CLI 优先)
    ff = dict(TIERS[args.config])
    ff.update(cli_flags)
    tier_name = feature_flags.describe(feature_flags.resolve({'feature_flags': ff}))

    import pandas as pd
    from avdb_env import AVDBEnv
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get('QWEN_API_KEY', '') or os.environ.get('DASHSCOPE_API_KEY', '')
    if not api_key:
        print('ERROR: No API key found (QWEN_API_KEY / DASHSCOPE_API_KEY)!')
        sys.exit(1)

    name = f'batch_{tier_name}_s{args.start}_n{args.episodes}'
    out_csv = args.out or f'logs/{name}.csv'
    # 长 (数小时) 批跑必须落盘日志: 终端缓冲/SSH 断连会丢 console 输出
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        handlers=[logging.FileHandler(f'logs/{name}.log'),
                                  logging.StreamHandler(sys.stdout)])
    config = {
        'task': 'ObjectNav',
        'agent_cls': 'ObjectNavAgent',
        'env_cls': 'AVDBEnv',
        'feature_flags': ff,
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
                    'api_key': api_key,
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
            'num_episodes': args.episodes,
            'batch_start': args.start,      # 连续切片 (bm_run 单跑用 start_episode_idx)
            'max_steps': args.max_steps,
            'log_freq': args.log_freq,
            'split': 'val',
            'success_threshold': 1.5 if ff.get('single_thresh') else 1.0,
            # 1.3 收紧: SR@1.0m 主判据 (single_thresh 对照档配 1.5 复现收紧前)
            'instances': 1,
            'instance': 0,
            'parallel': False,
            'name': name,
            'port': 5000,
        },
    }

    print(f'[BATCH] tier={tier_name} episodes={args.episodes} start={args.start} '
          f'max_steps={args.max_steps} flags={ff}')
    print(f'[BATCH] CSV → {out_csv}')
    t0 = time.time()
    env = AVDBEnv(cfg=config)

    # 增量落盘: 每 episode 终局后立刻重写 CSV — 数小时批跑中途崩溃
    # (API 连锁失败/内存/异常) 只丢当前 episode, 已完成的全部保留
    _orig_collect = env._collect_episode_stats

    def _collect_and_dump():
        _orig_collect()
        pd.DataFrame(env.episode_stats_list).to_csv(out_csv, index=False)
        st = env.episode_stats_list[-1]
        done = len(env.episode_stats_list)
        print(f'[BATCH] [{done}/{args.episodes}] ep{st["episode_ndx"]} '
              f'{st["category"][:28]} → {st["finish_status"]} '
              f'({st["final_distance"]:.2f}m, {st["vlm_calls"]} VLM calls, '
              f'{(time.time() - t0) / 60:.0f} min elapsed)')

    env._collect_episode_stats = _collect_and_dump
    env.run_experiment()
    mins = (time.time() - t0) / 60

    # ---- 汇总: episode_stats_list → CSV + 终端报告 ----
    rows = getattr(env, 'episode_stats_list', [])
    if not rows:
        print('[BATCH] WARNING: no episode stats collected!')
        return
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f'\n[BATCH] {len(df)} episodes → {out_csv} ({mins:.0f} min total, '
          f'{mins / max(1, len(df)):.1f} min/ep)')

    n = len(df)
    sr = df['goal_reached'].mean()
    spl = df['spl'].mean()
    fs = df['finish_status'].value_counts().to_dict()
    stops = int(df['stop_requests'].sum())
    fp_n = int((df['finish_status'] == 'fp').sum())
    succ_d = df.loc[df['goal_reached'], 'final_distance']
    print(f'\n===== SUMMARY ({tier_name}, n={n}) =====')
    print(f'SR@1.0m      : {sr:.1%}')
    print(f'SPL          : {spl:.3f}')
    print(f'finish       : {fs}')
    print(f'fp rate      : {fp_n}/{n} eps = {fp_n / n:.1%}'
          + (f' | fp/stops = {fp_n}/{stops} = {fp_n / max(1, stops):.1%}' if stops else ''))
    if len(succ_d):
        print(f'stop dist    : success median {succ_d.median():.2f}m '
              f'(max {succ_d.max():.2f}m)')
    print(f'final dist   : median {df["final_distance"].median():.2f}m '
          f'mean {df["final_distance"].mean():.2f}m')
    print(f'stop requests: {stops} total ({stops / n:.1f}/ep) | '
          f'vetoed {int(df["stop_vetoed"].sum())} | '
          f'votes cleared {int(df["votes_cleared"].sum())}')
    print(f'VLM calls    : mean {df["vlm_calls"].mean():.1f}/ep '
          f'(total {int(df["vlm_calls"].sum())})')
    print(f'steps        : mean {df["steps"].mean():.1f}/ep')


if __name__ == '__main__':
    main()
