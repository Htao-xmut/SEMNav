"""batch runner 计量单测: VLM 调用差分 / 两级拒停计数 / 终局汇总 / 档位映射

不启动模拟器, 不调用真实 VLM。
运行: python -u test_batch_metering.py
"""
import sys, logging
sys.path.insert(0, '/home/tao_h/VLMnav/src')
sys.path.insert(0, '/home/tao_h/avdb_habitat_converter/scripts')

logging.basicConfig(level=logging.WARNING)

import pandas as pd
from avdb_env import AVDBEnv

PASS, FAIL = [], []


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f'  [{info}]' if info and not cond else ''))


class _FakeVLM:
    def __init__(self, n=0):
        self.call_count = n


class _FakeAgent:
    def __init__(self, stopping_calls=None, action_n=0, stopping_n=0):
        self.actionVLM = _FakeVLM(action_n)
        self.stoppingVLM = _FakeVLM(stopping_n)
        self.stopping_calls = stopping_calls if stopping_calls is not None else [-2]
        self.stop_history = [True, True, False]


def make_env(agent=None, df=None, ep_stats=None, episode=None):
    env = object.__new__(AVDBEnv)
    env.agent = agent or _FakeAgent()
    env.df = df if df is not None else pd.DataFrame([{
        'finish_status': 'success', 'goal_reached': True,
        'distance_to_goal': 0.82, 'spl': 0.61, 'action_number': 3}])
    env._ep_stats = ep_stats if ep_stats is not None else {
        'episode_ndx': 4, 'stop_vetoed': 0, 'votes_cleared': 0,
        'vlm_calls_start': 10, 'initial_geodesic': 7.3}
    env.current_episode = episode or {'object_category': 'softsoap_white'}
    env.episode_stats_list = []
    return env


# ---------- ① VLM 调用差分 ----------
def test_vlm_total():
    env = make_env(agent=_FakeAgent(action_n=12, stopping_n=8))
    check('差分: 双 VLM call_count 求和 = 20', env._vlm_call_total() == 20)
    env2 = make_env(agent=_FakeAgent(action_n=3, stopping_n=0))
    env2.agent.stoppingVLM = None   # 缺属性容错 (getattr 默认 0)
    check('差分: 缺 stoppingVLM 属性 → 只计 actionVLM',
          env2._vlm_call_total() == 3)


# ---------- ② 两级拒停计数 ----------
def test_reject_levels():
    env = make_env()
    env._reject_stop_and_approach('[TEST] action-level veto')
    check('计数: 默认 (动作级) → stop_vetoed+1, votes_cleared 不动',
          env._ep_stats['stop_vetoed'] == 1 and env._ep_stats['votes_cleared'] == 0)
    env._reject_stop_and_approach('[TEST] vote-level clear', vote_level=True)
    check('计数: vote_level=True → votes_cleared+1 (与动作级分开)',
          env._ep_stats['stop_vetoed'] == 1 and env._ep_stats['votes_cleared'] == 1)
    # 清票从尾部 True 连续段清起; 尾部 False = 连票已断, 循环即止
    env.agent.stop_history = [False, True, True, True]
    env._reject_stop_and_approach('[TEST] action-level veto again')
    check('清票: 尾部连续 True 票被清, 早先 False 不动 (连票不复燃)',
          env.agent.stop_history == [False, False, False, False],
          str(env.agent.stop_history))
    env2 = make_env()   # 无 _ep_stats (异常路径) 不炸
    env2._ep_stats = None
    env2._reject_stop_and_approach('[TEST] no stats')
    check('计数: _ep_stats 缺失时容错 (不抛异常)', True)


# ---------- ③ 终局汇总 ----------
def test_collect():
    env = make_env(agent=_FakeAgent(stopping_calls=[-2, 5, 9]))
    env._collect_episode_stats()
    st = env.episode_stats_list[-1]
    check('汇总: 终局字段齐全 (df 尾行 + 类别 + 步数)',
          st['finish_status'] == 'success' and st['goal_reached'] is True
          and st['final_distance'] == 0.82 and st['spl'] == 0.61
          and st['category'] == 'softsoap_white' and st['steps'] == 1)
    check('汇总: stop_requests = len(stopping_calls)-1 = 2',
          st['stop_requests'] == 2)
    check('汇总: initial_geodesic 透传', st['initial_geodesic'] == 7.3)
    env.agent.actionVLM.call_count = 16
    env.agent.stoppingVLM.call_count = 11
    env._collect_episode_stats()
    st2 = env.episode_stats_list[-1]
    check('汇总: vlm_calls = 终值 27 - 基线 10 = 17', st2['vlm_calls'] == 17)
    check('汇总: 两行追加', len(env.episode_stats_list) == 2)

    # 空 df (异常终局) — 不炸, finish_status='no_log'
    env3 = make_env(df=pd.DataFrame({}))
    env3._collect_episode_stats()
    st3 = env3.episode_stats_list[-1]
    check('汇总: 空 df 容错 → no_log / 距离 -1',
          st3['finish_status'] == 'no_log' and st3['final_distance'] == -1.0)


# ---------- ④ 档位映射 (batch_run TIERS ↔ feature_flags 记名) ----------
def test_tiers():
    sys.path.insert(0, '/home/tao_h/VLMnav')
    import batch_run
    import feature_flags
    expect = {'full': '+C+G+S+W', 'CGS': '+C+G+S', 'CG': '+C+G',
              'C': '+C', 'base': 'base'}
    for tier, want in expect.items():
        ff = feature_flags.resolve({'feature_flags': batch_run.TIERS[tier]})
        got = feature_flags.describe(ff)
        check(f'档位: {tier} → {want}', got == want, got)
    # 基线档位必须四机制全关 (纯 VLMnav 复现)
    check('档位: base 四开关全 False',
          not any(batch_run.TIERS['base'].values()))


# ---------- ⑤ single_thresh 对照档: 收紧前单阈值判据复现 ----------
def test_single_thresh():
    env = make_env()
    env._ff = {'single_thresh': True}
    env._apply_flag_overrides()
    check('single_thresh: 深探边界退回 2.2 / 视觉上限退回 2.5',
          env.HARD_STOP_DEPTH == 2.2 and env.SOFT_APPROACH_DEPTH == 2.2
          and env.VISUAL_SUCCESS_THRESHOLD == 2.5)
    check('single_thresh: 类常量不被污染 (实例级覆盖)',
          AVDBEnv.HARD_STOP_DEPTH == 1.0
          and AVDBEnv.VISUAL_SUCCESS_THRESHOLD == 1.0)
    env2 = make_env()
    env2._ff = {}
    env2._apply_flag_overrides()
    check('single_thresh: 默认档不受影响 (沿用类常量)',
          env2.HARD_STOP_DEPTH == 1.0 and env2.VISUAL_SUCCESS_THRESHOLD == 1.0)


if __name__ == '__main__':
    test_vlm_total()
    test_reject_levels()
    test_collect()
    test_tiers()
    test_single_thresh()
    print(f"\n{'='*40}\n✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        print('Failed:', *FAIL, sep='\n  - ')
        sys.exit(1)
