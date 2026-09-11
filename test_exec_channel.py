"""执行通道单测: edge_idx 物理挂载 / stop 单例不污染 / 否决停票选边

bm7/bm12 实测大坑: env 层覆盖只改账面 (df/图/日志), 从不设置
action.edge_idx → wrapper 一直按 VLM raw 选择执行, "executed [X]" 说谎;
stop 单例无 edge_idx → wrapper 回退 [0] = 180° turn_around。
运行: python -u test_exec_channel.py
"""
import sys, logging
sys.path.insert(0, '/home/tao_h/VLMnav/src')
sys.path.insert(0, '/home/tao_h/avdb_habitat_converter/scripts')
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)

from avdb_env import AVDBEnv
from simWrapper import PolarAction

PASS, FAIL = [], []


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f'  [{info}]' if info and not cond else ''))


def make_env():
    env = object.__new__(AVDBEnv)
    env.step = 7
    env.cfg = {'sensor_cfg': {'fov': 131}}
    env.df = pd.DataFrame({'action_number': [3]})
    env._approach_bearing = None
    env._last_approach_idx = -1

    class SW:
        target_name = 't'
        current_node = 'N1'
        nav_graph = {'nodes': {
            'N1': {'world_pos': [0, 0, 0], 'direction': [0, 0, 1]}}}

    env.simWrapper = SW()
    return env


def opt(et, cnt=1):
    return {'chain_type': et, 'chain_count': cnt, 'composite': None}


def test_attach_exec_idx():
    env = make_env()
    # 普通动作: 原对象 + 挂上执行号
    a = PolarAction(0.65, 0)
    out = env._attach_exec_idx(a, 5)
    check('通道: 普通动作保留原对象并挂 edge_idx=5',
          out is a and out.edge_idx == 5)
    # stop 单例: 换新对象, 且单例本身不被污染
    out = env._attach_exec_idx(PolarAction.stop, 4)
    check('通道: stop 单例 → 新对象挂 edge_idx=4 (物理执行覆盖)',
          out is not PolarAction.stop and out.edge_idx == 4)
    check('通道: stop 单例未被污染 (无 edge_idx 属性)',
          not hasattr(PolarAction.stop, 'edge_idx'))
    # null 单例同理
    out = env._attach_exec_idx(PolarAction.null, 2)
    check('通道: null 单例 → 新对象挂 edge_idx=2',
          out is not PolarAction.null and out.edge_idx == 2)
    check('通道: null 单例未被污染',
          not hasattr(PolarAction.null, 'edge_idx'))
    # 非法 idx → 0
    out = env._attach_exec_idx(PolarAction(0, 0), None)
    check('通道: 非法 idx → 兜底 0', out.edge_idx == 0)


def test_vetoed_stop_action():
    # 有接近锁 (+90°) → 选 right (bearing 90) 而非 forward(0)/backward(180)
    env = make_env()
    obs = {'edge_options': [opt('forward'), opt('right'), opt('backward')]}
    env._approach_bearing = np.radians(90)      # yaw_world=0
    act = env._vetoed_stop_action(obs)
    check('否决停: 有锁 → 当步走锁定方位对齐选项 [1] (right)',
          act is not PolarAction.stop and act.edge_idx == 1,
          f'edge_idx={getattr(act, "edge_idx", None)}')
    # 锁在背后 (180°) 有 +10° 惩罚, forward(0) 差 180 也大 → 仍选最优
    env2 = make_env()
    obs2 = {'edge_options': [opt('forward'), opt('left'), opt('backward')]}
    env2._approach_bearing = np.radians(-90)
    act2 = env2._vetoed_stop_action(obs2)
    check('否决停: 锁 -90° → 选 left [1]',
          act2.edge_idx == 1, f'edge_idx={act2.edge_idx}')
    # 无锁 → df 原始移动选择 (3 超出选项数 → 弃用 → 0)
    env3 = make_env()
    obs3 = {'edge_options': [opt('forward'), opt('right'), opt('backward'),
                             opt('left')]}
    act3 = env3._vetoed_stop_action(obs3)
    check('否决停: 无锁 → VLM 原始移动选择 [3] (df)',
          act3.edge_idx == 3, f'edge_idx={act3.edge_idx}')
    # df 号超出选项 → 0
    env4 = make_env()
    obs4 = {'edge_options': [opt('forward'), opt('right')]}
    act4 = env4._vetoed_stop_action(obs4)
    check('否决停: df 号越界 → 兜底 [0]', act4.edge_idx == 0)
    # 空 df 无锁 → 0
    env5 = make_env()
    env5.df = pd.DataFrame({'action_number': []})
    act5 = env5._vetoed_stop_action({'edge_options': [opt('forward')]})
    check('否决停: 空 df 无锁 → [0]', act5.edge_idx == 0)
    # 全程不得污染 stop 单例
    check('否决停: stop 单例全程未被污染',
          not hasattr(PolarAction.stop, 'edge_idx'))


if __name__ == '__main__':
    test_attach_exec_idx()
    test_vetoed_stop_action()
    print(f"\n{'='*40}\n✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        print('Failed:', *FAIL, sep='\n  - ')
        sys.exit(1)
