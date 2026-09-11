#!/usr/bin/env python3
"""A2 两击规则单元自检 — 同区域 15 步内第 2 次 3步丢失 → 升级拉黑

不启动 habitat_sim: 用裸 env 实例 + mock simWrapper。
"""
import sys, types, logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
sys.path.insert(0, '/home/tao_h/VLMnav/src')

from avdb_env import AVDBEnv  # noqa: E402

env = object.__new__(AVDBEnv)

# --- mock simWrapper ---
sw = types.SimpleNamespace()
sw.current_node = 'nA'
sw._node_pos = lambda n: {'nA': (0.0, 0.0), 'nB': (0.8, 0.0),  # 0.8m: 同区域
                          'nC': (5.0, 5.0)}[n]                   # 远: 不同区域
# record_rejection 计数器
rejections = []
sw.record_rejection = lambda node, reason: rejections.append((node, reason))
sw.target_sighting_nodes = [(3, 'nS')]

env.simWrapper = sw
env._approach_bearing = 1.23
env._approach_miss = 3
env._forced_return_path = ['forward']
env.step = 10

# --- 第 1 次丢失: 只解锁, 不拉黑 ---
env._invalidate_lock('target lost for 3 steps — detection stale',
                     blacklist=False, drop_sighting=False)
assert env._approach_bearing is None
assert len(rejections) == 0, '第1次丢失不应拉黑'
assert sw.target_sighting_nodes, '第1次丢失应保留目击'
assert len(env._loss_strikes) == 1
print('✓ 第1次丢失: 只解锁, 无拉黑, 目击保留')

# --- 远处第 2 次丢失 (不同区域): 仍不拉黑 ---
env._approach_bearing = 0.5
sw.current_node = 'nC'
env.step = 20
env._invalidate_lock('target lost for 3 steps', blacklist=False, drop_sighting=False)
assert len(rejections) == 0, '不同区域不应触发两击'
assert len(env._loss_strikes) == 2
print('✓ 不同区域丢失: 不触发两击')

# --- 同区域第 2 次丢失 (nB 距 nA 0.8m, 步数间隔 5): 升级拉黑 ---
env._approach_bearing = 0.5
sw.current_node = 'nB'
sw.target_sighting_nodes = [(22, 'nS')]
env.step = 25  # 距第1次(10) 15 步内; 距第2次(20) 也在 15 内但 nC 远, recent 只匹配 nA
env._invalidate_lock('target lost for 3 steps', blacklist=False, drop_sighting=False)
assert len(rejections) >= 1, '同区域两次丢失应升级拉黑'
assert not sw.target_sighting_nodes, '升级后应丢弃目击'
kicked = [r for r in rejections if 'TWO-STRIKES' in r[1] or 'false sighting' in r[1]]
assert kicked, f'拉黑原因应含升级标记: {rejections}'
print(f'✓ 两击升级: 拉黑 {len(rejections)} 处 {rejections}')

# --- 超时 (>15步) 不触发 ---
env._loss_strikes = [(5, [0.0, 0.0])]
sw.current_node = 'nA'
sw.target_sighting_nodes = [(30, 'nS2')]
env.step = 40  # 间隔 35 步
env._approach_bearing = 0.5
before = len(rejections)
env._invalidate_lock('lost', blacklist=False, drop_sighting=False)
assert len(rejections) == before, '超时的旧 strike 不应触发'
print('✓ 超 15 步的旧记录不触发')

print('\n=== 两击规则单元自检全部通过 ===')
