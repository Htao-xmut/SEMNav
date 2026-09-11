#!/usr/bin/env python3
"""遮挡感知探索单元自检 — 深度图阴影检测 + 换角度偷看"""
import sys, types, logging
import numpy as np

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, '/home/tao_h/VLMnav/src')
sys.path.insert(0, '/home/tao_h/avdb_habitat_converter/scripts')

from avdb_depth import ExplorationMap2D  # noqa: E402
from avdb_env import AVDBEnv  # noqa: E402


def put_wall(em, x0, x1, z, state=3):
    """在深度图上放一排指定状态的格 (世界坐标)"""
    gx0, _ = em._to_grid(x0, 0)
    gx1, _ = em._to_grid(x1, 0)
    _, gz = em._to_grid(0, z)
    for gx in range(gx0, gx1 + 1):
        em.regions[gz, gx] = state


# --- 1) 遮挡阴影检测: 前方 2m 障碍背后从未观测 → pocket ---
em = ExplorationMap2D(center=(0.0, 0.0))
em._draw_segment(0, 0, 0, 1.8, 1.0)      # (0,0)→(0,1.8) 已观测走廊
put_wall(em, -0.15, 0.15, 2.0)            # 2m 处 0.3m 宽障碍
pockets = em.occlusion_pockets(0, 0, max_r=4.0)
assert pockets, '障碍背后 UNKNOWN 应产生 pocket'
pb, pd, punk = pockets[0]
assert abs(np.degrees(pb)) < 8, ('正前方障碍 → 方位≈0', np.degrees(pb))
assert 1.6 < pd < 2.4, ('距离≈2m', pd)
assert punk >= 4, ('阴影带 UNKNOWN 格数', punk)
print('✓ 阴影检测: 前方 %.1fm 障碍背后 → pocket (bearing %+.0f°, %d 个未知格)'
      % (pd, np.degrees(pb), punk))

# 背后已被观测过 (FREE) → 不是阴影
em2 = ExplorationMap2D(center=(0.0, 0.0))
em2._draw_segment(0, 0, 0, 1.8, 1.0)
put_wall(em2, -0.15, 0.15, 2.0)
for xoff in np.arange(-0.2, 0.21, 0.05):  # 障碍整宽背后都看过
    em2._draw_segment(xoff, 2.3, xoff, 3.2, 1.0)
assert not em2.occlusion_pockets(0, 0, max_r=4.0)
print('✓ 背后已观测 (FREE) → 不算遮挡阴影')

# 右侧 +x 障碍 → 方位 ≈ +90° (约定: 面朝 +z, +x 在右)
em3 = ExplorationMap2D(center=(0.0, 0.0))
em3._draw_segment(0, 0, 0, 1.0, 1.0)
put_wall(em3, 2.0, 2.0, 0.0)              # 单点障碍 → 网格上是一格
for gz in range(em3._to_grid(0, -0.15)[1], em3._to_grid(0, 0.15)[1] + 1):
    em3.regions[gz, em3._to_grid(2.0, 0)[0]] = 3   # (2.0, -0.15~0.15) 竖排
p3 = em3.occlusion_pockets(0, 0, max_r=4.0)
assert p3 and 75 < np.degrees(p3[0][0]) < 105, \
    ('右侧障碍 → 方位≈+90°', [round(np.degrees(b), 1) for b, _, _ in p3])
print('✓ 右侧障碍 → pocket 方位 %+.0f° (arctan2(dx,dz) 约定)' % np.degrees(p3[0][0]))


# --- 2) _occlusion_peek: 选贴合方位的选项 + 方位去重 ---
env = object.__new__(AVDBEnv)
env.simWrapper = types.SimpleNamespace(
    exploration_map=em, current_node='A',
    nav_graph=None,                      # _yaw_world → 0.0
    _node_pos=lambda n: np.array([0.0, 0.0]))
env.step = 10
env._peeked_bearings = []
calls = []
env._override_and_run = lambda obs, i, tag, msg: calls.append((i, tag)) or ('PEEKED', i, tag)

TABLE = [{'chain_type': t, 'chain_count': 1, 'composite': None, 'direction': t}
         for t in ('rotate_ccw', 'rotate_cw', 'forward', 'left', 'right')]
obs = {'edge_options': TABLE}

res = env._occlusion_peek(obs)
assert res is not None and calls, '应执行偷看'
i, tag = calls[0]
assert TABLE[i]['chain_type'] == 'forward', ('正前方阴影 → forward', TABLE[i])
assert tag == 'OCCLUSION-PEEK'
assert env._last_peek_step == 10 and env._peeked_bearings, '登记步数+方位'
print('✓ 偷看执行: 正前方阴影 → [forward], 方位已登记')

# 已偷看过的方位 (±20°) → 不重复
assert env._occlusion_peek(obs) is None, '同方位阴影应被去重'
print('✓ 方位去重: ±20° 内不重复偷看')

# 右侧阴影 (换 em3) → left 侧移 (世界 +90°, ob=+90 是 right?) —
# 约定: 面朝 +z, +x 在右。右转 (cw) 朝 +x。ob(right 侧移)=+90 → 贴合
env2 = object.__new__(AVDBEnv)
env2.simWrapper = types.SimpleNamespace(
    exploration_map=em3, current_node='A', nav_graph=None,
    _node_pos=lambda n: np.array([0.0, 0.0]))
env2.step = 11
env2._peeked_bearings = []
calls2 = []
env2._override_and_run = lambda obs, i, tag, msg: calls2.append(i) or 1
env2._occlusion_peek(obs)
assert calls2 and TABLE[calls2[0]]['chain_type'] == 'right', \
    ('右侧阴影 → right 侧移/转向', calls2)
print('✓ 右侧阴影 → [right] (换位比原地转看得开, 侧移有 −3° 加分)')

# 无阴影 → 不动作
env3 = object.__new__(AVDBEnv)
env3.simWrapper = types.SimpleNamespace(
    exploration_map=em2, current_node='A', nav_graph=None,
    _node_pos=lambda n: np.array([0.0, 0.0]))
env3.step = 12
assert env3._occlusion_peek(obs) is None
print('✓ 无阴影地图 → 返回 None (交给正常 VLM 步)')

print('\n=== 遮挡感知探索单元自检全部通过 ===')
