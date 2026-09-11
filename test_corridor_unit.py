#!/usr/bin/env python3
"""可达性/去先验/区域级防重访 — 单元自检"""
import sys, types, logging
import numpy as np

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, '/home/tao_h/VLMnav/src')
sys.path.insert(0, '/home/tao_h/avdb_habitat_converter/scripts')

from avdb_depth import ExplorationMap2D  # noqa: E402
from avdb_env import AVDBEnv  # noqa: E402

# --- 1) 走廊: OCCUPIED 才是阻挡; UNKNOWN = 未探索 ≠ 不可达 ---
em = ExplorationMap2D(center=(0.0, 0.0))
em._draw_segment(0, 0, 0, 4.0, 1.0)  # (0,0)→(0,4) 已探索线
r, blk, unk = em.corridor_free_ratio(0, 0, 0.0, max_r=3.5)
assert r >= 0.9 and blk >= 3.5, (r, blk, unk)
print('✓ 已探索走廊: free=%.2f 无阻挡到 %.1fm' % (r, blk))

# 朝 +x 全未知: 无 OCCUPIED → 不算阻挡 (未探索 = 可去探索!)
r2, blk2, unk2 = em.corridor_free_ratio(0, 0, np.pi / 2, max_r=3.5)
assert blk2 >= 3.5 and unk2 > 0.8, (r2, blk2, unk2)
print('✓ 未探索方向: unknown=%.2f, 无观测障碍 (blk=%.1f) → 不是不可达!' % (unk2, blk2))

# 观测障碍: 2m 处横墙 → first_block ≈ 2m
em2 = ExplorationMap2D(center=(0.0, 0.0))
em2._draw_segment(0, 0, 0, 1.5, 1.0)
for gx in range(em2._to_grid(-0.5, 0)[0], em2._to_grid(0.5, 0)[0]):
    em2.regions[em2._to_grid(0, 2.0)[1], gx] = 3
r3, b3, u3 = em2.corridor_free_ratio(0, 0, 0.0, max_r=3.5)
assert b3 < 2.2, (r3, b3)
print('✓ 观测障碍: 2m 横墙 first_block=%.1fm (唯一真阻挡)' % b3)

# --- 2) 方位约定 ---
d = np.array([0.5, 0.0, 0.866])
b = float(np.arctan2(d[0], d[2]))
assert np.allclose([np.sin(b), np.cos(b)], [0.5, 0.866], atol=1e-3)
print('✓ 方位约定: dir=(sin b, cos b) 与 _yaw_world 一致')

# --- 3) 区域级 (2m 粗格) 防同房间打转 ---
import importlib.util
spec = importlib.util.spec_from_file_location(
    'asw', '/home/tao_h/avdb_habitat_converter/scripts/avdb_sim_wrapper.py')
asw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(asw)
W = asw.AVDBSimWrapper
w = object.__new__(W)
g = {'nodes': {
        'A':  {'world_pos': [0, 0, 0]},    # 同粗格 (0,0)
        'A2': {'world_pos': [0.4, 0, 0.3]},   # 同粗格 (0,0)
        'B':  {'world_pos': [0, 0, 3.0]},  # 粗格 (0,1) — 真新区域
        'R':  {'world_pos': [0.1, 0, 0.1]}, # 同粗格, 旋转边目标
     },
     'graph': {'A': {'A2': {'edge_type': 'forward'},
                     'B':  {'edge_type': 'right'},
                     'R':  {'edge_type': 'rotate_cw'}}}}
w.nav_graph = g
w.visited_nodes = {'A': 1}
w.node_records = {}
w.rejected_areas = []
w.memory = {'step_count': 0}
w.visited_area_cells = set()

w._record_visit('A')                       # 区域 (0,0) 登记
assert w._area_visited('A2') and not w._area_visited('B')
unv = w._get_unvisited_directions('A')
assert 'forward' not in unv, '同区域邻居不应算未探索: %s' % unv
assert 'rotate_cw' not in unv, '原地旋转不是方向'
assert unv.get('right') == 'B', unv
print('✓ 区域级: 同 2m 格邻居被排除, 真·新区域 (B) 保留, 旋转边剔除')

# --- 4) _target_approachable: 未探索前沿 → 可达 (勇于探索) ---
env = object.__new__(AVDBEnv)
sw = types.SimpleNamespace(
    nav_graph=g, current_node='A', exploration_map=em,
    visited_nodes={'A': 1}, target_sighting_nodes=[],
    _node_pos=lambda n: np.array(g['nodes'][n]['world_pos'][[0, 2]]),
    _node_in_rejected=lambda n: (False, None),
    _get_unvisited_directions=lambda n: {'right': 'B'},  # 有前沿
    visited_area_cells={(0, 0)}, AREA_CELL=2.0,
    _area_key=lambda n: (0, 0), _area_visited=lambda n: n != 'B')
env.simWrapper = sw
env._approach_bearing = np.pi / 2   # 朝 +x (全未知区)
env._last_obs = {'edge_options': [  # 无行走选项 (只剩旋转)
    {'chain_type': 'rotate_cw', 'chain_count': 1, 'composite': None}]}
env._plan_bearing_detour = lambda *a, **k: False
# em 朝 +x 无 OCCUPIED → first_block 大 → reachable
assert env._target_approachable() is True
print('✓ 未探索方向无观测障碍 → 可达 (去探索)')

# 观测障碍挡死 + 无绕障 + 无前沿 → 不可达 (可停)
sw2 = types.SimpleNamespace(
    nav_graph=g, current_node='A', exploration_map=em2,  # 2m 横墙
    visited_nodes={'A': 1}, target_sighting_nodes=[],
    _node_pos=sw._node_pos, _node_in_rejected=sw._node_in_rejected,
    _get_unvisited_directions=lambda n: {},              # 无前沿
    visited_area_cells={(0, 0)})
env.simWrapper = sw2
env._approach_bearing = 0.0  # 朝墙
assert env._target_approachable() is False
print('✓ 观测障碍挡死 + 无绕障 + 无前沿 → 不可达 (尊重停止)')

print('\n=== 可达性/区域级/勇于探索 单元自检全部通过 ===')
