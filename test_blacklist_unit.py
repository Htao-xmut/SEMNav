#!/usr/bin/env python3
"""误报黑名单 + 结构化导航记忆 — 单元级自检 (不启动 habitat_sim)

用 object.__new__ 构造 AVDBSimWrapper 裸实例, 只装填需要的状态,
验证: 状态机流转 / 半径判定 / 幂等黑名单 / 记忆文本 / 选项标记。
"""
import sys, os, json, logging
import numpy as np

sys.path.insert(0, '/home/tao_h/avdb_habitat_converter/scripts')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# 避免导入 habitat_sim 依赖链 — 直接从模块文件加载类
import importlib.util
spec = importlib.util.spec_from_file_location(
    'avdb_sim_wrapper', '/home/tao_h/avdb_habitat_converter/scripts/avdb_sim_wrapper.py')

# 1) 语法/导入检查 (会拉起 habitat_sim, headless 可行)
import avdb_sim_wrapper as asw
W = asw.AVDBSimWrapper
print("✓ import OK")

# 2) 裸实例 (绕过 __init__ 的 sim 创建)
w = object.__new__(W)
with open('/home/tao_h/avdb_habitat_converter/data/processed/navigation_graph.json') as f:
    w.nav_graph = json.load(f)
w.current_node = None
w.memory = {'short_term': [], 'rooms': {}, 'adjacent': {},
            'target_hints': {}, 'step_count': 0, 'pending_locations': []}
w.visited_nodes = {}
w.visited_edges = {}
w.node_records = {}
w.rejected_areas = []
w.rejected_sightings = []
w.target_sighting_nodes = []
w.yolo_history = []
w.target_name = 'coca_cola_glass_bottle'

nodes = list(w.nav_graph['nodes'].keys())
A, B, C = nodes[0], nodes[1], nodes[2]
w.current_node = A

# --- 状态机流转 ---
w._record_visit(A)
assert w.node_records[A]['state'] == 'empty' and w.node_records[A]['visits'] == 1
w._record_visit(A)
assert w.node_records[A]['visits'] == 2
w._record_detection(A, 'YOLO left conf=0.95')
assert w.node_records[A]['state'] == 'promising', w.node_records[A]
w._record_confirmed(A)
assert w.node_records[A]['state'] == 'confirmed'
print("✓ 状态机: empty→promising→confirmed OK")
w.node_records[A]['state'] = 'promising'  # 回退继续测 rejected

# --- 黑名单: 登记 + 幂等 + 半径判定 ---
w.record_rejection(A, 'VLM wrong object')
assert w.node_records[A]['state'] == 'rejected'
assert len(w.rejected_areas) == 1
w.record_rejection(A, 'again')  # 幂等
assert len(w.rejected_areas) == 1
assert len(w.rejected_sightings) == 1
# 半径内邻居节点应被覆盖
posA = w._node_pos(A)
covered = [n for n in nodes[1:200]
           if np.linalg.norm(w._node_pos(n) - posA) <= w.REJECT_RADIUS]
assert len(covered) > 0, "1.5m 半径应覆盖至少一个邻居"
assert all(w._node_in_rejected(n)[0] for n in covered)
far = [n for n in nodes
       if np.linalg.norm(w._node_pos(n) - posA) > 3.0][0]
assert not w._node_in_rejected(far)[0]
print(f"✓ 黑名单: 幂等登记 OK, 半径覆盖 {len(covered)} 邻居节点")

# --- rejected 状态下的重复目击抑制 ---
w.target_sighting_nodes.clear()
w.record_vlm_sighting(f"i see coca cola glass bottle on the table at {A}")
assert not w.target_sighting_nodes, "黑名单节点 VLM 目击应被忽略"
w.current_node = far  # record_vlm_sighting 记的是 current_node
w.record_vlm_sighting(f"i see coca cola glass bottle on the table")
assert len(w.target_sighting_nodes) == 1
assert w.node_records.get(far, {}).get('state') == 'promising'
print("✓ 目击抑制: 黑名单节点忽略, 正常节点记录 OK")

# --- 记忆文本结构 ---
txt = w.get_memory_context()
for sec in ['📍 VISITED AREAS', '⛔ AVOID', '🎯 VALID SIGHTING', '🌟 UNVISITED']:
    assert sec in txt, f"记忆缺少段落: {sec}"
assert 'checked — NOT the target' in txt
assert 'bearing' in txt  # 避坑带方位/距离
print("✓ 记忆文本: 4 段结构 + 避坑方位 OK")

# --- 选项标记 (拦截层1) + landing node ---
w.current_node = far
w.last_edge_options = [{'chain_type': 'forward', 'chain_count': 1, 'composite': None, 'direction': 'forward'}]
landing = w._option_landing_node(w.last_edge_options[0])
print(f"  (info) option from {far}: landing={landing}, "
      f"blacklisted={w._node_in_rejected(landing)[0] if landing else 'N/A(纯旋转)'}")

# --- 深度候选过滤 (拦截层1b) ---
edges = w.nav_graph['graph'][far]
n_trans = sum(1 for t, e in edges.items()
              if e['edge_type'] in ('forward', 'backward', 'left', 'right'))
bl_trans = sum(1 for t, e in edges.items()
               if e['edge_type'] in ('forward', 'backward', 'left', 'right')
               and w._node_in_rejected(t)[0])
print(f"✓ 深度过滤入口: {far} 平移边 {n_trans} 条, 其中 {bl_trans} 条将被剔除")

print("\n=== 全部单元自检通过 ===")
