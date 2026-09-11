#!/usr/bin/env python3
"""精确方向伺服单元自检 — bbox→方位角 + 距离感知选边"""
import sys, types, logging

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, '/home/tao_h/VLMnav/src')
from avdb_env import AVDBEnv  # noqa: E402

env = object.__new__(AVDBEnv)
env.simWrapper = types.SimpleNamespace(target_name='coca_cola_glass_bottle')
env.cfg = {'sensor_cfg': {'fov': 131}}

# --- 1) bbox → 相对方位角 (FOV 131°: cx=1 → +65.5°, cx=0 → -65.5°) ---
def yolo(cx):
    return {'all_detections': [{'class_name': 'coca_cola_glass_bottle',
                                'bbox_norm': [cx - 0.05, 0.4, cx + 0.05, 0.6]}]}

assert env._detection_bearing_deg(yolo(0.5)) == 0.0
assert abs(env._detection_bearing_deg(yolo(1.0)) - 65.5) < 0.1
assert abs(env._detection_bearing_deg(yolo(0.0)) + 65.5) < 0.1
assert abs(env._detection_bearing_deg(yolo(0.33)) + 22.2) < 0.1  # 旧 9 宫格 "left" 边界
# 无框 / 别的类 → None
assert env._detection_bearing_deg({'all_detections': []}) is None
assert env._detection_bearing_deg({'all_detections': [
    {'class_name': 'other', 'bbox_norm': [0.1, 0.1, 0.2, 0.2]}]}) is None
print('✓ 方位角: cx→deg 精确换算, 无框返回 None')

# --- 2) 精确选边: 合成选项表 (模拟真实 edge_options) ---
def opts(*items):
    return [{'chain_type': t, 'chain_count': c, 'composite': comp, 'direction': t}
            for (t, c, comp) in items]

# 典型节点: ccw30 / ccw60 / cw30 / cw60 / fwd / 复合(ccw30+fwd) / 复合(cw30+fwd) / left / right
TABLE = opts(
    ('rotate_ccw', 1, None), ('rotate_ccw', 2, None),
    ('rotate_cw', 1, None), ('rotate_cw', 2, None),
    ('forward', 1, None),
    ('rotate_ccw', 1, [('rotate_ccw', 1), ('forward', 1)]),
    ('rotate_cw', 1, [('rotate_cw', 1), ('forward', 1)]),
    ('left', 1, None), ('right', 1, None),
)
def pick(rel, area, table=TABLE):
    return env._pick_steer_option_precise({'edge_options': table}, rel, area)

# 目标右 45° 远 (小框): 应选复合 cw30+fwd (转+走一步到位), 而非纯转/纯走
i = pick(45, 0.03)
assert TABLE[i]['composite'] and TABLE[i]['composite'][0] == ('rotate_cw', 1), i
# 目标右 45° 近 (大框): 复合被 +25 惩罚 → 纯转 cw60 更贴
i = pick(45, 0.40)
assert TABLE[i]['chain_type'] == 'rotate_cw' and not TABLE[i].get('composite'), i
# 目标居中 (rel≈0) 远: forward×1 (无 fwd×2 时) — 走, 不转
i = pick(3, 0.02)
assert TABLE[i]['chain_type'] == 'forward', i
# 目标居中 近: forward 仍可选 (无近惩罚作用于纯 forward? 有 walk → 罚)
#   近+居中: 全部含走动作都被罚 25 → rotate_cw/ccw (ob=±30) 分数 55,
#   forward 分数 0+25=25 → forward 仍最低, 可接受 (接近检查会接管)
i = pick(0, 0.5)
assert TABLE[i]['chain_type'] == 'forward', i
# 目标左 60° 中距: ccw×2 纯转 (ob=-60 贴合 0)
i = pick(-60, 0.2)
assert TABLE[i]['chain_type'] == 'rotate_ccw' and TABLE[i]['chain_count'] == 2, i
# 同侧过滤: 目标右 40° 时左向动作 (ccw/left) 不可能中选
i = pick(40, 0.2)
assert TABLE[i]['chain_type'] not in ('rotate_ccw', 'left'), i
print('✓ 精确选边: 方向贴合 + 距离整形 (近只转/远复合) + 同侧过滤')

# --- 3) 复合动作方位 = 旋转分量 (执行后面朝方向) ---
assert env._option_bearing_deg(TABLE[5]) == -30.0
assert env._option_bearing_deg(TABLE[6]) == 30.0
assert env._option_bearing_deg(TABLE[1]) == -60.0
print('✓ 选项方位: 复合取旋转分量, cnt 倍乘')

print('\n=== 精确伺服单元自检全部通过 ===')
