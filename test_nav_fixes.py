"""单元测试: 5 项导航修复 (滞回/反震荡/死区/身份确认/scan→地图)

不启动模拟器 — object.__new__ 绕过 __init__, 只装配被测属性。
运行: python -u test_nav_fixes.py
"""
import sys, os, types, logging
sys.path.insert(0, '/home/tao_h/VLMnav/src')
sys.path.insert(0, '/home/tao_h/avdb_habitat_converter/scripts')
import numpy as np

logging.basicConfig(level=logging.WARNING)

from avdb_sim_wrapper import AVDBSimWrapper
from avdb_depth import DepthConfig, ExplorationMap2D
from avdb_env import AVDBEnv

PASS, FAIL = [], []


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f'  [{info}]' if info and not cond else ''))


def make_wrapper():
    w = object.__new__(AVDBSimWrapper)
    w.nav_graph = {'nodes': {
        'A': {'world_pos': [0.0, 0.0, 0.0], 'direction': [0, 0, 1]},
        'B': {'world_pos': [0.0, 0.0, 0.65], 'direction': [0, 0, 1]},
        'C': {'world_pos': [0.65, 0.0, 0.0], 'direction': [0, 0, 1]},
        'N1': {'world_pos': [0.0, 0.0, 0.0], 'direction': [0, 0, 1]},
        'N2': {'world_pos': [0.3, 0.0, 0.3], 'direction': [0.87, 0, 0.5]},
        # 区域分/逃逸测试用: B2 在另一 2m 粗格 (新区域)
        'C2': {'world_pos': [1.3, 0.0, 0.0], 'direction': [0, 0, 1]},
        'B2': {'world_pos': [2.5, 0.0, 0.5], 'direction': [0, 0, 1]},
    }, 'graph': {
        'N1': {
            'C': {'edge_type': 'right', 'distance': 0.65},
            'B': {'edge_type': 'forward', 'distance': 0.65},
        },
        'C': {'N1': {'edge_type': 'left', 'distance': 0.65},
              'C2': {'edge_type': 'right', 'distance': 0.65}},
        'C2': {'C': {'edge_type': 'left', 'distance': 0.65},
               'B2': {'edge_type': 'forward', 'distance': 0.65}},
        'B': {'N1': {'edge_type': 'backward', 'distance': 0.65}},
        'B2': {'C2': {'edge_type': 'backward', 'distance': 0.65}},
    }}
    w.visited_nodes = {}
    w.visited_area_cells = set()
    w.rejected_areas = []
    w.exploration_map = None
    w.depth_cfg = DepthConfig()
    w.current_node = 'N1'
    w._walk_hist = []

    class CamStub:
        cam_R = staticmethod(lambda d: np.eye(3))
    w.depth_cam = CamStub()
    return w


# ---------- ② 深度推荐滞回 ----------
def test_hysteresis():
    w = make_wrapper()
    # 上一步沿 +90° 走 (wdeg=90); 当前节点朝向 0°
    w._walk_hist = [{'wdeg': 90.0, 'from': 'A', 'to': 'C'}]
    # 新推荐 -89° (与 +90 相差 179° → 反向), 分数只比非反向备选高 0.02
    scored = {'recommended': {'rep_deg': -89.0, 'score': 0.90},
              'alternatives': [{'rep_deg': 85.0, 'score': 0.88},
                               {'rep_deg': 180.0, 'score': 0.95}]}
    out = w._apply_rec_hysteresis(scored, 'N1')
    check('滞回: 反向低分差推荐被换成非反向备选',
          abs(out['recommended']['rep_deg'] - 85.0) < 1e-6)
    check('滞回: 旧推荐降级进 alternatives',
          any(abs(a['rep_deg'] + 89.0) < 1e-6 for a in out['alternatives']))
    # 分数差足够大 (0.45 ≥ 0.15) → 保留反向推荐
    scored2 = {'recommended': {'rep_deg': -89.0, 'score': 0.95},
               'alternatives': [{'rep_deg': 85.0, 'score': 0.50}]}
    out2 = w._apply_rec_hysteresis(scored2, 'N1')
    check('滞回: 反向但明显更优 → 保留', abs(out2['recommended']['rep_deg'] + 89.0) < 1e-6)
    # 推荐不反向 → 原样
    scored3 = {'recommended': {'rep_deg': 30.0, 'score': 0.9},
               'alternatives': [{'rep_deg': 60.0, 'score': 0.89}]}
    out3 = w._apply_rec_hysteresis(scored3, 'N1')
    check('滞回: 同向推荐不受影响', abs(out3['recommended']['rep_deg'] - 30.0) < 1e-6)


# ---------- ③ 反震荡 A→B→A ----------
def test_oscillation():
    w = make_wrapper()
    w._record_walk('A', 'B', 'forward')     # A→B
    w._record_walk('B', 'A', 'backward')    # B→A (回头)
    check('震荡: A→B→A 检出禁回节点 B', w._osc_ban_node() == 'B')
    w2 = make_wrapper()
    w2._record_walk('A', 'B', 'forward')
    w2._record_walk('B', 'C', 'right')      # B→C 前进, 不回头
    check('震荡: A→B→C 无摆动', w2._osc_ban_node() is None)
    w3 = make_wrapper()
    w3._record_walk('A', 'B', 'rotate_cw')  # 纯旋转不记历史
    check('震荡: 旋转动作不入历史', len(w3._walk_hist) == 0)


# ---------- ④ 伺服死区 ----------
def make_env():
    e = object.__new__(AVDBEnv)
    e.simWrapper = types.SimpleNamespace(
        current_node='N1', target_name='coca_cola_glass_bottle',
        recent_edge_types=['rotate_cw'], nav_graph={'nodes': {}})
    e.cfg = {'sensor_cfg': {'fov': 131}}
    return e


OPTS = [  # idx: 0..4
    {'chain_type': 'rotate_cw', 'chain_count': 1},            # 0: +30°
    {'chain_type': 'rotate_ccw', 'chain_count': 1},           # 1: -30°
    {'composite': [('rotate_cw', 1), ('forward', 1)]},        # 2: +30°再走
    {'chain_type': 'forward', 'chain_count': 1},              # 3: 前进
    {'chain_type': 'left', 'chain_count': 1},                 # 4: 左移
]


def test_deadband():
    e = make_env()
    obs = {'edge_options': OPTS}
    # rel=15° (死区内): 纯旋转全排除 → 只会选含 walk 的选项
    idx = e._pick_steer_option_precise(obs, 15.0, 0.02)
    check('死区: |rel|≤20° 不再原地转', idx is not None and idx in (2, 3),
          f'idx={idx}')
    # rel=35° (死区外): 旋转恢复可选
    idx2 = e._pick_steer_option_precise(obs, 35.0, 0.02)
    check('死区外: 朝向旋转/复合可选', idx2 is not None and idx2 in (0, 2),
          f'idx={idx2}')
    # 刚转过 (last rotate_cw) + rel=-25° (30° 分级死区内):
    # 反向纯转被吸收, 走平移选项 → 不再右转完立刻左转
    e2 = make_env()
    idx3 = e2._pick_steer_option_precise(obs, -25.0, 0.02)
    check('分级死区: 刚转过 + 偏25° 不反向转 (吸收过冲噪声)',
          idx3 is not None and idx3 != 1, f'idx={idx3}')
    # 真跑偏 (rel=-35°, 超过 30° 分级死区) → 反向转恢复可用
    e3 = make_env()
    e3.simWrapper.recent_edge_types = []
    idx4 = e3._pick_steer_option_precise(obs, -35.0, 0.02)
    check('真跑偏 >30°: 反向旋转恢复可用 (idx ∈ {1,4})', idx4 in (1, 4),
          f'idx={idx4}')


# ---------- ① 身份确认分支 ----------
def test_identity_gate():
    e = make_env()
    e.step = 10
    yolo = {'target_found': True, 'confidence': 0.51, 'position': 'middle_center',
            'area_ratio': 0.004,
            'all_detections': [{'class_name': 'coca_cola_glass_bottle',
                                'confidence': 0.51, 'bbox_norm': [0.4, 0.4, 0.6, 0.6]}]}
    obs = {'color_sensor': np.zeros((480, 640, 3), np.uint8), 'edge_options': OPTS,
           'yolo_detection': yolo}
    calls = {'vlm': 0, 'inval': [], 'ovr': []}

    e._detection_bearing_deg = lambda yi: 5.0
    e._vlm_identity_check = lambda o, y: (calls.__setitem__('vlm', calls['vlm'] + 1)
                                          or 'no')
    e._invalidate_lock = lambda r, **kw: calls['inval'].append((r, kw))
    e._override_and_run = lambda o, i, t, n='': calls['ovr'].append((i, t)) or ('ret', i)

    out = e._approach_step(obs)
    check('身份=no: 解锁 (drop_sighting) 并回落探索 (返回 None)',
          out is None and calls['vlm'] == 1
          and calls['inval'] and calls['inval'][0][1].get('drop_sighting') is True
          and calls['inval'][0][1].get('blacklist') is False)
    check('身份=no: 节点记入排除窗', e._identity_reject.get('N1') == 10)

    # 排除窗内 (step 未过 10 步) 不再问 VLM, 直接探索
    e.step = 12
    out2 = e._approach_step(obs)
    check('排除窗内: 不再重复 VLM 确认 (返回 None)', out2 is None and calls['vlm'] == 1)

    # unsure 分支: 走近/换角度覆盖
    e2 = make_env()
    e2.step = 20
    e2._detection_bearing_deg = lambda yi: 5.0
    e2._vlm_identity_check = lambda o, y: 'unsure'
    e2._invalidate_lock = lambda r, **kw: None
    e2._override_and_run = lambda o, i, t, n='': calls['ovr'].append((i, t)) or ('ret', i)
    out3 = e2._approach_step(obs)
    check('身份=unsure: 触发 CONFIRM 覆盖 (走近/换角度)',
          out3 is not None and calls['ovr'] and calls['ovr'][-1][1] == 'CONFIRM')

    # 高置信 (0.8) 不进身份确认, 直接伺服
    e3 = make_env()
    e3.step = 30
    hi = dict(yolo, confidence=0.8)
    obs_hi = dict(obs, yolo_detection=hi)
    e3._detection_bearing_deg = lambda yi: 5.0
    e3._vlm_identity_check = lambda o, y: (_ for _ in ()).throw(
        AssertionError('high-conf should NOT call identity check'))
    e3._invalidate_lock = lambda r, **kw: None
    e3._override_and_run = lambda o, i, t, n='': ('ret', i)
    out4 = e3._approach_step(obs_hi)
    check('高置信 0.8: 跳过身份确认直接 AUTO-STEER', out4 is not None)


# ---------- ⑤ scan → 探索地图 ----------
def test_scan_map():
    w = make_wrapper()
    upd = {'n': 0}

    class FakeCam:
        def floor_params(self, n):
            return (1.0, 0.2)
        def walkable_rays(self, n):
            return [(a * np.pi / 6, 1.5) for a in range(12)]
    w.depth_cam = FakeCam()
    w.integrate_scan_into_map(['N1', 'N2'])
    check('scan→地图: 地图已初始化', w.exploration_map is not None)
    # 用 spy 包一层 update 验证二次调用也写图
    orig = w.exploration_map.update
    w.exploration_map.update = lambda *a, **k: (upd.__setitem__('n', upd['n'] + 1),
                                                orig(*a, **k))[1]
    w.integrate_scan_into_map(['N1', 'N2'])
    check('scan→地图: 每个扫描节点各更新一次', upd['n'] == 2, f"n={upd['n']}")
    # 坏输入不炸
    w.integrate_scan_into_map(['GHOST_NODE', None])
    check('scan→地图: 缺失节点静默跳过', True)


# ---------- ⑥ 区域级探索分 ----------
def test_area_score():
    w = make_wrapper()
    # N1 朝 +z; C 在 +x (bearing +90°, 同 2m 粗格=老区域, 到访 3 次)
    # B 在 +z (bearing 0°, 同格也是老区域但到访 0 次 → 无惩罚无加分)
    w.visited_area_cells = {(0, 0)}          # 起点格已翻过
    w.visited_nodes = {'C': 3}
    scored = {'recommended': {'rep_deg': 90.0, 'score': 1.0, 'safe_dist_m': 0.6,
                              'confidence': 0.9, 'source': 'graph'},
              'alternatives': [{'rep_deg': 0.0, 'score': 0.9, 'safe_dist_m': 0.6,
                                'confidence': 0.9, 'source': 'graph'}]}
    out = w._area_exploration_adjust(scored, 'N1')
    check('区域分: 翻烂区域推荐 (+90°,v3) 被压制 → 换 0°',
          abs(out['recommended']['rep_deg']) < 1e-6,
          f"rec={out['recommended']['rep_deg']}")
    # 新区域候选: N1→B2? B2 不与 N1 相邻 → 用 C2 演示不相邻不影响匹配
    # 落点在新粗格 → 加分后应胜出
    w2 = make_wrapper()
    w2.visited_area_cells = {(1, 0)}         # C 所在格 (C 在 [0.65,0] → 格(0,0)?) 不对 — C 在 (0,0)
    w2.visited_area_cells = {(0, 0)}
    w2.visited_nodes = {'C': 1, 'B': 1}
    # 把 B 挪到新粗格里测新鲜加分: 直接改 world_pos
    w2.nav_graph['nodes']['B']['world_pos'] = [2.2, 0.0, 0.4]  # 格 (1,0) 新区域
    # 重算 B bearing: delta=[2.2,0,0.4] → bearing≈+80° 近似右; 改成 forward 更直观:
    w2.nav_graph['nodes']['B']['world_pos'] = [0.2, 0.0, 2.2]  # 正前方 2.2m, 格(0,1) 新
    scored2 = {'recommended': {'rep_deg': 90.0, 'score': 1.0, 'safe_dist_m': 0.6,
                               'confidence': 0.9, 'source': 'graph'},
               'alternatives': [{'rep_deg': 0.0, 'score': 0.9, 'safe_dist_m': 2.2,
                                 'confidence': 0.9, 'source': 'graph'}]}
    out2 = w2._area_exploration_adjust(scored2, 'N1')
    check('区域分: 新区域候选 (0°, 正前 2.2m) 加分胜出',
          abs(out2['recommended']['rep_deg']) < 1e-6,
          f"rec={out2['recommended']['rep_deg']}")


# ---------- ⑦ 口袋逃逸 ----------
def test_escape():
    w = make_wrapper()
    # 口袋: N1 的两条平移边 (→C v3, →B v3) 全部 ≥2 次 → 触发逃逸
    w.visited_nodes = {'C': 3, 'B': 3, 'N1': 4, 'C2': 2}
    w.visited_area_cells = {(0, 0)}
    # B2 在格 (1,0) 未到访 → 逃逸目标; 路径 N1→C→C2→B2, 第一跳 C
    esc = w._escape_route()
    check('逃逸: 找到去新区域 (B2) 的路线', esc is not None)
    if esc:
        t1, hops, cost = esc
        check('逃逸: 第一跳是 C (口袋出口方向)', t1 == 'C', f't1={t1}')
        check('逃逸: 跳数 3 (N1→C→C2→B2)', hops == 3, f'hops={hops}')
    # 无新区域 → None
    w2 = make_wrapper()
    w2.visited_nodes = {'C': 3, 'B': 3, 'N1': 4, 'C2': 2, 'B2': 1}
    w2.visited_area_cells = {(0, 0), (1, 0)}
    check('逃逸: 无新区域时返回 None', w2._escape_route() is None)


if __name__ == '__main__':
    test_hysteresis()
    test_oscillation()
    test_deadband()
    test_identity_gate()
    test_scan_map()
    test_area_score()
    test_escape()
    print(f"\n{'='*40}\n✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        print('Failed:', *FAIL, sep='\n  - ')
        sys.exit(1)
