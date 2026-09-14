"""#47 扫描目击 APPROACH-LOCK 世界方位双修 单测 (2026-09-13)

bmv10 实弹 (logs/verify_v10_20260913.log, 52 步 3.99m max_steps, 全程
目击链全真): 两次扫描目击 APPROACH-LOCK 世界方位 180° 级全错 → 伺服
反向 → miss → TWO-STRIKES 拉黑真目标区 (与 bmv8 同终局家族)。
GT 离线取证 (val ep0 mahatma, 目标 world (−3.4362, 5.0731)):
  ① step38b: 扫描起始 001600101 yaw −28.3°, 锁 153° — 锁前 rescan 取
     obs 的途中 null 步把 current_node 偷换成同位置朝向变体 001660101
     (yaw 153.2°, _find_closest_node 位置全平局取 dict 首键) + sim 相机
     被物理转到该任意朝向 (A1 对齐被中途误用), 错 181.6°;
  ② step42: 锁 304° (123.9°+180°, base 本身没错) — 漏画面内框偏移:
     目击在 middle_right, GT 相对相机 +41.1°, 真 344.9°≡−15.1°。
修复: ① wrapper 途中 null 纯读 (重定位+A1 只在开局); ② 锁公式 =
扫描起始朝向 + 扫描角 + 目标框 cx 反解偏移 (照片 HFOV≈89°)。
真实图回放: 两病例修正后误差 ≤1.5°。

运行: python -u test_lock_bearing.py
"""
import sys
sys.path.insert(0, '/home/tao_h/VLMnav/src')
sys.path.insert(0, '/home/tao_h/avdb_habitat_converter/scripts')
import logging

logging.basicConfig(level=logging.CRITICAL)

PASS, FAIL = [], []

WRAP = '/home/tao_h/avdb_habitat_converter/scripts/avdb_sim_wrapper.py'
SNAP = '/home/tao_h/VLMnav/任务/wrapper_patch/avdb_sim_wrapper.snapshot.py'
GRAPH = '/home/tao_h/avdb_habitat_converter/data/processed/navigation_graph.json'
GT = (-3.4362, 5.0731)   # ep0 mahatma 目标 (离线诊断 oracle, 不进系统决策)


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f'  [{info}]' if info and not cond else ''))


def mk_env(ff=None, node=None, nodes=None):
    from avdb_env import AVDBEnv
    e = object.__new__(AVDBEnv)
    e._ff = dict(ff) if ff else {}
    e.simWrapper = type('SW', (), {
        'nav_graph': {'nodes': nodes or {}, 'graph': {}},
        'current_node': node})()
    return e


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def test_frame_offset():
    from avdb_env import AVDBEnv
    e = mk_env()
    t = 'mahatma_rice'
    det = lambda cx: {'all_detections': [
        {'class_name': 'other', 'center_x': 0.9},
        {'class_name': t, 'center_x': cx}]}

    # 1 bmv10 step42: GT 相对相机 +41.1° → cx 0.96 反解 ≈ +41°
    off = AVDBEnv._yolo_frame_offset_deg(e, det(0.96), t)
    check('#47b: cx 0.96 → +40.9° (bmv10 step42 GT 取证值 ±41°)',
          off is not None and abs(off - 40.94) < 0.5, f'{off}')

    # 2 左半画面 → 负偏移
    off2 = AVDBEnv._yolo_frame_offset_deg(e, det(0.2), t)
    check('#47b: cx 0.2 → −26.7° (左负)', abs(off2 + 26.7) < 0.5, f'{off2}')

    # 3 夹到照片半视界 ±44.5° (框中心不可能出画)
    off3 = AVDBEnv._yolo_frame_offset_deg(e, det(0.0), t)
    check('#47b: cx 0.0 → 夹 −44.5°', abs(off3 + 44.5) < 0.01, f'{off3}')

    # 4 无目标类框 → 顶层 center_x 兜底; 全无 → None (caller 回落 0)
    off4 = AVDBEnv._yolo_frame_offset_deg(
        e, {'all_detections': [{'class_name': 'x', 'center_x': 0.9}],
            'center_x': 0.7}, t)
    off5 = AVDBEnv._yolo_frame_offset_deg(e, {'all_detections': []}, t)
    off6 = AVDBEnv._yolo_frame_offset_deg(e, None, t)
    check('#47b: 顶层 center_x 兜底 +0.17.8°→0.7→+17.8', abs(off4 - 17.8) < 0.5,
          f'{off4}')
    check('#47b: 无框信息 → None (回落旧行为)', off5 is None and off6 is None)


def test_lock_formula():
    from avdb_env import AVDBEnv
    import numpy as np
    # bmv10 step42 病例: 起始 6830101 yaw 123.9°, 帧 180°, 偏移 +41°
    e = mk_env({'lock_bearing_fix': True})
    e._scan_base_yaw_deg = 123.9
    e._scan_hit_offset_deg = 41.0
    b = np.degrees(AVDBEnv._scan_lock_bearing_rad(e, 180))
    # GT −13.8° ≡ 346.2°; 修后 344.9° — 差 ≤3° (旧 303.9° 差 42°)
    check('#47: step42 回放 — 修后锁 344.9° vs GT 346.2° (≤3°)',
          abs(wrap180(b - 346.2)) <= 3.0, f'{b:.1f}')

    # bmv10 step38b 病例: 起始 001600101 yaw −28.3°, 帧 0°, 偏移 ≈0
    e2 = mk_env({'lock_bearing_fix': True})
    e2._scan_base_yaw_deg = -28.3
    e2._scan_hit_offset_deg = -0.4
    b2 = np.degrees(AVDBEnv._scan_lock_bearing_rad(e2, 0))
    check('#47: step38b 回放 — 修后锁 −28.7° vs GT −28.6° (≤3°)',
          abs(wrap180(b2 - (-28.6))) <= 3.0, f'{b2:.1f}')

    # 消融: flag off → 锁时刻 _yaw_world() + 无偏移 (精确旧公式)
    e3 = mk_env({'lock_bearing_fix': False},
                node='N', nodes={'N': {'direction': [0.5, 0, 0.866]}})  # yaw 30°
    e3._scan_base_yaw_deg = 123.9      # 应被忽略
    e3._scan_hit_offset_deg = 41.0     # 应被忽略
    b3 = np.degrees(AVDBEnv._scan_lock_bearing_rad(e3, 90))
    check('#47: 消融 flag off → 旧公式 (实时 yaw 30° + 90° = 120°)',
          abs(wrap180(b3 - 120.0)) < 0.1, f'{b3:.1f}')

    # _ff 缺失 → 默认开 (scan 基准 + 偏移生效)
    e4 = mk_env()
    e4._scan_base_yaw_deg = 0.0
    e4._scan_hit_offset_deg = 30.0
    b4 = np.degrees(AVDBEnv._scan_lock_bearing_rad(e4, 60))
    check('#47: _ff 缺失 → 默认修复开 (0+60+30 = 90°)', abs(wrap180(b4 - 90)) < 0.1,
          f'{b4:.1f}')

    # 偏移 None → 0 (无框回落)
    e5 = mk_env({'lock_bearing_fix': True})
    e5._scan_base_yaw_deg = 10.0
    e5._scan_hit_offset_deg = None
    b5 = np.degrees(AVDBEnv._scan_lock_bearing_rad(e5, 20))
    check('#47: 偏移 None → 0 (无框回落旧行为)', abs(wrap180(b5 - 30)) < 0.1, f'{b5:.1f}')


def test_real_graph_replay():
    """真实导航图回放 bmv10 两病例 + 走链符号验证 (取证实锤)"""
    import json
    import os
    import numpy as np
    from avdb_env import AVDBEnv
    if not os.path.exists(GRAPH):
        check('#47: 真实图回放 (跳过: 图文件不在本机)', True)
        return
    g = json.load(open(GRAPH))
    nodes, graph = g['nodes'], g['graph']

    def yaw(n):
        d = nodes[n]['direction']
        return float(np.degrees(np.arctan2(d[0], d[2])))

    def gt_brg(n):
        p = nodes[n]['world_pos']
        return float(np.degrees(np.arctan2(GT[0] - p[0], GT[1] - p[2])))

    e = object.__new__(AVDBEnv)
    e._ff = {'lock_bearing_fix': True}

    # 11 走链符号: 扫描走 rotate_cw, 帧 180° 相机朝向 = 起始 +180°
    #    (bmv10 step42: 起始 6830101 → 帧180 节点 6770101, 实测验证)
    cur = '000110006830101.jpg'
    for _ in range(6):
        nxt = next((t for t, ed in graph.get(cur, {}).items()
                    if ed['edge_type'] == 'rotate_cw'), None)
        if not nxt:
            break
        cur = nxt
    cam_head = yaw(cur)
    check('#47: 真实图走链 — 帧180 相机朝向 = 起始+180° (rotate_cw=+30°/跳)',
          abs(wrap180(cam_head - (yaw('000110006830101.jpg') + 180.0))) <= 5.0
          and abs(wrap180(cam_head - yaw('000110006770101.jpg'))) <= 5.0,
          f'cam={cam_head:.1f} origin+180={yaw("000110006830101.jpg")+180:.1f}')

    # 12 step38b: base=001600101, 帧 0 — 旧病理 (null 偷换到 001660101)
    #    错 ≥178°, 修后 ≤3°
    e._scan_base_yaw_deg = yaw('000110001600101.jpg')
    e._scan_hit_offset_deg = -0.4     # GT 相对帧0 相机 −0.4° (≈画面正中)
    new = np.degrees(AVDBEnv._scan_lock_bearing_rad(e, 0))
    old = yaw('000110001660101.jpg') + 0.0   # 旧公式: null 步偷换后的 yaw
    gt38 = gt_brg('000110001600101.jpg')
    check('#47: 真实图 step38b — 旧锁错 ~181.6° (病理复现)',
          abs(wrap180(old - gt38)) >= 178.0, f'old={old:.1f} gt={gt38:.1f}')
    check('#47: 真实图 step38b — 修后 ≤3°',
          abs(wrap180(new - gt38)) <= 3.0, f'new={new:.1f} gt={gt38:.1f}')

    # 13 step42: base=6830101, 帧 180, 偏移 +41 (GT 相对帧180 相机 +41.1°)
    e._scan_base_yaw_deg = yaw('000110006830101.jpg')
    e._scan_hit_offset_deg = 41.0
    new2 = np.degrees(AVDBEnv._scan_lock_bearing_rad(e, 180))
    old2 = yaw('000110006830101.jpg') + 180.0   # 旧公式: base 对但漏偏移
    gt42 = gt_brg('000110006770101.jpg')
    check('#47: 真实图 step42 — 旧锁 (漏偏移) 错 ~42° (病理复现)',
          abs(wrap180(old2 - gt42)) >= 40.0, f'old={old2:.1f} gt={gt42:.1f}')
    check('#47: 真实图 step42 — 修后 ≤3°',
          abs(wrap180(new2 - gt42)) <= 3.0, f'new={new2:.1f} gt={gt42:.1f}')


def test_wiring_and_flags():
    import hashlib
    import feature_flags as F
    SRC = open('/home/tao_h/VLMnav/src/avdb_env.py').read()
    WSRC = open(WRAP).read()

    # 14 flag 注册 + CLI 剥离
    flags, rest = F.parse_argv(['--no-lock-bearing-fix', 'x'])
    check('#47: DEFAULTS True + --no-lock-bearing-fix 注册剥离',
          F.DEFAULTS.get('lock_bearing_fix') is True
          and F.FLAG_ARGS.get('--no-lock-bearing-fix') == ('lock_bearing_fix', False)
          and flags == {'lock_bearing_fix': False} and rest == ['x'])

    # 15 env 接线: wrapper 属性注入 + 两条锁路径都走共享 helper + 瞄准角含偏移
    check('#47: env 接线 — simWrapper.lock_bearing_fix 注入',
          'self.simWrapper.lock_bearing_fix = bool(' in SRC)
    check('#47: env 接线 — rescan+warmup 两锁共用 _scan_lock_bearing_rad',
          SRC.count('self._scan_lock_bearing_rad(') == 2)
    check('#47: env 接线 — 扫描入口记基准朝向/重置偏移',
          '_scan_base_yaw_deg = float(np.degrees(self._yaw_world()))'
          in SRC.split('def _warmup_scan')[1][:2000]
          and 'self._scan_hit_offset_deg = None'
          in SRC.split('def _warmup_scan')[1][:2000])

    # 16 wrapper: 途中 null 纯读分支 + 开局路径保持
    check('#47: wrapper — 途中 null 纯读分支 (lock_bearing_fix 门)',
          "if self.current_node and getattr(self, 'lock_bearing_fix', True):" in WSRC
          and 'obs = self.sim.get_sensor_observations(0)' in WSRC)
    null_seg = WSRC.split('if action is PolarAction.null:')[1].split(
        'if not self.current_node:')[0]
    check('#47: wrapper — null 分支内 _record_visit 只剩开局路径 (途中不灌 visit)',
          null_seg.count('_record_visit') == 1
          and null_seg.index('_record_visit') > null_seg.index(
              "if self.current_node and getattr(self, 'lock_bearing_fix', True):"))

    # 17 快照与线上一致 + manifest SHA 对账
    h1 = hashlib.sha256(open(WRAP, 'rb').read()).hexdigest()
    h2 = hashlib.sha256(open(SNAP, 'rb').read()).hexdigest()
    man = open('/home/tao_h/VLMnav/任务/wrapper_patch/manifest.md').read()
    check('#47: 快照=线上 wrapper + manifest SHA 对账 (批次C)',
          h1 == h2 and h1 in man)


if __name__ == '__main__':
    test_frame_offset()
    test_lock_formula()
    test_real_graph_replay()
    test_wiring_and_flags()
    print('=' * 40)
    print(f"✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        for n in FAIL:
            print(f'  FAIL: {n}')
        sys.exit(1)
