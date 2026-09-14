"""#50 正修: 小框 bbox-grid 深度隔离 单测 (2026-09-13, d03r fp@5.84m)

三件套实弹证据 (小框深度双向都错):
  bmv12 step17/19: area 0.003 九点全无效 → None (饿死, 保守方向错)
  d01r step40:     area 0.003 探到 0.42m 近探针 → 误 FAR "probe occluded"
  d03r step25:     area 0.004 探到 0.53m (框外近处地板) → 误 CLOSE
                   放行 fp @5.84m — 0.53 恰好溜过 0.5m 矛盾崖,
                   视差 (框刚出现无前帧) / 前向带 (bottom_right 侧向)
                   双哑, #49b 桥接守卫只挂桥接路径没挂 proximity 路径

修: verdict==close 且 area<0.005 → 须独立源佐证 (视差≤1.2 /
    支撑面≤1.0 / 前向带≤1.0 且目标在视轴带), 无佐证 → FAR 拒停转接近
    (走近框变大: ≥2% 面积捷径或佐证 CLOSE 自然到)。bmv12 成功停
    area 0.020 走面积捷径, 不受影响。

运行: python -u test_small_box_quarantine.py
"""
import sys
sys.path.insert(0, '/home/tao_h/VLMnav/src')
import logging
import numpy as np

logging.basicConfig(level=logging.CRITICAL)

PASS, FAIL = [], []

SRC = open('/home/tao_h/VLMnav/src/avdb_env.py').read()


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f'  [{info}]' if info and not cond else ''))


def mk_env(ff=None, parallax=None, support=None, fband=None):
    from avdb_env import AVDBEnv
    e = object.__new__(AVDBEnv)
    e._ff = dict(ff) if ff else {}

    class SW:
        target_name = 'nutrigrain_harvest_blueberry_bliss'
    e.simWrapper = SW()
    # 独立源 mock: None = 该源无证据
    e._parallax_distance = lambda: parallax
    e._support_depth = lambda *a, **k: support
    e._forward_band_depth = lambda *a, **k: fband
    return e


def mk_obs(depth_m=0.53, x1=0.68, y1=0.75, x2=0.73, y2=0.83):
    """小框 (area 0.005×0.08=0.004, cx≈0.705 在前向带适用域内)"""
    d = np.full((480, 640), depth_m, dtype=np.float32)
    return {
        'depth_sensor': d,
        'yolo_detection': {
            'target_found': True, 'confidence': 0.70,
            'position': 'bottom_right',
            'all_detections': [{
                'class_name': 'nutrigrain_harvest_blueberry_bliss',
                'confidence': 0.70, 'bbox_norm': [x1, y1, x2, y2]}]},
    }


def test_flags():
    import feature_flags as F
    check('#50: DEFAULTS small_box_quarantine=True',
          F.DEFAULTS.get('small_box_quarantine') is True)
    flags, rest = F.parse_argv(['--no-small-box-quarantine', 'x'])
    check('#50: --no-small-box-quarantine 注册剥离',
          F.FLAG_ARGS.get('--no-small-box-quarantine')
          == ('small_box_quarantine', False)
          and flags == {'small_box_quarantine': False} and rest == ['x'])


def test_quarantine():
    from avdb_env import AVDBEnv
    # d03r step25 病例复现: 网格 0.53m 小框, 三独立源全哑 → 隔离 FAR
    e = mk_env()
    obs = mk_obs(0.53)
    r = AVDBEnv._stop_evidence(e, obs)
    check('#50: d03r 病例 — 小框 0.53m 无佐证 → FAR (拒停转接近)',
          r is not None and r[0] == 'far', f'r={r and r[0]}')

    # 前向带在适用域 (cx 0.705) 且说远 (3m) → 仍无佐证 → FAR
    e2 = mk_env(fband=3.0)
    r2 = AVDBEnv._stop_evidence(e2, mk_obs(0.53))
    check('#50: 前向带 3m (>1.0) 不构成佐证 → FAR',
          r2 is not None and r2[0] == 'far')

    # 视差佐证 0.9m → CLOSE 保持 (真近小框不被误杀)
    e3 = mk_env(parallax=0.9)
    r3 = AVDBEnv._stop_evidence(e3, mk_obs(0.53))
    check('#50: 视差 0.9m 佐证 → CLOSE (真近小框放行)',
          r3 is not None and r3[0] == 'close', f'r={r3 and r3[0]}')

    # 支撑面佐证 0.8m → CLOSE 保持
    e4 = mk_env(support=0.8)
    r4 = AVDBEnv._stop_evidence(e4, mk_obs(0.53))
    check('#50: 支撑面 0.8m 佐证 → CLOSE',
          r4 is not None and r4[0] == 'close', f'r={r4 and r4[0]}')

    # 前向带佐证 0.8m (目标在视轴带内) → CLOSE 保持
    e5 = mk_env(fband=0.8)
    r5 = AVDBEnv._stop_evidence(e5, mk_obs(0.53))
    check('#50: 前向带 0.8m 佐证 → CLOSE',
          r5 is not None and r5[0] == 'close', f'r={r5 and r5[0]}')

    # 大框 (area 0.0225 ≥2%) 面积捷径 → CLOSE, 不进隔离 (bmv12 成功停路径)
    e6 = mk_env()
    big = mk_obs(0.53, x1=0.30, y1=0.30, x2=0.45, y2=0.45)  # 0.15×0.15
    r6 = AVDBEnv._stop_evidence(e6, big)
    check('#50: 大框 ≥2% 面积捷径不受隔离影响 (bmv12 成功停保护)',
          r6 is not None and r6[0] == 'close')

    # d01r mismatch 病例保持: 小框 0.42m (<0.5 矛盾崖) → FAR (原行为)
    e7 = mk_env(parallax=0.9)   # 即使有佐证, mismatch 崖先行
    r7 = AVDBEnv._stop_evidence(e7, mk_obs(0.42))
    check('#50: mismatch 崖 (0.42m) 行为保持 → FAR',
          r7 is not None and r7[0] == 'far')

    # 消融 off → 旧行为: 0.53m 小框 CLOSE (d03r fp 复现, 供对照)
    e8 = mk_env({'small_box_quarantine': False})
    r8 = AVDBEnv._stop_evidence(e8, mk_obs(0.53))
    check('#50: 消融 off → 旧行为 CLOSE (d03r fp 病理保留对照)',
          r8 is not None and r8[0] == 'close', f'r={r8 and r8[0]}')


def test_static():
    check('#50: 常量 SMALL_BOX_QUARANTINE_AREA=0.005 在源',
          'SMALL_BOX_QUARANTINE_AREA = 0.005' in SRC)
    check('#50: 隔离块接线 (verdict far + #50 日志串)',
          "approach, not '\n                                f'stop (#50)'" in SRC
          and 'small_box_quarantine' in SRC)


if __name__ == '__main__':
    test_flags()
    test_quarantine()
    test_static()
    print('=' * 40)
    print(f"✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        for n in FAIL:
            print(f'  FAIL: {n}')
        sys.exit(1)
