"""#49 桥接配对 close 米制守卫 + 停票证据同源补全 单测 (2026-09-13)

bmv11 实弹 (logs/verify_v11_20260913.log, 29 步 fp@2.42m, #47 已实弹验证
——APPROACH-LOCK world −31° ≈ GT −28.6°, 帧 0° 偏移 −2° 都进了日志):
终局链 — step27 停票 2.86m 被 GT 门正确拒 (bm17 门), step28 rescan 后
YOLO middle_center conf 0.86 真目击 + #47 锁指对方向, 但:
  ① 证据不同源 (#30a 未补全): 中途重扫在 _step_env 入口存 _last_obs
     之后才替换决策 obs → 停票仲裁 _stop_evidence(_last_obs) 看旧图,
     决策帧有框 (area 0.003) 米制门却静默 None (全程零 STOP-EVIDENCE
     行, D2 框门自 verify_v3 后再没开过火);
  ② 桥接配对纯 VLM 放行: ev=None → B1 配对 (目击帧+当前帧并排) 判
     close → HONORED @2.42m — 配对能证 present 证不了 close (bm16
     教训同源); fp 救济线 VISUAL_SUCCESS_THRESHOLD=1.0m 不救 2.42m。
修: _re_scan 后 _last_obs 同步刷新 + _bridge_close_guard (当前帧小框
<2% = present≠close → 拒停转接近) + _stop_evidence 静默 None 全改 INFO。

运行: python -u test_bridge_guard.py
"""
import sys
sys.path.insert(0, '/home/tao_h/VLMnav/src')
sys.path.insert(0, '/home/tao_h/avdb_habitat_converter/scripts')
import logging

logging.basicConfig(level=logging.CRITICAL)

PASS, FAIL = [], []

SRC = open('/home/tao_h/VLMnav/src/avdb_env.py').read()


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f'  [{info}]' if info and not cond else ''))


def mk_env(ff=None, box=None, target_name='mahatma_rice'):
    from avdb_env import AVDBEnv
    e = object.__new__(AVDBEnv)
    e._ff = dict(ff) if ff else {}
    det = None
    if box is not None:
        x1, y1, x2, y2 = box
        det = {'class_name': target_name, 'bbox_norm': [x1, y1, x2, y2]}
    e._last_obs = {'yolo_detection': {
        'target_found': det is not None,
        'all_detections': [det] if det else [],
    }} if det is not None else {'yolo_detection': {}}
    e.simWrapper = type('SW', (), {'target_name': target_name})()
    return e


def test_guard():
    from avdb_env import AVDBEnv

    # 1 bmv11 step28 病例: 小框 0.003 (<2%) → 拦 (present≠close)
    box = (0.48, 0.40, 0.52, 0.46)   # 0.04×0.06 = 0.0024
    e = mk_env({'bridge_metric_guard': True}, box=box)
    check('#49: 小框 0.0024 → 拦下 (verdict 降 far)',
          AVDBEnv._bridge_close_guard(e, 2.42) is False)

    # 2 目视级大框 ≥2% → 放行 (近证据在)
    e2 = mk_env({'bridge_metric_guard': True},
                box=(0.30, 0.30, 0.50, 0.70))   # 0.2×0.4 = 0.08
    check('#49: 大框 0.08 → 放行', AVDBEnv._bridge_close_guard(e2, 0.8) is True)

    # 3 当前帧无框 → 放行 (B1/P0 语义保持 — 配对是唯一证据)
    e3 = mk_env({'bridge_metric_guard': True})
    check('#49: 无框 → 放行 (B1 语义保持)',
          AVDBEnv._bridge_close_guard(e3, 2.0) is True)

    # 4 消融 --no-bridge-metric-guard → 放行 (旧行为)
    e4 = mk_env({'bridge_metric_guard': False}, box=(0.48, 0.40, 0.52, 0.46))
    check('#49: 消融 flag off → 放行 (旧行为)',
          AVDBEnv._bridge_close_guard(e4, 2.42) is True)

    # 5 _ff 缺失 → 默认开 (小框仍拦)
    e5 = mk_env()
    e5._last_obs = {'yolo_detection': {
        'target_found': True,
        'all_detections': [{'class_name': 'mahatma_rice',
                            'bbox_norm': [0.48, 0.40, 0.52, 0.46]}]}}
    check('#49: _ff 缺失 → 默认守卫开',
          AVDBEnv._bridge_close_guard(e5, 2.42) is False)

    # 6 框无 bbox_norm / 类名不符 → 放行 (保持旧行为, 不误杀)
    e6 = mk_env()
    e6._last_obs = {'yolo_detection': {
        'target_found': True,
        'all_detections': [{'class_name': 'other_thing'}]}}
    check('#49: 无目标框坐标 → 放行 (不误杀)',
          AVDBEnv._bridge_close_guard(e6, 2.0) is True)


def test_wiring():
    import feature_flags as F

    # 7 flag 注册 + CLI 剥离
    flags, rest = F.parse_argv(['--no-bridge-metric-guard', 'x'])
    check('#49: DEFAULTS True + --no-bridge-metric-guard 注册剥离',
          F.DEFAULTS.get('bridge_metric_guard') is True
          and F.FLAG_ARGS.get('--no-bridge-metric-guard')
          == ('bridge_metric_guard', False)
          and flags == {'bridge_metric_guard': False} and rest == ['x'])

    # 8 仲裁接线: 配对 close 须过守卫
    check('#49: 仲裁接线 — _bridge_close_guard 在配对 close 后调用',
          'and not self._bridge_close_guard(distance)' in SRC)

    # 9 同源补全: _re_scan 刷新 _last_obs 在 _scan_obs 之后 (#30a 补全)
    seg = SRC[SRC.index('def _re_scan'):SRC.index('def _vlm_grounding_detect')]
    check('#49: _re_scan 同源刷新 — _last_obs = obs 在 _scan_obs 之后',
          'self._scan_obs = obs' in seg
          and 'self._last_obs = obs' in seg
          and seg.index('self._last_obs = obs') > seg.index('self._scan_obs = obs'))

    # 10 静默 None 可审计: 三条 INFO 理由都在
    check('#49: _stop_evidence 静默 None 三路径全改 INFO',
          'decision obs has no target box' in SRC
          and 'depth samples' in SRC
          and 'no depth_sensor' in SRC
          and 'depth probe exception' in SRC)

    # 11 BRIDGE-ARRIVAL 日志不再预支 "honor" (先候选后守卫)
    check('#49: 配对日志措辞 — candidate honor (guard pending)',
          'candidate honor' in SRC and 'guard pending' in SRC)


if __name__ == '__main__':
    test_guard()
    test_wiring()
    print('=' * 40)
    print(f"✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        for n in FAIL:
            print(f'  FAIL: {n}')
        sys.exit(1)
