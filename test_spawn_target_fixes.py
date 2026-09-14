"""#30a + #31 出生在目标旁局修复 单测 (2026-09-13, coca-ab 实弹诊断)

出生在目标旁 (softsoap init 0.05m / coca init 0.06m) 的局跨 p16/p26/
bmab 从未成功。coca bmab 全链条定位 (logs/retest_AB_20260913.log):

  warmup 120° 台面可乐瓶 YOLO 61% (scan_02 图实测可见) → APPROACH-LOCK
  → 接近跟丢 (小目标近距检出难) → 六图重扫 VLM 没认出 → 误 SCAN_CLEAR
  → v2 三次强制离区 (step 12/15/21) 拖离真目标 → step23 YOLO 复检
  (撤销阀工作) → fastpath 单票停 area=0.009/conf=0.37 → 米制门读
  _last_obs (替换前的原始 obs) 找不到框 → "no yolo box → FAR" →
  UNREACHABLE 放行 1.98m 停止 → 1.98 ≥ VISUAL_SUCCESS_THRESHOLD 1.0
  (默认档; 2.5 是 single_thresh 对照档) → fp。

修复:
  #30a _last_obs 存 scan 替换后的决策观测 (停票证据与投票证据同源;
       本例中框可见 → 深探 ~2m ∈ (1.0,2.2] 软接近区 → 拒停转接近锁
       走近, 而非 UNREACHABLE 收场)
  #31 有目击史的 zone 不登记 CLEAR (YOLO 目击 > VLM 六图判词;
       zone_clear_sight_guard 消融开关)

运行: python -u test_spawn_target_fixes.py
"""
import sys
sys.path.insert(0, '/home/tao_h/VLMnav/src')
import logging

logging.basicConfig(level=logging.CRITICAL)

PASS, FAIL = [], []


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f'  [{info}]' if info and not cond else ''))


# ========== 功能: #31 CLEAR 目击史守卫 (6 项) ==========
def test_clear_guard():
    from avdb_env import AVDBEnv

    class SW:
        """current_node 站 cur_key 格; 目击记录节点在 sight_key 格"""
        def __init__(self, cur_key, sight_key, sightings):
            self.current_node = 'cur'
            self._ck, self._sk = cur_key, sight_key
            self.target_sighting_nodes = sightings

        def _area_key(self, n):
            return self._ck if n == 'cur' else self._sk

    def mk_env(cur_key, sight_key, sightings, guard=True, inquiry=True):
        e = object.__new__(AVDBEnv)
        e._ff = {'scan_inquiry': inquiry, 'zone_clear_sight_guard': guard}
        e.step = 5
        e.simWrapper = SW(cur_key, sight_key, sightings)
        return e

    SAW = [(3, 'n0', 'YOLO on counter, conf=0.61')]

    # 1 flag 关 (scan_inquiry off) → 不解析
    e = mk_env((1, -2), (1, -2), SAW, inquiry=False)
    ang, note = AVDBEnv._parse_scan_inquiry(e, 'SCAN_CLEAR', 'coca')
    check('#31: scan_inquiry off → 不解析', ang is None and note == '')

    # 2 真空区 (无目击) → 照常登记 (mahatma 6 次 CLEAR 的离区机制不误伤)
    e = mk_env((2, 3), (2, 3), [])
    AVDBEnv._parse_scan_inquiry(e, 'SCAN_CLEAR', 'mahatma_rice')
    check('#31: 无目击史 zone → CLEAR 照常登记 (v2 离区保住)',
          (2, 3) in e._zone_clear_cells)

    # 3 coca 病灶: 同 zone 有目击 → 不登记
    e = mk_env((1, -2), (1, -2), SAW)
    AVDBEnv._parse_scan_inquiry(e, 'SCAN_CLEAR', 'coca')
    check('#31: 有目击史 zone → 不登记 CLEAR (coca 病灶)',
          (1, -2) not in getattr(e, '_zone_clear_cells', set()))

    # 4 目击在别区 → 当前 zone 照常登记
    e = mk_env((5, 5), (1, -2), SAW)
    AVDBEnv._parse_scan_inquiry(e, 'SCAN_CLEAR', 'coca')
    check('#31: 目击在别区 → 当前 zone 照常登记',
          (5, 5) in e._zone_clear_cells)

    # 5 SUSPICIOUS 路径不受守卫影响 (角度照常解析 + 挂可疑点)
    e = mk_env((1, -2), (1, -2), SAW)
    ang, _ = AVDBEnv._parse_scan_inquiry(e, 'SCAN_SUSPICIOUS: 120', 'coca')
    check('#31: SUSPICIOUS 路径不受守卫影响 (角度照常)',
          ang == 120 and (1, -2) in getattr(e, '_suspicious_spots', {}))

    # 6 消融: zone_clear_sight_guard=False → 有目击也登记 (对照行)
    e = mk_env((1, -2), (1, -2), SAW, guard=False)
    AVDBEnv._parse_scan_inquiry(e, 'SCAN_CLEAR', 'coca')
    check('#31: 消融开关 off → 有目击也登记 (对照行)',
          (1, -2) in e._zone_clear_cells)


# ========== 功能: 开关注册 (2 项) ==========
def test_flags():
    from feature_flags import DEFAULTS, FLAG_ARGS, parse_argv, resolve
    check('#31: DEFAULTS 含 zone_clear_sight_guard=True (默认开)',
          DEFAULTS.get('zone_clear_sight_guard') is True)
    check('#31: --no-zone-clear-sight-guard 消融开关注册',
          FLAG_ARGS.get('--no-zone-clear-sight-guard')
          == ('zone_clear_sight_guard', False))
    flags, rest = parse_argv(['--no-zone-clear-sight-guard', 'x'])
    check('#31: parse_argv 剥离 + resolve 默认',
          flags == {'zone_clear_sight_guard': False} and rest == ['x']
          and resolve()['zone_clear_sight_guard'] is True)


# ========== 静态: #30a 证据同源 (源码顺序级) ==========
def test_static_30a():
    ENV_SRC = open('/home/tao_h/VLMnav/src/avdb_env.py').read()
    # _last_obs 赋值必须晚于 scan-obs 替换 (决策观测 = 证据观测)。
    # #49 后 _re_scan 里也有一处 _last_obs = obs (中途重扫同源刷新,
    # 定义在 _step_env 之前) — 用 _step_env 段内那处判顺序。
    seg = ENV_SRC[ENV_SRC.index('def _step_env'):]
    i_sub = seg.index('obs = self._scan_obs')
    i_last = seg.index('self._last_obs = obs')
    check('#30a: _last_obs 存 scan 替换后的决策观测 (顺序)',
          i_last > i_sub)
    check('#30a: 病灶注释锚点在源 (coca 实弹)',
          '#30a' in ENV_SRC and 'coca-ab 实弹' in ENV_SRC)


if __name__ == '__main__':
    test_clear_guard()
    test_flags()
    test_static_30a()
    print('=' * 40)
    print(f"✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        for n in FAIL:
            print(f'  FAIL: {n}')
        sys.exit(1)
