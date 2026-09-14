"""扫描机制 v2 + #29 强制探索闩死 单测 (2026-09-13)

扫描机制 v2 (用户指令三连):
  ① 困惑即原地 360° 扫 (≥2/4 零样本信号, 不等整 10 步)
  ② 周期扫描降级为保底计时 (距上次任意扫描 ≥10 步)
  ③ SCAN_CLEAR = 无价值区登记 → 机制强制第一时间离区
    (YOLO 活检证据撤销)
#29: "20 steps done=0" 触发条件逐步闩死 — step20 后每步强制,
  VLM 决策被旁路 30 步 (mahatma-ab 实弹) → 一次性踢 + 5 步冷却。

事故锚点 (logs/retest_AB_20260913.log, mahatma ep):
  step20-49 全部 "Forced exploration (20 steps done=0)" 跟 depth
  rec -5deg 在同区蹭; 6 次扫描 CLEAR 无一被执行。

运行: python -u test_scan_v2.py
"""
import sys
sys.path.insert(0, '/home/tao_h/VLMnav/src')
import logging

logging.basicConfig(level=logging.CRITICAL)

PASS, FAIL = [], []


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f'  [{info}]' if info and not cond else ''))


# ========== 功能: _in_cleared_zone (5 项) ==========
def test_in_cleared_zone():
    from avdb_env import AVDBEnv
    env = object.__new__(AVDBEnv)

    class SW:
        def __init__(self, key):
            self.current_node = 'n'
            self._k = key

        def _area_key(self, n):
            return self._k

    # 1 无登记 → False (没扫过 CLEAR 的 zone 永不触发离区)
    env._zone_clear_cells = set()
    env.simWrapper = SW((1, 0))
    check('v2③: 无 CLEAR 登记不触发',
          AVDBEnv._in_cleared_zone(env, {}) is False)

    # 2 站在已判 CLEAR zone → True
    env._zone_clear_cells = {(1, 0)}
    check('v2③: 当前 zone 已判 CLEAR → True (强制离区依据)',
          AVDBEnv._in_cleared_zone(env, {}) is True)

    # 3 YOLO 活检 → 撤销登记 + 返回 False (新证据 > 旧判词)
    env._zone_clear_cells = {(1, 0)}
    r = AVDBEnv._in_cleared_zone(
        env, {'yolo_detection': {'target_found': True}})
    check('v2③: YOLO 活检撤销 CLEAR (证据可撤销, 恢复追击)',
          r is False and (1, 0) not in env._zone_clear_cells)

    # 4 其他 zone 的登记不影响当前
    env._zone_clear_cells = {(2, 3)}
    check('v2③: 只看当前 zone, 别区 CLEAR 不误伤',
          AVDBEnv._in_cleared_zone(env, {}) is False)

    # 5 wrapper 异常 → 安全 False
    env.simWrapper = None
    check('v2③: wrapper 异常安全返回 False',
          AVDBEnv._in_cleared_zone(env, {}) is False)


# ========== 功能: #29 _zero_done_streak (4 项) ==========
def test_zero_streak():
    from agent import ObjectNavAgent
    ag = object.__new__(ObjectNavAgent)

    ag.stop_history = [False] * 10
    check('#29: <20 票不成立', ag._zero_done_streak() is False)

    ag.stop_history = [False] * 20
    check('#29: 20 票全 done=0 → 成立', ag._zero_done_streak() is True)

    ag.stop_history = [False] * 19 + [True]
    check('#29: 最近 20 票内有目击 → 不成立',
          ag._zero_done_streak() is False)

    ag.stop_history = [True] + [False] * 30   # 更早目击不算
    check('#29: 只看最近 20 票 (旧目击过期)', ag._zero_done_streak() is True)


# ========== 功能: zone_clear_exit 消融开关 (2 项) ==========
def test_flag():
    from feature_flags import DEFAULTS, FLAG_ARGS, parse_argv, resolve
    check('v2③: DEFAULTS 含 zone_clear_exit=True (默认开)',
          DEFAULTS.get('zone_clear_exit') is True)
    check('v2③: --no-zone-clear-exit 消融开关注册',
          FLAG_ARGS.get('--no-zone-clear-exit') == ('zone_clear_exit', False))
    flags, rest = parse_argv(['--no-zone-clear-exit', 'foo'])
    check('v2③: parse_argv 剥离开关且位置参数保留',
          flags == {'zone_clear_exit': False} and rest == ['foo'])
    check('v2③: resolve 默认 True',
          resolve()['zone_clear_exit'] is True)


# ========== 静态断言 (源码级, 8 项) ==========
def test_static():
    ENV_SRC = open('/home/tao_h/VLMnav/src/avdb_env.py').read()
    AGENT_SRC = open('/home/tao_h/VLMnav/src/agent.py').read()
    FF_SRC = open('/home/tao_h/VLMnav/src/feature_flags.py').read()
    UTILS_SRC = open('/home/tao_h/VLMnav/src/utils.py').read()

    # v2① 困惑即扫: ≥2 信号 + 冷却 4 步 + 预算 4
    check('v2①: 困惑触发 (≥2/4 零样本信号, 预算 ≤4)',
          'Triggered: CONFUSION' in ENV_SRC
          and 'len(sig) >= 2' in ENV_SRC
          and '_confusion_rescans < 4' in ENV_SRC)
    check('v2①: 困惑信号含 vlm 连选 x3 / 零检出≥6 / 无新方向 / 已扫过',
          'vlm repeat choice x3' in ENV_SRC
          and 'no-detect' in ENV_SRC and 'no fresh direction' in ENV_SRC
          and 'zone already 360-scanned' in ENV_SRC)
    check('v2①: 冷却 4 步 (扫描后不连发)',
          "- getattr(self, '_last_scan_step', 0) >= 4" in ENV_SRC)

    # v2② 周期保底计时: 任意扫描重置, 不死等整 10 倍数
    check('v2②: 周期保底 = 距上次任意扫描 ≥10 步',
          "self.step - getattr(self, '_last_scan_step', 0) >= 10" in ENV_SRC
          and 'self._last_scan_step = step_num' in ENV_SRC)

    # v2③ CLEAR 登记 + 强制离区 + 证据撤销
    check('v2③: SCAN_CLEAR → _zone_clear_cells 登记 + 无价值区日志',
          '_zone_clear_cells.add(cell)' in ENV_SRC
          and '无价值区, 将强制离区' in ENV_SRC)
    check('v2③: 离区块查 flag 且最高优先 (前沿跳跃离区)',
          "'zone_clear_exit', True)" in ENV_SRC
          and '当前 zone 已判' in ENV_SRC
          and '_plan_frontier_hop()' in ENV_SRC)

    # #29 一次性踢 + 5 步冷却
    check('#29: 20-done=0 改一次性踢 (5 步冷却 + 锚点重置)',
          'kick #29' in AGENT_SRC
          and "_last_zero_force_step', -99) >= 5" in AGENT_SRC
          and 'self._last_zero_force_step = -99' in AGENT_SRC)
    check('#29: 门槛独立成方法 (_zero_done_streak)',
          'def _zero_done_streak' in AGENT_SRC
          and 'if self._zero_done_streak()' in AGENT_SRC)

    # GIF 合成: chosen + 俯视深度图同帧
    check('GIF: chosen 与 topdown_depth_map 同帧合成',
          'topdown_depth_map.png' in UTILS_SRC
          and 'np.hstack([chosen, sep, depth])' in UTILS_SRC)


if __name__ == '__main__':
    test_in_cleared_zone()
    test_zero_streak()
    test_flag()
    test_static()
    print('=' * 40)
    print(f"✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        for n in FAIL:
            print(f'  FAIL: {n}')
        sys.exit(1)
