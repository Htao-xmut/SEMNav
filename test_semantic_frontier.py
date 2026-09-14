"""#32 语义前沿 + 已清区排除 单测 (2026-09-13, bmv2 mahatma 实弹)

用户指令: "为什么vlm不去更可能放目标物品的方位去再看看呢？？既然困惑
就去呗！一直在同一个区域确认也没啥意义"

bmv2 mahatma 实弹 (logs/verify_v2_20260913.log): 机制标记全发 (4 困惑扫
/ 3 强制离区 / 6 前沿跳 / 7 CLEAR) 但仍 51 步 max_steps 终距 10.46m —
病灶是 `_plan_frontier_hop` 纯几何"最近灰格":
  ① 最近前沿永远 2.5m 外的隔壁 zone → 跳出又进, zone (-1,-2) 被
    CLEAR 两次 (line 316/580), 全程困同一功能区;
  ② 六图分析只问扫尽没, 不产"哪个方向更通向目标常在区域"的探索去向。

修复 (#32):
  - 扫描 prompt 第 5 问 → LIKELY-BEARING:<angle>° (探索偏好, 非目击)
  - _parse_scan_inquiry 解析 → _semantic_bearing_deg (世界方位 =
    yaw + scan 角, 与 APPROACH-LOCK 同换算)
  - _pick_frontier_target: 落点 zone 已判 CLEAR → 出池; 语义方位
    ±90° 半平面内优先取最近前沿; 全排除 → 回退全局最近 (不死锁)。
    ep6 教训保持: 只排序探索落点, 不作转向命令/不加 bonus。

运行: python -u test_semantic_frontier.py
"""
import sys
sys.path.insert(0, '/home/tao_h/VLMnav/src')
import logging
import numpy as np

logging.basicConfig(level=logging.CRITICAL)

PASS, FAIL = [], []


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f'  [{info}]' if info and not cond else ''))


class Em:
    """exploration_map 桩: 一列前沿格, 格号 → 世界坐标 (z=格号*1.0)"""
    def __init__(self, world_pts):
        self.pts = world_pts
        self.regions = np.zeros((len(world_pts), 2), dtype=np.int8)
        self.regions[:, 0] = 2          # 每行一个前沿格

    def _to_world(self, gz, gx):
        return self.pts[gz]


class SW:
    AREA_CELL = 2.0

    def __init__(self, cur=(0.0, 0.0)):
        self.current_node = 'n'
        self._p = cur

    def _node_pos(self, n):
        return self._p


def mk_env(frontiers, cur=(0.0, 0.0)):
    from avdb_env import AVDBEnv
    e = object.__new__(AVDBEnv)
    e._ff = {'semantic_frontier': True}
    e._zone_clear_cells = set()
    e.simWrapper = SW(cur)
    e._em = Em(frontiers)
    return e


# ========== 功能: _pick_frontier_target (5 项) ==========
def test_pick():
    from avdb_env import AVDBEnv

    # 1 旧行为保底: 无语义无已清区 → 全局最近
    e = mk_env([(0, 3.0), (0, 6.0)])            # (wx, wz): 3m / 6m 远
    r = AVDBEnv._pick_frontier_target(e, (0.0, 0.0), np.argwhere(
        e._em.regions == 2), e._em, 2.5, 10.0)
    check('#32: 无语义/无已清区 → 最近前沿 (旧行为)',
          r is not None and abs(r[2] - 3.0) < 1e-6 and r[3] == 'nearest')

    # 2 已清区排除: 3m 处落点在已清 zone → 选 6m 处的
    e = mk_env([(0, 3.0), (0, 6.0)])
    e._zone_clear_cells = {(0, 1)}              # zone of (0,3.0) = (0,1)
    r = AVDBEnv._pick_frontier_target(e, (0.0, 0.0), np.argwhere(
        e._em.regions == 2), e._em, 2.5, 10.0)
    check('#32: 最近前沿在已清 zone → 排除选远的 (bmv2 (-1,-2) 二次 CLEAR 病灶)',
          r is not None and abs(r[2] - 6.0) < 1e-6 and r[3] == 'cleared-filtered')

    # 3 语义偏好: 前方 6m (对齐 0°) vs 后左 5.15m (≈-125°, 半平面外) → 选对齐的
    e = mk_env([(-4.2, -3.0), (0, 6.0)])        # 后左 ~5.2m / 前 6m
    e._semantic_bearing_deg = 0.0               # 世界 0° = +z 方向
    e._semantic_votes = [(0, 0.0)]             # #38 聚合通道同源票
    r = AVDBEnv._pick_frontier_target(e, (0.0, 0.0), np.argwhere(
        e._em.regions == 2), e._em, 2.5, 10.0)
    check('#32: 语义方位 ±90° 内优先 (稍远也对齐优先)',
          r is not None and abs(r[2] - 6.0) < 1e-6 and r[3] == 'semantic')

    # 4 语义但无对齐候选 → 回退最近 (不因语义死锁)
    e = mk_env([(-3.0, 0.0)])                   # 只有左侧
    e._semantic_bearing_deg = 0.0
    e._semantic_votes = [(0, 0.0)]
    r = AVDBEnv._pick_frontier_target(e, (0.0, 0.0), np.argwhere(
        e._em.regions == 2), e._em, 2.5, 10.0)
    check('#32: 语义无对齐候选 → 回退最近 (不死锁)',
          r is not None and abs(r[2] - 3.0) < 1e-6)

    # 5 全部已清 → 回退全局最近 (探索不因排除停摆)
    e = mk_env([(0, 3.0), (0, 6.0)])
    e._zone_clear_cells = {(0, 1), (0, 3)}      # 两个落点 zone 全清
    r = AVDBEnv._pick_frontier_target(e, (0.0, 0.0), np.argwhere(
        e._em.regions == 2), e._em, 2.5, 10.0)
    check('#32: 候选全在已清 zone → 回退最近 (不停摆)',
          r is not None and abs(r[2] - 3.0) < 1e-6)

    # 6 消融: semantic_frontier=False → 只排除已清区, 无方位偏好
    e = mk_env([(-3.0, 0.0), (0, 6.0)])
    e._ff = {'semantic_frontier': False}
    e._semantic_bearing_deg = 0.0
    e._semantic_votes = [(0, 0.0)]
    r = AVDBEnv._pick_frontier_target(e, (0.0, 0.0), np.argwhere(
        e._em.regions == 2), e._em, 2.5, 10.0)
    check('#32: 消融 off → 无方位偏好 (回纯几何)',
          r is not None and abs(r[2] - 3.0) < 1e-6 and r[3] == 'nearest')


# ========== 功能: LIKELY-BEARING 解析 (2 项) ==========
def test_parse_lb():
    from avdb_env import AVDBEnv

    class SW2(SW):
        pass

    e = object.__new__(AVDBEnv)
    e._ff = {'scan_inquiry': True, 'zone_clear_sight_guard': True}
    e.step = 8
    e.simWrapper = SW2()
    e._yaw_world = lambda: np.radians(-178.0)   # bmv2 起点附近朝向
    AVDBEnv._parse_scan_inquiry(
        e, 'LIKELY-BEARING: 120°\nSCAN_CLEAR', 'mahatma_rice')
    # 世界方位 = -178 + 120 = -58 (与 coca 局 APPROACH-LOCK 同换算)
    check('#32: LIKELY-BEARING 解析 → 世界方位 (yaw+scan 角)',
          abs(getattr(e, '_semantic_bearing_deg', 1e9) - (-58.0)) < 1e-6)
    check('#32: 无 LIKELY-BEARING 行 → 不设 (旧局回放兼容)',
          _no_lb_sets_none())


def _no_lb_sets_none():
    from avdb_env import AVDBEnv

    class SW2(SW):
        pass

    e = object.__new__(AVDBEnv)
    e._ff = {'scan_inquiry': True, 'zone_clear_sight_guard': True}
    e.step = 8
    e.simWrapper = SW2()
    e._yaw_world = lambda: 0.0
    AVDBEnv._parse_scan_inquiry(e, 'SCAN_CLEAR', 'x')
    return getattr(e, '_semantic_bearing_deg', None) is None


# ========== 开关注册 + 静态 (4 项) ==========
def test_flags_static():
    from feature_flags import DEFAULTS, FLAG_ARGS, parse_argv, resolve
    check('#32: DEFAULTS 含 semantic_frontier=True',
          DEFAULTS.get('semantic_frontier') is True)
    check('#32: --no-semantic-frontier 注册',
          FLAG_ARGS.get('--no-semantic-frontier') == ('semantic_frontier', False))
    flags, rest = parse_argv(['--no-semantic-frontier', 'y'])
    check('#32: parse_argv/resolve', flags == {'semantic_frontier': False}
          and rest == ['y'] and resolve()['semantic_frontier'] is True)

    ENV_SRC = open('/home/tao_h/VLMnav/src/avdb_env.py').read()
    check('#32: 扫描 prompt 第5问 + CLEAR 注记带方位 + 模式日志',
          'LIKELY-BEARING: <angle>°' in ENV_SRC
          and 'Prefer the direction toward' in ENV_SRC
          and "mode={mode}" in ENV_SRC)


if __name__ == '__main__':
    test_pick()
    test_parse_lb()
    test_flags_static()
    print('=' * 40)
    print(f"✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        for n in FAIL:
            print(f'  FAIL: {n}')
        sys.exit(1)
