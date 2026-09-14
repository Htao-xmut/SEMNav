"""#51 区域记忆 + 回访门 + 远距小框二确认 单测 (2026-09-13)

bmd01 coca / bmd03 nutrigrain 实弹 (用户报障 "不停地去探索同一个区域,
是要反复确认啥?? 那其他还没探索的区域都还没完成探索吖!"):
  d01: 真可乐瓶被 YOLO 看到 9 次 (conf 0.37-0.51), 4 次 IDENTITY no 直接
       放走 — unarmed 目击路径没接 #42 二确认 (armed 有), 远距小框
       (area 0.003-0.006) VLM 根本看不清, no 不是证据 → 目击区因 #31
       守卫永不 CLEAR → 记忆提示拽回再看再否, 50 步终距 10.76m。
  d03: 前沿池只认"站立点", 走过扫过的走廊边格子仍是灰格 → 最近灰格
       永远在走过的功能区里 → 152° 前沿跳把机器人送回起点旁。

修:
  #51a unarmed identity-no: conf<0.50 或 框<1% → 前进二次确认 (≤3/节点,
       预算与 #42 共用); armed (#42) 同步补 area<0.01 条件
  #51b zone 否决记账 (_zone_neg_strike, ≥2 击 → 否决区) + 每步 prompt
       注入"已否决勿回"警示
  #51c 前沿分层回访门: 全新 zone > 扫过未否决 > 否决区 (仅兜底),
       zone_revisit_gate 消融开关

运行: python -u test_zone_memory.py
"""
import sys
sys.path.insert(0, '/home/tao_h/VLMnav/src')
import logging

logging.basicConfig(level=logging.CRITICAL)

PASS, FAIL = [], []

SRC = open('/home/tao_h/VLMnav/src/avdb_env.py').read()


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f'  [{info}]' if info and not cond else ''))


def mk_env(ff=None):
    from avdb_env import AVDBEnv
    e = object.__new__(AVDBEnv)
    e._ff = dict(ff) if ff else {}
    e._zone_negative_strike = {}      # reset() 内的等价初始化
    e._zone_negative_zones = set()

    class SW:
        current_node = 'n0'

        def _area_key(self, n):
            return (1, 2)

    e.simWrapper = SW()
    return e


# ========== 功能: #51b 否决记账 (4 项) ==========
def test_neg_strikes():
    from avdb_env import AVDBEnv
    e = mk_env()
    AVDBEnv._zone_neg_strike(e)
    check('#51b: 一击 → 计数 1, 未进否决区',
          e._zone_negative_strike == {(1, 2): 1}
          and (1, 2) not in e._zone_negative_zones)
    AVDBEnv._zone_neg_strike(e)
    check('#51b: 两击 → 否决区',
          e._zone_negative_strike == {(1, 2): 2}
          and (1, 2) in e._zone_negative_zones)
    # _area_key 抛错 → 不炸 (静默跳过)
    e2 = mk_env()

    class SW2:
        current_node = 'n0'

        def _area_key(self, n):
            raise RuntimeError('no graph')

    e2.simWrapper = SW2()
    try:
        AVDBEnv._zone_neg_strike(e2)
        check('#51b: _area_key 异常 → 不炸', True)
    except Exception:
        check('#51b: _area_key 异常 → 不炸', False)


# ========== 功能: #51b 警示注入 (3 项) ==========
def test_caution_injection():
    from avdb_env import AVDBEnv
    e = mk_env()
    e._zone_negative_zones = {(1, 2), (-3, 0)}
    obs = {'memory_context': 'base'}
    AVDBEnv._inject_zone_caution(e, obs)
    check('#51b: 否决区非空 → memory_context 注入警示',
          'ALREADY CHECKED & REJECTED' in obs['memory_context']
          and 'zone_1_2' in obs['memory_context']
          and obs['memory_context'].startswith('base'))
    e2 = mk_env()
    e2._zone_negative_zones = set()
    obs2 = {'memory_context': 'base'}
    AVDBEnv._inject_zone_caution(e2, obs2)
    check('#51b: 无否决区 → 不注入', obs2['memory_context'] == 'base')
    e3 = mk_env({'zone_revisit_gate': False})
    e3._zone_negative_zones = {(1, 2)}
    obs3 = {'memory_context': 'base'}
    AVDBEnv._inject_zone_caution(e3, obs3)
    check('#51b: 消融 off → 不注入', obs3['memory_context'] == 'base')


# ========== 功能: #51c 前沿分层 (7 项) ==========
def test_frontier_tiers():
    from avdb_env import AVDBEnv
    import numpy as np

    class EM:
        def _to_world(self, gz, gx):
            return (float(gx), float(gz))

    def mk(ff=None, neg=(), scanned=(), cleared=()):
        e = mk_env(ff)
        e._zone_negative_zones = set(neg)
        e._scan_positions_by_cell = {c: {'x'} for c in scanned}
        e._zone_clear_cells = set(cleared)
        e._semantic_bearing_deg = None
        e.simWrapper.AREA_CELL = 2.0
        return e

    # cells: (gz, gx) → world (x, gx) — zone = (gx//2, gz//2)
    # cur=(0,0): A 新 zone 距 4m; B 扫过 zone 距 2m; C 否决 zone 距 1m
    CELLS = [(0, 4), (0, 2), (0, 1)]   # world x=4 (zone2,新), x=2 (zone1,扫过), x=1 (zone0,否决)
    e = mk(neg={(0, 0)}, scanned={(1, 0)})
    r = AVDBEnv._pick_frontier_target(e, (0.0, 0.0), CELLS, EM(), 0.5, 10.0)
    check('#51c: 新 zone (4m) 赢过更近的扫过 (2m) / 否决 (1m)',
          r is not None and r[0] == 4.0)

    # 无新 zone → 扫过未否决 (2m) 赢过否决 (1m)
    e = mk(neg={(0, 0)}, scanned={(1, 0), (2, 0)})
    r = AVDBEnv._pick_frontier_target(e, (0.0, 0.0), CELLS, EM(), 0.5, 10.0)
    check('#51c: 扫过未否决 (2m) 赢过否决 (1m)',
          r is not None and r[0] == 2.0)

    # 全是否决/扫过 → 兜底取最近 (不死锁, 用户规则的反面 = 永远不出门)
    e = mk(neg={(0, 0), (1, 0), (2, 0)})
    r = AVDBEnv._pick_frontier_target(e, (0.0, 0.0), CELLS, EM(), 0.5, 10.0)
    check('#51c: 全否决 → 兜底最近 (1m, 不死锁)',
          r is not None and r[0] == 1.0)

    # cleared 出池保持 (#32 原语义): 唯一 cleared 格更近也不选
    e = mk(cleared={(1, 0)}, scanned={(2, 0)})
    r = AVDBEnv._pick_frontier_target(e, (0.0, 0.0), CELLS[:2], EM(), 0.5, 10.0)
    check('#51c: cleared 出池保持 (#32)',
          r is not None and r[0] == 4.0)

    # 消融 off → 回到旧行为: 最近灰格 (否决 1m 也照选)
    e = mk({'zone_revisit_gate': False}, neg={(0, 0)}, scanned={(1, 0)})
    r = AVDBEnv._pick_frontier_target(e, (0.0, 0.0), CELLS, EM(), 0.5, 10.0)
    check('#51c: 消融 off → 旧最近行为 (否决 1m 照选)',
          r is not None and r[0] == 1.0)

    # 语义偏好在新 zone 层内仍生效: 两个新 zone, 语义对齐远的那个优先
    e = mk()
    e._semantic_bearing_agg = lambda: (0.0, 0)   # 聚合方位朝 +x (mock 投票)
    CELLS2 = [(0, 4), (0, -4)]     # x=+4 对齐 90°±90 (4m); x=-4 反向不对齐 (4m)
    r = AVDBEnv._pick_frontier_target(e, (0.0, 0.0), CELLS2, EM(), 0.5, 10.0)
    check('#51c: 语义偏好在分层内保持 (对齐 4m 赢不对齐同距者)',
          r is not None and r[3] == 'semantic' and r[0] == 4.0)


# ========== 功能: 开关注册 (2 项) ==========
def test_flags():
    import feature_flags as F
    check('#51: DEFAULTS 含 zone_revisit_gate=True',
          F.DEFAULTS.get('zone_revisit_gate') is True)
    flags, rest = F.parse_argv(['--no-zone-revisit-gate', 'x'])
    check('#51: --no-zone-revisit-gate 注册剥离',
          F.FLAG_ARGS.get('--no-zone-revisit-gate')
          == ('zone_revisit_gate', False)
          and flags == {'zone_revisit_gate': False} and rest == ['x'])


# ========== 静态: #51a 接线 (5 项) ==========
def test_static_51a():
    # unarmed 路径: conf<0.50 或 area<0.01 → 二确认 (bmd01 病灶)
    check('#51a: unarmed 条件 conf_u<0.50 or area_u<0.01 在源',
          '(conf_u < 0.50 or area_u < 0.01)' in SRC
          and '#51a' in SRC)
    # 预算与 #42 共用同一个 _lowconf_confirm_used 字典
    check('#51a: 预算共用 _lowconf_confirm_used (armed/unarmed 同字典)',
          SRC.count('_lowconf_confirm_used') >= 6)
    # armed (#42) 同步补 area<0.01 条件 (d01 step1: conf 0.51 area 0.004)
    check('#51a: armed 路径条件含 area<0.01',
          '(conf < 0.50 or area < 0.01)' in SRC)
    # 两个 reject 点都挂了区域否决记账
    check('#51b: 两处 _identity_reject 后接 _zone_neg_strike()',
          SRC.count('self._zone_neg_strike()') == 2)
    # reset 复位 (跨回合不残留)
    check('#51b: reset 初始化 _zone_negative_strike/_zone_negative_zones',
          'self._zone_negative_strike = {}' in SRC
          and 'self._zone_negative_zones = set()' in SRC)


if __name__ == '__main__':
    test_neg_strikes()
    test_caution_injection()
    test_frontier_tiers()
    test_flags()
    test_static_51a()
    print('=' * 40)
    print(f"✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        for n in FAIL:
            print(f'  FAIL: {n}')
        sys.exit(1)
