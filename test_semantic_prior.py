"""#38 语义先验 V2 单测 (2026-09-13, bmv6 mahatma 实弹 + 用户三选拍板)

用户拍板 (AskUserQuestion 三连, 全选推荐项):
  ① 语义先验 = 目击后第一优先 (warmup 即承诺, depth rec 降兜底);
  ② FIRST_DIRECTION 证据门分类处理 ("看见"过门, "可能方向"降级投票);
  ③ 投票聚合 (圆均值 + 翻向守卫)。

bmv6 实弹病灶对照:
  - warmup LIKELY-BEARING 240°→world 57° (厨房) 无消费方 → step1-3 跟
    depth 几何走沙发;
  - step4 FIRST_DIRECTION 240° (还是厨房) 被 likelihood 门一杀 →
    `FIRST_DIRECTION: REJECTED-240deg (likelihood claim, not actionable)`;
  - 单次 LIKELY-BEARING 180°→world 331° 拿 11 边承诺 → 331°→128°→−15°
    三旅程互殴。

运行: python -u test_semantic_prior.py
"""
import sys
sys.path.insert(0, '/home/tao_h/VLMnav/src')
import math
import logging

logging.basicConfig(level=logging.CRITICAL)

PASS, FAIL = [], []


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f'  [{info}]' if info and not cond else ''))


def mk_env(votes=None, committed=None, cstep=-99, step=9, prior=True,
           go_flag=True, sem_deg=None, over_sus=True):
    from avdb_env import AVDBEnv
    e = object.__new__(AVDBEnv)
    e._ff = {'semantic_prior': prior, 'semantic_go': go_flag,
             'scan_inquiry': True, 'first_dir_gate': True,
             'semantic_over_suspicious': over_sus}
    e.step = step
    e._semantic_votes = list(votes or [])
    e._last_committed_bearing = committed
    e._last_committed_step = cstep
    e._approach_bearing = None
    e._forced_return_path = []
    e._semantic_bearing_deg = sem_deg
    e._semantic_bearing_step = step
    e._planned = []
    e._inspected_spots = {}
    e._suspicious_spots = {}
    e.simWrapper = type('SW', (), {
        'current_node': 'n', '_area_key': staticmethod(lambda n: (1, -2))})()
    e._yaw_world = lambda: 0.0

    def _plan(bearing_rad, **kw):
        e._planned.append((float(bearing_rad), kw.get('tag')))
        e._forced_return_path = ['forward', 'forward']
        return True

    e._plan_bearing_path = _plan
    return e


# ========== ① 投票聚合 (5 项) ==========
def test_agg():
    from avdb_env import AVDBEnv

    # 1 单票 → 原值
    e = mk_env(votes=[(9, 57.0)])
    agg = AVDBEnv._semantic_bearing_agg(e)
    check('#38: 单票 → 原值 (warmup 厨房 57° 直接可用)',
          agg is not None and abs(agg[0] - 57.0) < 1.0)

    # 2 三票一致 → 圆均值 (bmv6: 两扫都说 240° 厨房本该稳)
    e = mk_env(votes=[(5, 350.0), (7, 10.0), (9, 355.0)])
    agg = AVDBEnv._semantic_bearing_agg(e)
    check('#38: 跨 0° 圆均值不翻车 (350/10/355 → ~358°)',
          agg is not None and abs((agg[0] - 358.0 + 180) % 360 - 180) < 5.0)

    # 3 翻向守卫: 上次承诺 331°, 单票 128° (相反) → 拒翻 (bmv6 乒乓病灶)
    e = mk_env(votes=[(9, 128.0)], committed=331.0, cstep=4)
    agg = AVDBEnv._semantic_bearing_agg(e)
    check('#38: 与上次承诺相反且单票 → 不翻向 (防 331°→128°)',
          agg is None)

    # 4 合法换向: 走过了没找到, 连续两票一致指新方向 → 允许翻
    e = mk_env(votes=[(8, 130.0), (9, 125.0)], committed=331.0, cstep=4)
    agg = AVDBEnv._semantic_bearing_agg(e)
    check('#38: 两票一致指新方向 → 允许翻 (合法换向不被误杀)',
          agg is not None and abs((agg[0] - 128.0 + 180) % 360 - 180) < 10.0)

    # 5 消融: --no-semantic-prior → 退回 #32 单值行为
    e = mk_env(votes=[(9, 128.0)], committed=331.0, cstep=4,
               prior=False, sem_deg=331.0)
    agg = AVDBEnv._semantic_bearing_agg(e)
    check('#38: 消融 → 单值回退 (无聚合无守卫)',
          agg is not None and abs(agg[0] - 331.0) < 1.0)


# ========== ② 票注册 + 证据门降级 (4 项) ==========
def test_votes_and_gate():
    from avdb_env import AVDBEnv

    # 6 票注册: 只留 30 步内最近 3 张
    e = mk_env(step=30, votes=[(1, 10.0), (20, 40.0), (25, 60.0), (28, 80.0)])
    AVDBEnv._register_semantic_vote(e, 100.0, 'test')
    check('#38: 票过期 (30 步) + 上限 3 张',
          [v[1] for v in e._semantic_votes] == [60.0, 80.0, 100.0])

    # 7 bmv6 step4 病灶: FIRST_DIRECTION 推测声明 → 降级投票不再蒸发
    e = mk_env(step=4)
    e.simWrapper = type('SW', (), {'current_node': 'n'})()
    e._yaw_world = lambda: 0.0
    resp = ('The mahatma_rice is likely located in the kitchen area. '
            'FIRST_DIRECTION: 240°\nSCAN_CLEAR')
    ang, resp2 = AVDBEnv._verify_first_direction(
        e, resp, 240, [{'angle': 240, 'img': object(), 'yolo_found': False}],
        'mahatma_rice')
    check('#38: likelihood 声明 → 角度作废 + 票进通道 (240° 不再蒸发)',
          ang is None and 'downgraded' in resp2
          and any(abs(v[1] - 240.0) < 1.0 for v in e._semantic_votes))

    # 8 降级票不带目击语义: +20 bonus 注入线碰不到 (REJECTED 行格式拦住)
    import re as _re
    check('#38: 降级不复活 +20 bonus (REJECTED 行下游正则解析不到)',
          not _re.search(r'FIRST_DIRECTION:\s*\d+°', resp2))

    # 9 消融: --no-semantic-prior → 推测声明照旧一杀 (不投票)
    e = mk_env(step=4, prior=False)
    e.simWrapper = type('SW', (), {'current_node': 'n'})()
    e._yaw_world = lambda: 0.0
    ang, _ = AVDBEnv._verify_first_direction(
        e, resp, 240, [{'angle': 240, 'img': object(), 'yolo_found': False}],
        'mahatma_rice')
    check('#38: 消融 → 推测声明不投票 (旧行为)',
          ang is None and not e._semantic_votes)


# ========== ③ 承诺通道 (3 项) ==========
def test_commit():
    from avdb_env import AVDBEnv

    # 10 P6 成行走聚合方位 (非最新单值): 票 50/50 → 旅程 bearing 50°
    e = mk_env(votes=[(9, 50.0), (9, 50.0)], sem_deg=310.0)
    AVDBEnv._p6_resolve_suspicious(e, 'SCAN_CLEAR', 9)
    check('#38: SEMANTIC-GO 走聚合票 (不被最新单值 310° 劫持)',
          len(e._planned) == 1
          and abs(math.degrees(e._planned[0][0]) - 50.0) < 3.0
          and e._planned[0][1] == 'SEMANTIC-GO')

    # 11 成功承诺记录翻向守卫基准
    check('#38: 承诺落盘 _last_committed_bearing (守卫基准)',
          abs((e._last_committed_bearing - 50.0 + 180) % 360 - 180) < 3.0)

    # 12 warmup 即承诺: 接线 + 门 (无目击 angle_to_use=None + 无可疑才开)
    ENV_SRC = open('/home/tao_h/VLMnav/src/avdb_env.py').read()
    check('#38: warmup 即承诺接线 (bmv6 厨房 57° 无消费方病灶)',
          '[SEMANTIC-GO] warmup 语义先验即承诺' in ENV_SRC
          and "angle_to_use is None and warmup_summary" in ENV_SRC
          and "SCAN_SUSPICIOUS:\\s*(\\d+)" in ENV_SRC)

    # 13 前沿落点排序同源聚合 (不打架)
    check('#38: _pick_frontier_target 也用聚合方位',
          '_pick_frontier_target' in ENV_SRC
          and 'agg_f = self._semantic_bearing_agg()' in ENV_SRC)


# ========== ④ #39 语义优先, 可疑顺路查 (5 项) ==========
def test_over_suspicious():
    from avdb_env import AVDBEnv

    # 14 不顺路可疑 + 本次新鲜票 → 不转身留记忆, 方向成行
    e = mk_env(votes=[(9, 50.0)])          # 语义 world 50°
    ang = AVDBEnv._p6_resolve_suspicious(e, 'SCAN_SUSPICIOUS: 200', 9)
    check('#39: 不顺路可疑 (200° vs 语义 50°) → 不追, 方向成行',
          ang is None and len(e._planned) == 1
          and e._planned[0][1] == 'SEMANTIC-GO'
          and (1, -2) not in e._inspected_spots)

    # 15 顺路可疑 (±60°) → 照旧转身查看 (查它 = 往语义方向走)
    e = mk_env(votes=[(9, 50.0)])
    ang = AVDBEnv._p6_resolve_suspicious(e, 'SCAN_SUSPICIOUS: 60', 9)
    check('#39: 顺路可疑 (60° vs 语义 50°) → 照旧转身查看',
          ang == 60 and 1 in e._inspected_spots.get((1, -2), set())
          and e._planned == [])

    # 16 无新鲜票 (票是 step3 的) → 可疑照旧优先 (旧行为)
    e = mk_env(votes=[(3, 50.0)])
    ang = AVDBEnv._p6_resolve_suspicious(e, 'SCAN_SUSPICIOUS: 200', 9)
    check('#39: 无本次票 → 可疑照旧优先 (原行为)',
          ang == 200 and e._planned == [])

    # 17 消融: --no-semantic-over-suspicious → 不顺路也转身 (回到旧序)
    e = mk_env(votes=[(9, 50.0)], over_sus=False)
    ang = AVDBEnv._p6_resolve_suspicious(e, 'SCAN_SUSPICIOUS: 200', 9)
    check('#39: 消融 → 不顺路也转身 (旧序)',
          ang == 200 and e._planned == [])

    # 18 warmup 门同规则: 顺路可疑让查看, 不顺路/无语义才走 (静态)
    ENV_SRC = open('/home/tao_h/VLMnav/src/avdb_env.py').read()
    check('#39: warmup 承诺门用同一顺路规则 (可疑不再一票否决)',
          '_sus_blocks' in ENV_SRC
          and 'elif _sus_m is not None and agg_w is None:' in ENV_SRC)


# ========== ⑤ 开关注册 (3 项) ==========
def test_flags():
    from feature_flags import DEFAULTS, FLAG_ARGS, parse_argv
    check('#38: DEFAULTS 含 semantic_prior=True',
          DEFAULTS.get('semantic_prior') is True)
    flags, rest = parse_argv(['--no-semantic-prior', 'x'])
    check('#38: --no-semantic-prior 注册 + 剥离',
          FLAG_ARGS.get('--no-semantic-prior') == ('semantic_prior', False)
          and flags == {'semantic_prior': False} and rest == ['x'])
    check('#39: --no-semantic-over-suspicious 注册',
          FLAG_ARGS.get('--no-semantic-over-suspicious')
          == ('semantic_over_suspicious', False))


if __name__ == '__main__':
    test_agg()
    test_votes_and_gate()
    test_commit()
    test_over_suspicious()
    test_flags()
    print('=' * 40)
    print(f"✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        for n in FAIL:
            print(f'  FAIL: {n}')
        sys.exit(1)
