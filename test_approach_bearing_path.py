"""#53 APPROACH 大角度锁方位双修 单测 (2026-09-13, d01r2 coca 实弹)

病灶 (d01r2 step25-31, #52 拆除 VLM 角度折算拐杖后暴露):
  扫描锁 world bearing −124° vs 当前朝向 ~−3° (差 ~121°) — 对齐选边
  一步只能转 30°, 每步还涨 miss; miss 预算读者 (3326 行)
  `miss_budget = 6 if _approach_last_area < 0.10 else 3` — 扫描锁路径
  从不写 _approach_last_area (默认 1.0) → 吃 3 步预算 → 3 步耗尽释锁
  → #37 REACQUIRE 拉回扫描站位 → 重锁同方位 −124° (log 20:48:39 与
  20:50:24 两次 identical) → 无限循环, 50 步 max_steps 终距 10.57m。

修:
  #53a approach_bearing_path (flag, 默认开): _approach_step 中
      |锁方位 − 当前朝向| > 45° → 不再 30° 一步一挪, 沿锁方位图路径
      走 (_plan_bearing_path tag='APPROACH-TURN', 带平移的 GOTO 通道;
      4561 行旅程执行期接近让路, 锁保持世界系不丢); 规划失败回落旧行为
  #53b 扫描锁补记 _approach_last_area (无 flag, 纯数据补全):
      _warmup_scan 记 _scan_hit_area (area_ratio/bbox), warmup +
      rescan 两条锁路径都写 → 远距扫描锁 (area<0.10) miss 预算自然 6

运行: python -u test_approach_bearing_path.py
"""
import sys
sys.path.insert(0, '/home/tao_h/VLMnav/src')
import logging
import numpy as np

logging.basicConfig(level=logging.CRITICAL)

PASS, FAIL = [], []

SRC = open('/home/tao_h/VLMnav/src/avdb_env.py').read()
FF_SRC = open('/home/tao_h/VLMnav/src/feature_flags.py').read()


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f'  [{info}]' if info and not cond else ''))


class SW:
    current_node = None
    nav_graph = None


def mk_env(lock_deg=-124.0, heading_deg=-3.0, miss=1, last_area=0.004,
           plan_ok=True, ff=None):
    """最小 env: 直达 _approach_step 共享段 (miss 预算/圈检/对齐选边)

    无活体 YOLO (target_found False) → else 分支 miss+1 → 落到
    yaw=... 处 (#53a 插入点)。edge_options 给 forward → has_walk_opt
    True → 不触发绕障。current_node None → 圈检跳过。
    """
    from avdb_env import AVDBEnv
    e = object.__new__(AVDBEnv)
    e.simWrapper = SW()
    e._ff = dict(ff) if ff else {'approach_bearing_path': True,
                                 'servo_geometry': True}
    e._approach_bearing = np.radians(lock_deg)
    e._approach_miss = miss
    e._approach_last_area = last_area
    e._approach_nodes = []
    e.step = 26
    e._last_approach_idx = -1

    e._yaw_world = lambda: np.radians(heading_deg)
    e._option_world_bearing_deg = lambda o: None   # 断链回落旧模型
    calls = {'plan': [], 'exec': [], 'invalidate': []}
    e._plan_calls = calls

    def _plan(bearing, **kw):
        calls['plan'].append((float(np.degrees(bearing)), kw))
        if plan_ok:
            e._forced_return_path = ['a', 'b', 'c']
        return plan_ok

    e._plan_bearing_path = _plan
    sentinel = ('FORCED', 0)
    e._execute_forced_return = lambda obs: (calls['exec'].append(1), sentinel)[1]
    e._invalidate_lock = lambda *a, **k: calls['invalidate'].append(1)
    e._arm_reacquire_after_miss = lambda: None
    e._override_and_run = lambda obs, i, tag, note='': ('ALIGNED', i)
    return e, sentinel, calls


def mk_obs():
    return {
        'yolo_detection': {'target_found': False, 'confidence': 0.0},
        'edge_options': [
            {'chain_type': 'forward', 'chain_count': 1,
             'direction': 'forward'},
            {'chain_type': 'rotate_ccw', 'chain_count': 1,
             'direction': 'rotate_left'},
        ],
    }


# ========== 开关注册 (2 项) ==========
def test_flags():
    import feature_flags as F
    check('#53a: DEFAULTS approach_bearing_path=True',
          F.DEFAULTS.get('approach_bearing_path') is True)
    flags, rest = F.parse_argv(['--no-approach-bearing-path', 'x'])
    check('#53a: --no-approach-bearing-path 消融注册',
          F.FLAG_ARGS.get('--no-approach-bearing-path')
          == ('approach_bearing_path', False)
          and flags == {'approach_bearing_path': False} and rest == ['x'])


# ========== #53a: 大角度锁 → 图路径 (5 项) ==========
def test_big_gap_bearing_path():
    from avdb_env import AVDBEnv
    # d01r2 病例: 锁 −124° vs 朝向 −3° (差 −121°) → 图路径, 不再 30° 挪
    e, sentinel, calls = mk_env()
    r = AVDBEnv._approach_step(e, mk_obs())
    check('#53a: d01r2 病例 (差 121°) → APPROACH-TURN 图路径执行',
          r == sentinel and calls['exec'] == [1]
          and calls['plan'] and calls['plan'][0][0] == -124.0,
          f'r={r} plan={calls["plan"]}')
    check('#53a: 规划参数 tag=APPROACH-TURN + max_path=8 (短旅程)',
          calls['plan'][0][1].get('tag') == 'APPROACH-TURN'
          and calls['plan'][0][1].get('max_path') == 8
          and calls['plan'][0][1].get('max_edges') == 8)

    # 规划失败 (无已知空间进展) → 回落旧对齐选边 (旧行为: 121° 差 +
    # 只有 forward/rotate_x1 选项 → 对齐门 d<=90 不中 → 返回 None 交还
    # VLM — d01r2 振荡正是这条路; 关键断言 = 不卡死/不强制执行)
    e2, _, calls2 = mk_env(plan_ok=False)
    r2 = AVDBEnv._approach_step(e2, mk_obs())
    check('#53a: 规划失败 → 回落旧行为 (对齐门不中 → None, 不卡死)',
          r2 is None and not calls2['exec'] and len(calls2['plan']) == 1)

    # 小角度差 (10°) → 不走图路径, 直接对齐选边 (旧行为)
    e3, _, calls3 = mk_env(lock_deg=-13.0)
    r3 = AVDBEnv._approach_step(e3, mk_obs())
    check('#53a: 小角度差 (10°) → 旧行为对齐选边, 不规划',
          r3 == ('ALIGNED', 0) and not calls3['plan'])

    # 消融 off → 大角度差也走旧行为 (对齐门不中 → None; 30° 一步一挪
    # + 3 步预算释锁的 d01r2 病理保留供对照)
    e4, _, calls4 = mk_env(ff={'approach_bearing_path': False,
                               'servo_geometry': True})
    r4 = AVDBEnv._approach_step(e4, mk_obs())
    check('#53a: 消融 off → 旧行为 (不规划图路径)',
          r4 is None and not calls4['plan'])

    # 边界: 恰好 46° 差 (>45 阈) → 图路径; 活体重瞄 ≤44.7° 天然不触发
    e5, sentinel5, calls5 = mk_env(lock_deg=-49.0)
    r5 = AVDBEnv._approach_step(e5, mk_obs())
    check('#53a: 阈值边界 — 46° 差触发图路径',
          r5 == sentinel5 and len(calls5['plan']) == 1)


# ========== #53b: 扫描锁 miss 预算分档 (4 项) ==========
def test_scan_lock_budget():
    from avdb_env import AVDBEnv
    # 注: else 分支入口 miss+=1 后才查预算 — 传入 miss 是查前值
    # 远距扫描锁 (area 0.004 <0.10) → 预算 6: 查前 miss=3 (查时 4) 不释锁
    # (d01r2 病例: 默认 1.0 吃 3 步预算, 转身未完成即释锁)
    e, _, calls = mk_env(lock_deg=-13.0, miss=3, last_area=0.004)
    r = AVDBEnv._approach_step(e, mk_obs())
    check('#53b: 远距锁 (area 0.004) miss=3 < 预算 6 → 不释锁',
          not calls['invalidate'] and r == ('ALIGNED', 0))

    # 近距锁 (area 1.0 ≥0.10) → 预算 3: 查前 miss=3 (查时 4) 释锁
    e2, _, calls2 = mk_env(lock_deg=-13.0, miss=3, last_area=1.0)
    r2 = AVDBEnv._approach_step(e2, mk_obs())
    check('#53b: 近距锁 miss=3 → 释锁 (bm12 近距止损保持)',
          len(calls2['invalidate']) == 1 and r2 is None)

    # 查前 miss=4 (查时 5) 仍在远距预算内
    e3, _, calls3 = mk_env(lock_deg=-13.0, miss=4, last_area=0.004)
    AVDBEnv._approach_step(e3, mk_obs())
    check('#53b: 远距锁 miss=4 仍未释锁', not calls3['invalidate'])

    # 查前 miss=5 (查时 6) → 远距预算耗尽释锁 (预算 6 不是无限)
    e4, _, calls4 = mk_env(lock_deg=-13.0, miss=5, last_area=0.004)
    r4 = AVDBEnv._approach_step(e4, mk_obs())
    check('#53b: 远距锁 miss=5 → 释锁 (预算封顶)',
          len(calls4['invalidate']) == 1 and r4 is None)


# ========== #53b: 扫描锁写入 + 面积捕获接线 (4 项) ==========
def test_scan_lock_wiring():
    # warmup 扫描: _scan_hit_offset_deg 同址捕获 _scan_hit_area
    check('#53b: _warmup_scan 捕获 _scan_hit_area (offset 同址)',
          '_scan_hit_area' in SRC
          and SRC.count("self._scan_hit_area = float(_area) if _area else None") == 1
          and "self._scan_hit_offset_deg = self._yolo_frame_offset_deg("
              in SRC)
    # rescan 锁路径 (3865 系) 写 _approach_last_area
    check('#53b: rescan 锁写 _approach_last_area',
          "self._approach_bearing = self._scan_lock_bearing_rad(new_angle)"
          in SRC
          and "'_scan_hit_area', None) or 1.0" in SRC
          and SRC.count("self._approach_last_area = getattr(") >= 2)
    # 两条锁路径 (warmup 1188 / rescan 3865) 都写
    check('#53b: warmup + rescan 两条锁路径都写 (>1 处)',
          SRC.count("self._approach_last_area = getattr(") >= 2)
    # #53a 静态接线: 阈值 45 + tag + 旅程让路守卫 (4561) 保持
    check('#53a: 源接线 (d_lock 阈值 45 + APPROACH-TURN tag)',
          'APPROACH-TURN' in SRC and '> 45.0' in SRC
          and "if not self._forced_return_path and (" in SRC)


if __name__ == '__main__':
    test_flags()
    test_big_gap_bearing_path()
    test_scan_lock_budget()
    test_scan_lock_wiring()
    print('=' * 40)
    print(f"✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        for n in FAIL:
            print(f'  FAIL: {n}')
        sys.exit(1)
