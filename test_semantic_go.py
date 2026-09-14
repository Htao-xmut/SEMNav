"""#35 扫描定方向即承诺 + #36 零步转身 单测 (2026-09-13, bmv5 mahatma 实弹)

用户指令 (bmv5 看图三连问):
  "rescan变得毫无意义啊！！！scan完了以后难道更困惑了？"
  "没有周期性触发scan的必要吧。只要方向明确，那就继续往下探索，不用scan了。
   scan以后就是要找出探索的思路，而不是继续困惑！"
  "logs/.../step27直到这步才开始走出这个区域！！！前面都在干啥啊？"

bmv5 实弹 (logs/verify_v5_20260913.log): 7 次扫描 → 7 个可疑点 → 7 次
P6 转身 → 0 次方向承诺 — LIKELY-BEARING 每次都算了 (357°/331°/31°/…)
但只有前沿跳/离区会消费它, 而这两个通道一次都没触发 (区域看尽门没过)
→ 每 zone 磨 ~10 步漂移, step27 才走出客厅, 50 步 max_steps 终距 5.58m。
另: step6 P6 可疑 0° → rot_steps=0 → final=rotate_cw x0 空转烧掉一步,
step7 VLM 跟着 depth rec 选了 backward。

修复:
  #35 `_p6_resolve_suspicious`: zone 查尽 (无可疑 / 全查过 / 已查 ≥2 桶
      无果不再追第 3 个"新可疑") + 本次扫描 LIKELY-BEARING → 立即
      _plan_bearing_path(tag='SEMANTIC-GO') 旅程承诺;
  #36 转身 0 步的可疑 (视野正前) → 当场销账不空转; FIRST_DIRECTION
      <30° 同样不设 0 步 auto-action。

运行: python -u test_semantic_go.py
"""
import sys
sys.path.insert(0, '/home/tao_h/VLMnav/src')
import logging

logging.basicConfig(level=logging.CRITICAL)

PASS, FAIL = [], []


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f'  [{info}]' if info and not cond else ''))


class SW:
    def __init__(self, key=(1, -2)):
        self.current_node = 'cur'
        self._k = key

    def _area_key(self, n):
        return self._k


def mk_env(inspected=None, spots=None, bearing=None, bstep=9, lock=None,
           go_flag=True):
    from avdb_env import AVDBEnv
    e = object.__new__(AVDBEnv)
    e._ff = {'scan_inquiry': True, 'semantic_go': go_flag}
    e.step = 9
    e.simWrapper = SW()
    e._inspected_spots = inspected if inspected is not None else {}
    e._suspicious_spots = spots if spots is not None else {}
    e._approach_bearing = lock
    e._forced_return_path = []
    e._semantic_bearing_deg = bearing
    # #38 起聚合通道读票 (单值仅消融回退用) — 同步造票保持契约
    e._semantic_votes = [(bstep, float(bearing))] if bearing is not None else []
    e._semantic_bearing_step = bstep if bearing is not None else -99
    e._planned = []

    def _plan(bearing_rad, **kw):
        e._planned.append((float(bearing_rad), kw.get('tag')))
        e._forced_return_path = ['forward', 'forward']
        return True

    e._plan_bearing_path = _plan
    return e


# ========== 功能: _p6_resolve_suspicious (8 项) ==========
def test_resolve():
    from avdb_env import AVDBEnv

    # 1 未查过的可疑 → 照常转身 + 记桶出队 (原行为)
    e = mk_env(spots={(1, -2): [(5, 60)]})
    ang = AVDBEnv._p6_resolve_suspicious(e, 'SCAN_SUSPICIOUS: 60', 9)
    check('#33 保持: 未查过 → 转身查看 + 记桶',
          ang == 60 and (1, -2) in e._inspected_spots
          and e._suspicious_spots[(1, -2)] == []
          and e._planned == [])

    # 2 已查桶 → 销账不转身; 队列空 + 本次扫描方向 → SEMANTIC-GO 承诺
    e = mk_env(inspected={(1, -2): {1}}, bearing=90.0)
    ang = AVDBEnv._p6_resolve_suspicious(e, 'SCAN_SUSPICIOUS: 60', 9)
    check('#35: 已查桶销账 + 队列空 → 朝扫描方向立即成行',
          ang is None and len(e._planned) == 1
          and e._planned[0][1] == 'SEMANTIC-GO')

    # 3 bmv5 病灶: 已查 2 桶 + VLM 又发明第 3 个新可疑 → 不追, 查尽离区
    e = mk_env(inspected={(1, -2): {0, 2}}, bearing=120.0)
    ang = AVDBEnv._p6_resolve_suspicious(e, 'SCAN_SUSPICIOUS: 200', 9)
    check('#35: 已查 2+ 桶无果 → 新可疑不追 → 离区成行 (bmv5 病灶)',
          ang is None and len(e._planned) == 1
          and 200 // 60 not in e._inspected_spots[(1, -2)])

    # 4 #36: 可疑 <30° 在视野正前 → 不空转, 当场销账
    e = mk_env(spots={(1, -2): [(5, 20)]}, bearing=45.0)
    ang = AVDBEnv._p6_resolve_suspicious(e, 'SCAN_SUSPICIOUS: 20', 9)
    check('#36: <30° 可疑 → 不转身当场销账 (bmv5 step6 空转)',
          ang is None and 0 in e._inspected_spots[(1, -2)]
          and e._suspicious_spots[(1, -2)] == []
          and len(e._planned) == 1)

    # 5 本次扫描无可疑 → LIKELY-BEARING 直接成行
    e = mk_env(bearing=-31.0)
    AVDBEnv._p6_resolve_suspicious(e, 'SCAN_CLEAR\nLIKELY-BEARING: 200', 9)
    check('#35: 扫描无可疑 → 方向立即成行',
          len(e._planned) == 1)

    # 6 队列还有别的未查桶 + 无新鲜语义票 (#39: 新鲜票时语义优先会成行)
    e = mk_env(inspected={(1, -2): {1}},
               spots={(1, -2): [(5, 120)]}, bearing=90.0, bstep=3)
    ang = AVDBEnv._p6_resolve_suspicious(e, 'SCAN_SUSPICIOUS: 60', 9)
    check('#35: 队列尚有未查点且无新鲜票 → 只销账不成行',
          ang is None and e._planned == []
          and e._suspicious_spots[(1, -2)] == [(5, 120)])

    # 7 旧扫描的方位 (非本次) → 不成行 (防陈旧方向)
    e = mk_env(bearing=90.0, bstep=3)     # 扫描在 step3, 现在是 step9
    AVDBEnv._p6_resolve_suspicious(e, 'SCAN_CLEAR', 9)
    check('#35: 非本次扫描的 LIKELY-BEARING → 不成行',
          e._planned == [])

    # 8 消融: semantic_go off → 查尽也只销账不成行
    e = mk_env(inspected={(1, -2): {1}}, bearing=90.0, go_flag=False)
    AVDBEnv._p6_resolve_suspicious(e, 'SCAN_SUSPICIOUS: 60', 9)
    check('#35: 消融 --no-semantic-go → 不成行',
          e._planned == [])


# ========== 开关注册 + 静态 (4 项) ==========
def test_flags_static():
    from feature_flags import DEFAULTS, FLAG_ARGS, parse_argv
    check('#35: DEFAULTS 含 semantic_go=True',
          DEFAULTS.get('semantic_go') is True)
    check('#35: --no-semantic-go 注册',
          FLAG_ARGS.get('--no-semantic-go') == ('semantic_go', False))
    flags, rest = parse_argv(['--no-semantic-go', 'x'])
    check('#35: parse_argv 剥离', flags == {'semantic_go': False}
          and rest == ['x'])

    ENV_SRC = open('/home/tao_h/VLMnav/src/avdb_env.py').read()
    check('#36: 转身 0 步守卫在源 (FIRST_DIRECTION 路径同护)',
          'if rot_steps == 0:' in ENV_SRC
          and 'SEMANTIC-GO' in ENV_SRC)


if __name__ == '__main__':
    test_resolve()
    test_flags_static()
    print('=' * 40)
    print(f"✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        for n in FAIL:
            print(f'  FAIL: {n}')
        sys.exit(1)
