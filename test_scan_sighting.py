"""#37 扫描目击登记 + 失而复得 单测 (2026-09-13, bmv6 mahatma 实弹)

bmv6 实弹 (logs/verify_v6_20260913.log): 3 次 APPROACH-LOCK 全来自六图
扫描的 YOLO 命中 (conf 0.51/0.56/0.56; 节点 000110005880101.jpg 被命中
两次), 每次伺服 3-6 步零活体检出 → `_invalidate_lock(blacklist=False)`
释放。但扫描命中只产 target_priority 文本 — 从未进 target_sighting_nodes
(wrapper 只登记活体帧), `_start_forced_return` 的 >3 步新鲜度也早过期
→ "确曾看见目标的视角"证据全部蒸发, 机器人继续盲走 7.52m max_steps。

修复 (#37):
  - 扫描命中按 wrapper 同构格式登记目击 (含 ③c 配对帧 + promising 状态);
  - 释锁当步 `_arm_reacquire_after_miss`: 最近扫描目击 ≤12 步且未补救过
    → 走回扫描站位重看 (P4 释锁重扫在旅程后从该视角开火);
  - 防同坑循环: 每个目击节点只补救一次 (`_reacquire_done`)。

运行: python -u test_scan_sighting.py
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
    def __init__(self, sightings=None, cur='A'):
        self.current_node = cur
        self.target_sighting_nodes = sightings or []
        self._recorded = []

    def _record_detection(self, node, desc=''):
        self._recorded.append((node, desc))


def mk_env(sightings=None, cur='A', path=('forward', 'forward', 'rotate_cw'),
           flag=True, done=None, step=9, sstep=4):
    from avdb_env import AVDBEnv
    e = object.__new__(AVDBEnv)
    e._ff = {'scan_sighting': flag}
    e.step = step
    e.simWrapper = SW(sightings, cur)
    e._forced_return_path = []
    e._forced_return_dest = None
    if done is not None:
        e._reacquire_done = done

    def _gpt(a, b):
        return list(path) if b == 'B' else None

    e._graph_path_to = _gpt
    e._scan_sighting = (sstep, 'B', 'YOLO scan hit 300deg, conf=0.51')
    return e


def scan_hit(step=4, node='B', ang='300'):
    return (step, node, f'YOLO scan hit {ang}deg, conf=0.51')


# ========== 功能: _arm_reacquire_after_miss (7 项) ==========
def test_arm():
    from avdb_env import AVDBEnv

    # 1 bmv6 病灶: 释锁当步, 扫描目击 5 步前 → 走回目击站位
    e = mk_env(sightings=[scan_hit(4)])
    AVDBEnv._arm_reacquire_after_miss(e)
    check('#37: 扫描目击 ≤12 步 → 走回重看旅程成行 (bmv6 病灶)',
          e._forced_return_path == ['forward', 'forward', 'rotate_cw']
          and e._forced_return_dest == 'B'
          and e._reacquire_done == {'B'}
          and e._last_returned_sighting_step == 4)

    # 2 释锁时活体检目击更近 (非 scan hit) → 不走本通道 (原 3 步通道管)
    e = mk_env(sightings=[(8, 'C', 'YOLO confirmed at left, conf=0.61')])
    AVDBEnv._arm_reacquire_after_miss(e)
    check('#37: 纯活体目击 → 不补救 (原 FORCED-RETURN 通道管)',
          e._forced_return_path == [])

    # 3 混合: 最近是活体, 但仍有更早扫描目击 ≤12 步 → 仍补救扫描视角
    e = mk_env(sightings=[scan_hit(4),
                          (8, 'C', 'YOLO confirmed at left, conf=0.61')])
    AVDBEnv._arm_reacquire_after_miss(e)
    check('#37: 活体之后仍有新鲜扫描目击 → 补救扫描视角',
          e._forced_return_dest == 'B')

    # 4 陈旧 (>12 步) → 不补救
    e = mk_env(sightings=[scan_hit(1)], step=14)   # 13 步前
    AVDBEnv._arm_reacquire_after_miss(e)
    check('#37: >12 步陈旧扫描目击 → 不补救',
          e._forced_return_path == [])

    # 5 同坑防循环: 该节点已补救过 → 不再走第二次
    e = mk_env(sightings=[scan_hit(4)], done={'B'})
    AVDBEnv._arm_reacquire_after_miss(e)
    check('#37: 每节点只补救一次 (防扫描命中→丢→回→再丢循环)',
          e._forced_return_path == [])

    # 6 已在目击节点 → 不补救
    e = mk_env(sightings=[scan_hit(4)], cur='B')
    AVDBEnv._arm_reacquire_after_miss(e)
    check('#37: 已在目击站位 → 不补救',
          e._forced_return_path == [])

    # 7 消融: --no-scan-sighting → 登记与补救全关
    e = mk_env(sightings=[scan_hit(4)], flag=False)
    AVDBEnv._arm_reacquire_after_miss(e)
    check('#37: 消融 --no-scan-sighting → 不补救',
          e._forced_return_path == [])


# ========== 登记 + 接线 (静态/半功能, 6 项) ==========
def test_registration():
    ENV_SRC = open('/home/tao_h/VLMnav/src/avdb_env.py').read()

    # 登记块: scan_data 带 node + 目击/帧/promising 三件套
    check('#37: scan_data 带 node (目击登记数据源)',
          "'node': walk_node,   # #37" in ENV_SRC)
    check('#37: 扫描命中登记目击 + ③c 配对帧 + promising',
          '[SCAN-SIGHTING]' in ENV_SRC
          and "self._sighting_frames[best['node']] = best.get('img')" in ENV_SRC
          and "_record_detection(best['node'], _desc)" in ENV_SRC)

    # 接线: 释锁分支在 _invalidate_lock 之后调用补救 (顺序不能反 —
    # _invalidate_lock 会清 _forced_return_path)
    i_inv = ENV_SRC.find(
        "f'target lost for {self._approach_miss} steps — detection stale',")
    i_arm = ENV_SRC.find('self._arm_reacquire_after_miss()')
    check('#37: 释锁分支接线 (invalidate 之后 arm)',
          0 < i_inv < i_arm
          and '_arm_reacquire_after_miss' in ENV_SRC)

    # 复位: 补救集不跨回合残留
    check('#37: 回合复位不残留', 'self._reacquire_done = set()' in ENV_SRC)

    # 开关注册
    from feature_flags import DEFAULTS, FLAG_ARGS, parse_argv
    check('#37: DEFAULTS 含 scan_sighting=True',
          DEFAULTS.get('scan_sighting') is True)
    check('#37: --no-scan-sighting 注册 + 剥离',
          FLAG_ARGS.get('--no-scan-sighting') == ('scan_sighting', False)
          and parse_argv(['--no-scan-sighting', 'x'])
          == ({'scan_sighting': False}, ['x']))


if __name__ == '__main__':
    test_arm()
    test_registration()
    print('=' * 40)
    print(f"✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        for n in FAIL:
            print(f'  FAIL: {n}')
        sys.exit(1)
