"""#33 可疑点查过销账 单测 (2026-09-13, bmv3 mahatma 实弹)

用户报告: "还是在原地打转！！！为什么不去更可能的区域去探索下呢？
还有那么大片的区域还没探索啊！就在沙发、电视、窗户附近"

bmv3 实弹 (logs/verify_v3_20260913.log): LIKELY-BEARING 正常解析 5 次,
语义前沿成功 1 次 (4.3m mode=semantic) — 但离区通道被可疑点死循环堵死:
  zone (1,-2) 连续两次 SCAN_SUSPICIOUS: 0° → P6 自动转向查看 → 无 YOLO
  → 下次扫描 VLM 又在同一位置 (沙发/电视/窗) 找出"可疑点" → 再转 →
  再可疑 … zone 永远 CLEAR 不了 → zone_clear_exit 永不触发 → 困死原地。

修复 (#33):
  - P6 自动转向查看时记 _inspected_spots[cell] |= {angle//60} (60° 桶);
  - 再报同桶可疑 → 销账不排队 (日志"已查过");
  - 队列空 + 唯一可疑点已查 → 事实 CLEAR → 走 CLEAR 登记 (含 #31 目击
    史守卫) → 离区链路接通;
  - 真 SUSPICIOUS (未查过) 仍优先于 CLEAR (原语义)。

运行: python -u test_suspect_resolve.py
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
        self.target_sighting_nodes = []

    def _area_key(self, n):
        return self._k


def mk_env(inspected=None, spots=None, guard=True):
    from avdb_env import AVDBEnv
    e = object.__new__(AVDBEnv)
    e._ff = {'scan_inquiry': True, 'zone_clear_sight_guard': guard}
    e.step = 9
    e.simWrapper = SW()
    e._suspicious_spots = spots if spots is not None else {}
    e._inspected_spots = inspected if inspected is not None else {}
    return e


def test_resolve():
    from avdb_env import AVDBEnv

    # 1 未查过的可疑点 → 照常排队 (原行为)
    e = mk_env()
    ang, _ = AVDBEnv._parse_scan_inquiry(e, 'SCAN_SUSPICIOUS: 120', 'x')
    check('#33: 未查过 → 照常排队 (原行为)',
          ang == 120 and e._suspicious_spots.get((1, -2)))

    # 2 病灶: 已查过同桶 (0°↔bucket 0) + 队列空 → 事实 CLEAR 登记
    e = mk_env(inspected={(1, -2): {0}})
    ang, note = AVDBEnv._parse_scan_inquiry(
        e, 'SCAN_SUSPICIOUS: 20', 'x')     # 20//60 = 0 同桶
    check('#33: 已查过同桶 → 销账 + 事实 CLEAR 登记 (bmv3 病灶)',
          ang is None and (1, -2) in getattr(e, '_zone_clear_cells', set())
          and 'nothing suspicious' in note)

    # 2b #34 (bmv4 病灶): 队列里有同桶旧条目 → 出队后空 → 事实 CLEAR
    #    (bmv4 实弹: 5 次销账 0 次 CLEAR, 0 次离区 — 旧条目永不出队)
    e = mk_env(inspected={(1, -2): {0}}, spots={(1, -2): [(7, 20)]})
    ang, _ = AVDBEnv._parse_scan_inquiry(e, 'SCAN_SUSPICIOUS: 40', 'x')
    check('#34: 同桶旧排队条目出队 → 队列空 → 事实 CLEAR (bmv4 病灶)',
          ang is None and e._suspicious_spots[(1, -2)] == []
          and (1, -2) in getattr(e, '_zone_clear_cells', set()))

    # 3 已查过但队列还有别的可疑点 → 只销账不 CLEAR (还有要查的)
    e = mk_env(inspected={(1, -2): {0}},
               spots={(1, -2): [(7, 120)]})
    ang, _ = AVDBEnv._parse_scan_inquiry(e, 'SCAN_SUSPICIOUS: 20', 'x')
    check('#33: 队列尚有未查点 → 只销账不判 CLEAR',
          ang is None and (1, -2) not in getattr(e, '_zone_clear_cells', set())
          and e._suspicious_spots[(1, -2)] == [(7, 120)])

    # 4 别桶已查 → 同 zone 不同方位仍排队 (只销已查桶)
    e = mk_env(inspected={(1, -2): {2}})   # 120° 桶已查
    ang, _ = AVDBEnv._parse_scan_inquiry(e, 'SCAN_SUSPICIOUS: 40', 'x')
    check('#33: 只销已查桶, 别的方位照常排队',
          ang == 40 and e._suspicious_spots.get((1, -2)))

    # 5 #31 联动: 已查桶销账成 CLEAR, 但 zone 有目击史 → 不登记 (守卫仍生效)
    e = mk_env(inspected={(1, -2): {0}})
    e.simWrapper.target_sighting_nodes = [(5, 'n0', 'YOLO conf=0.61')]
    AVDBEnv._parse_scan_inquiry(e, 'SCAN_SUSPICIOUS: 20', 'x')
    check('#33: 销账 CLEAR 也过 #31 目击史守卫',
          (1, -2) not in getattr(e, '_zone_clear_cells', set()))

    # 6 真 SUSPICIOUS 优先于 CLEAR (同响应两者都在, 原语义)
    e = mk_env()
    ang, note = AVDBEnv._parse_scan_inquiry(
        e, 'SCAN_SUSPICIOUS: 60\nSCAN_CLEAR', 'x')
    check('#33: 真可疑仍优先于 CLEAR (原语义不破)',
          ang == 60 and (1, -2) not in getattr(e, '_zone_clear_cells', set())
          and 'suspicious spot' in note)


def test_static():
    ENV_SRC = open('/home/tao_h/VLMnav/src/avdb_env.py').read()
    check('#33: P6 自动转向处记已查 (60° 桶)',
          '_inspected_spots' in ENV_SRC
          and '_mark_inspected(new_angle // 60)' in ENV_SRC)
    check('#33: 回合复位不残留', 'self._inspected_spots = {}' in ENV_SRC)
    check('#33: elif→守卫 if (销账后接 CLEAR 登记)',
          'if m_clear and not m_s:' in ENV_SRC)
    check('#34: P6 记录处同桶排队条目同步出队 (队列能自然清空)',
          'if e[1] // 60 != bucket' in ENV_SRC)


if __name__ == '__main__':
    test_resolve()
    test_static()
    print('=' * 40)
    print(f"✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        for n in FAIL:
            print(f'  FAIL: {n}')
        sys.exit(1)
