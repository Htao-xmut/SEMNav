"""#34 强制旅程完整性 单测 (2026-09-13, bmv4 mahatma 实弹)

用户报告: "没有实际解决，还在转圈！！！！都不去下其他区域。就在原地。
你从深度图也可以看出来啊"

bmv4 实弹 (logs/verify_v4_20260913.log) 三处实证 — 前沿跳其实发了, 旅程
却到不了:
  ① "no option for left, aborting" ×2 (10:28:54 / 10:33:27): 7-9 边的
    FRONTIER-HOP 旅程分别死在第 2/4 边 — 计划边是规划时节点链的
    edge_type 序列, 执行落点被定位吸附到别的节点 → 当前节点图边里
    没有该原语 → 整程作废;
  ② step18 困惑扫在 GOTO_WAYPOINT 旅程中途开火 (state=forced-return
    path active): obs 被扫描帧替换 → 计划边失配 (①的触发源之一),
    还白烧 ~7 次 VLM 调用;
  ③ 可疑点队列永不清空 (5 次销账 0 次 CLEAR, 0 次离区) — 见
    test_suspect_resolve.py #34 项。

修复 (#34):
  - _rescue_journey_edge 两级挽救: ① 当前节点→目的地节点重规划
    (_forced_return_dest 在 _plan_bearing_path 落盘) ② 仍缺 → 方位
    等价边 (世界方位差 ≤75° 的可用选项代走, 旅程下一节点重新对齐);
  - 困惑扫/周期扫/P4/换机位 在 _forced_return_path 非空时全部让路
    (旅程本身就是困惑的出路, 中途打断 = 原地打转的帮凶)。

运行: python -u test_journey_integrity.py
"""
import sys
sys.path.insert(0, '/home/tao_h/VLMnav/src')
import logging

logging.basicConfig(level=logging.CRITICAL)

PASS, FAIL = [], []


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f'  [{info}]' if info and not cond else ''))


def opt(et, cnt=1):
    return {'chain_type': et, 'chain_count': cnt, 'direction': et,
            'composite': None}


class SW:
    """nav_graph 桩: 节点世界坐标 + 图边 (edge_type/distance)"""

    def __init__(self, nodes, edges, cur):
        # nodes: {name: (x, z)}; edges: {u: {v: {'edge_type','distance'}}}
        self.nav_graph = {
            'nodes': {n: {'world_pos': (p[0], 0.0, p[1]),
                          'direction': (0.0, 0.0, 1.0)}
                      for n, p in nodes.items()},
            'graph': edges,
        }
        self.current_node = cur


def mk_env(nodes, edges, cur, dest, opts, ret_dest_attr='_forced_return_dest'):
    from avdb_env import AVDBEnv
    e = object.__new__(AVDBEnv)
    e.simWrapper = SW(nodes, edges, cur)
    setattr(e, ret_dest_attr, dest)
    e._forced_return_path = ['left']      # 计划边 (bmv4 病灶边型)
    return e, {'edge_options': opts}


# ========== 功能: _rescue_journey_edge (5 项) ==========
def test_rescue():
    from avdb_env import AVDBEnv

    # 1 重规划挽救: cur 无 'left' 原语, 但重规划首边 forward 可用
    #   图: A --forward--> B --forward--> D (目的地)
    e, obs = mk_env(
        nodes={'A': (0, 0), 'B': (0, 1), 'D': (0, 2)},
        edges={'A': {'B': {'edge_type': 'forward', 'distance': 1}},
               'B': {'D': {'edge_type': 'forward', 'distance': 1}},
               'D': {}},
        cur='A', dest='D', opts=[opt('forward'), opt('rotate_cw')])
    idx = AVDBEnv._rescue_journey_edge(e, obs, 'left')
    check('#34: 计划边失配 → 重规划挽救 (新首边可执行)',
          idx == 0 and e._forced_return_path == ['forward', 'forward'],
          f'idx={idx} path={e._forced_return_path}')

    # 2 方位等价边: 重规划首边也不可用 → 朝目的地方向的选项代走
    #   图: A --left--> B --left--> D, 选项只有 right; D 在 A 的 +x
    #   (世界 90°) → right (yaw0+90°) 方位差 0
    e, obs = mk_env(
        nodes={'A': (0, 0), 'B': (1, 0), 'D': (2, 0)},
        edges={'A': {'B': {'edge_type': 'left', 'distance': 1}},
               'B': {'D': {'edge_type': 'left', 'distance': 1}},
               'D': {}},
        cur='A', dest='D', opts=[opt('right')])
    idx = AVDBEnv._rescue_journey_edge(e, obs, 'left')
    check('#34: 重规划也缺 → 方位等价边代走 (Δ0° 朝目的地)',
          idx == 0, f'idx={idx}')

    # 3 已在目的地 0.8m 内 → 挽救放弃 = 旅程自然完成
    e, obs = mk_env(
        nodes={'A': (0, 0), 'B': (0, 0.4), 'D': (0, 0.5)},
        edges={'A': {'B': {'edge_type': 'forward', 'distance': 0.4}},
               'B': {'D': {'edge_type': 'forward', 'distance': 0.4}},
               'D': {}},
        cur='B', dest='D', opts=[opt('right')])
    check('#34: 目的地 <0.8m → 不再挽救 (旅程自然结束)',
          AVDBEnv._rescue_journey_edge(e, obs, 'left') is None)

    # 4 无目的地 (旧旅程未记 dest) → 不炸, 返回 None
    e, obs = mk_env(
        nodes={'A': (0, 0)}, edges={'A': {}},
        cur='A', dest=None, opts=[opt('forward')])
    check('#34: 无目的地 → 安全返回 None (不炸)',
          AVDBEnv._rescue_journey_edge(e, obs, 'left') is None)

    # 5 目击回访通道共用: _return_dest_node 同样作为目的地
    e, obs = mk_env(
        nodes={'A': (0, 0), 'B': (0, 1), 'S': (0, 2)},
        edges={'A': {'B': {'edge_type': 'forward', 'distance': 1}},
               'B': {'S': {'edge_type': 'forward', 'distance': 1}},
               'S': {}},
        cur='A', dest='S', opts=[opt('forward')],
        ret_dest_attr='_return_dest_node')
    idx = AVDBEnv._rescue_journey_edge(e, obs, 'left')
    check('#34: 目击回访 _return_dest_node 也接挽救',
          idx == 0 and e._forced_return_path == ['forward', 'forward'],
          f'idx={idx}')


# ========== 静态: 让路守卫 + 目的地落盘 + 复位 (4 项) ==========
def test_static():
    ENV_SRC = open('/home/tao_h/VLMnav/src/avdb_env.py').read()
    n_defer = ENV_SRC.count(
        "not getattr(self, '_forced_return_path', None)")
    check('#34: 困惑/周期/P4/换机位 四处旅程让路守卫在源',
          n_defer >= 4, f'count={n_defer}')
    check('#34: 失配弃程时目的地一并清',
          'no option for {next_et} ' in ENV_SRC
          and 'self._forced_return_dest = None' in ENV_SRC)
    check('#34: _plan_bearing_path 落盘目的地节点 (重规划依据)',
          'self._forced_return_dest = best_node' in ENV_SRC)
    check('#34: 回合复位不残留目的地', 'self._forced_return_dest = None' in ENV_SRC)


if __name__ == '__main__':
    test_rescue()
    test_static()
    print('=' * 40)
    print(f"✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        for n in FAIL:
            print(f'  FAIL: {n}')
        sys.exit(1)
