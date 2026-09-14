"""#41 旅程/探索多步平移 单测 (2026-09-13, bmv6 用户报障)

用户实弹指控 (logs/ObjectNav_bmv6_mahatmav6/0_of_1/0_Home_001_1/):
  "还有，安全距离完全够的话，直行什么一点一点移？"
  — SEMANTIC-GO 旅程执行器每步只弹 1 条图边 (~0.5-0.9m), VLM 侧也只有
    单边平移选项, 深度走廊通畅时依然一步步挪。

修复:
  wrapper `_stride_chain_hops` (图拓扑: 同型边最大链 ≤3 跳/≤2.4m)
    + 走廊通畅 (`corridor_free_ratio`, 只有深度观测 OCCUPIED 算阻挡,
      UNKNOWN≠不可达 — 零样本合规) → 提供链式选项;
  env `_execute_forced_return` 同向 K 边并步弹 K 条 (无链式选项回落
    单边, 原行为); `journey_stride` 消融开关。

运行: python -u test_journey_stride.py
"""
import sys
sys.path.insert(0, '/home/tao_h/VLMnav/src')
sys.path.insert(0, '/home/tao_h/avdb_habitat_converter/scripts')
import logging

logging.basicConfig(level=logging.CRITICAL)

PASS, FAIL = [], []


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f'  [{info}]' if info and not cond else ''))


# ========== wrapper 功能: _stride_chain_hops (4 项) ==========
def mk_wrapper(chain_len, edge_d=0.7):
    """N1 --forward--> N2 --forward--> ... 共 chain_len 条 forward 边"""
    from avdb_sim_wrapper import AVDBSimWrapper
    w = object.__new__(AVDBSimWrapper)
    nodes, graph = {}, {}
    names = [f'N{i}' for i in range(chain_len + 1)]
    for i, n in enumerate(names):
        nodes[n] = {'world_pos': [0.0, 0.0, 0.7 * i], 'direction': [0, 0, 1]}
    for i in range(chain_len):
        graph.setdefault(names[i], {})[names[i + 1]] = {
            'edge_type': 'forward', 'distance': edge_d}
        graph.setdefault(names[i + 1], {})[names[i]] = {
            'edge_type': 'backward', 'distance': edge_d}
    w.nav_graph = {'nodes': nodes, 'graph': graph}
    w.current_node = 'N0'
    return w


def test_chain_hops():
    # 1 三连边 → hops=3, 总长 2.1m
    w = mk_wrapper(3)
    edges = w.nav_graph['graph']['N0']
    hop, total = w._stride_chain_hops(edges, 'forward')
    check('#41: 三连 forward 边 → 3 跳 2.1m (首跳不双计)',
          hop == 3 and abs(total - 2.1) < 0.01, f'{hop},{total:.2f}')

    # 2 两连边 → 2 跳 1.4m
    w = mk_wrapper(2)
    hop, total = w._stride_chain_hops(w.nav_graph['graph']['N0'], 'forward')
    check('#41: 两连边 → 2 跳 1.4m', hop == 2 and abs(total - 1.4) < 0.01)

    # 3 链断 (只有 1 条边) → 1 跳 (无链, 不提供选项)
    w = mk_wrapper(1)
    hop, total = w._stride_chain_hops(w.nav_graph['graph']['N0'], 'forward')
    check('#41: 单边无链 → 1 跳', hop == 1)

    # 4 长度上限: 3×0.9=2.7>2.4 → 截到 2 跳
    w = mk_wrapper(3, edge_d=0.9)
    hop, total = w._stride_chain_hops(w.nav_graph['graph']['N0'], 'forward')
    check('#41: 总长超 2.4m 截链 (2×0.9=1.8 ≤2.4 < 2.7)',
          hop == 2 and abs(total - 1.8) < 0.01)

    # 5 无该类型边 → (1, 0)
    w = mk_wrapper(3)
    hop, total = w._stride_chain_hops(w.nav_graph['graph']['N0'], 'left')
    check('#41: 无该类型边 → 1 跳兜底', hop == 1 and total == 0.0)


# ========== env 功能: _execute_forced_return 并步 (5 项) ==========
OPTS = [
    {'chain_type': 'rotate_cw', 'chain_count': 1, 'composite': None},
    {'chain_type': 'forward', 'chain_count': 1, 'composite': None},
    {'chain_type': 'forward', 'chain_count': 3, 'composite': None},   # #41 链式
    {'chain_type': 'left', 'chain_count': 2, 'composite': None},      # #41 链式
    {'chain_type': 'left', 'chain_count': 1, 'composite': None},
]


def mk_env(path, opts=OPTS, flag=True):
    from avdb_env import AVDBEnv
    e = object.__new__(AVDBEnv)
    e._ff = {'journey_stride': flag}
    e.step = 9
    e._forced_return_path = list(path)
    e._forced_return_dest = 'D'
    e._arrival_check_step = None
    e.outer_run_name = e.inner_run_name = e.curr_run_name = 'x'
    e.ran = []

    def _ov(obs, idx, tag, note=''):
        e.ran.append((idx, tag, note))
        return ('ret', idx)

    e._override_and_run = _ov
    return e


def test_env_stride():
    from avdb_env import AVDBEnv

    # 6 三条 forward 边 + 链式选项在 → 并步弹 3 条选链式
    e = mk_env(['forward', 'forward', 'forward', 'left'])
    AVDBEnv._execute_forced_return(e, {'edge_options': OPTS})
    check('#41: 同向 3 边并步 (走廊通畅链式选项) → 弹 3 选 forward x3',
          e.ran[0][0] == 2 and e._forced_return_path == ['left'])

    # 7 链式选项不在 (走廊不通 wrapper 未生成) → 回落单边逐步 (原行为)
    e = mk_env(['forward', 'forward', 'left'],
               opts=[o for o in OPTS if o.get('chain_count', 1) == 1
                     or o['chain_type'] == 'rotate_cw'])
    AVDBEnv._execute_forced_return(e, {'edge_options': e._ff and OPTS[:1] + OPTS[1:2] + OPTS[4:]})
    check('#41: 无链式选项 → 回落单边 (一次只弹 1 条)',
          e.ran[0][0] == 1 and e._forced_return_path == ['forward', 'left'])

    # 8 混合边 (forward 后跟 left) → run=1 单边
    e = mk_env(['forward', 'left', 'left'])
    AVDBEnv._execute_forced_return(e, {'edge_options': OPTS})
    check('#41: 下一边不同型 → 不并步',
          e.ran[0][0] == 1 and e._forced_return_path == ['left', 'left'])

    # 9 left x2 也并: 路径 [left,left] + left x2 选项 → 弹 2 选链式, 空 → 到场检查
    e = mk_env(['left', 'left'])
    AVDBEnv._execute_forced_return(e, {'edge_options': OPTS})
    check('#41: 侧移同向 2 边并步 + 旅程收尾武装到场检查',
          e.ran[0][0] == 3 and e._forced_return_path == []
          and e._arrival_check_step == 10)

    # 10 旋转边永不并步 (多转无走廊语义)
    e = mk_env(['rotate_cw', 'rotate_cw', 'rotate_cw'])
    AVDBEnv._execute_forced_return(e, {'edge_options': [
        {'chain_type': 'rotate_cw', 'chain_count': 1, 'composite': None}]})
    check('#41: 旋转边不并步 (原行为)',
          e.ran[0][0] == 0 and len(e._forced_return_path) == 2)

    # 11 消融: --no-journey-stride → 有链式选项也不并步
    e = mk_env(['forward', 'forward', 'forward', 'left'], flag=False)
    AVDBEnv._execute_forced_return(e, {'edge_options': OPTS})
    check('#41: 消融 --no-journey-stride → 单边原行为',
          e.ran[0][0] == 1
          and e._forced_return_path == ['forward', 'forward', 'left'])


# ========== 静态: wrapper 选项生成 + 开关注册 (4 项) ==========
def test_static():
    SRC = open('/home/tao_h/avdb_habitat_converter/scripts/'
               'avdb_sim_wrapper.py').read()

    # 12 选项生成走 helper + 走廊门 (first_blk ≥ total+0.1)
    check('#41: wrapper 选项生成接线 _stride_chain_hops + corridor 门',
          'self._stride_chain_hops(edges, et)' in SRC
          and 'first_blk >= total_d + 0.1' in SRC
          and 'corridor_free_ratio' in SRC)

    # 13 链式选项 ≤3 跳 / ≤2.4m 上限在源
    check('#41: 上限 3 跳 / 2.4m 硬编码在方法签名',
          'max_hops=3, max_dist=2.4' in SRC)

    # 14 env 开关注入 simWrapper
    ENV_SRC = open('/home/tao_h/VLMnav/src/avdb_env.py').read()
    check('#41: env 注入 simWrapper.journey_stride',
          'self.simWrapper.journey_stride = bool(' in ENV_SRC)

    # 15 flag 注册 + 剥离
    from feature_flags import DEFAULTS, FLAG_ARGS, parse_argv
    flags, rest = parse_argv(['--no-journey-stride', 'x'])
    check('#41: DEFAULTS True + --no-journey-stride 注册剥离',
          DEFAULTS.get('journey_stride') is True
          and FLAG_ARGS.get('--no-journey-stride') == ('journey_stride', False)
          and flags == {'journey_stride': False} and rest == ['x'])


if __name__ == '__main__':
    test_chain_hops()
    test_env_stride()
    test_static()
    print('=' * 40)
    print(f"✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        for n in FAIL:
            print(f'  FAIL: {n}')
        sys.exit(1)
