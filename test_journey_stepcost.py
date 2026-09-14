"""#46 旅程 Dijkstra 步数等价代价 单测 (2026-09-13)

bmv9 实弹 (logs/verify_v9_20260913.log, 52 步 6.95m max_steps 全程零目击):
图里旋转边 distance 均值 0.014m (128 条 literally 0.0) — _graph_path_to 旧按
米数计价 = 旋转免费, step18 救援重规划产出 17 边含 7×rotate_ccw ≈ 原地转
~330°, step19-33 烧 15 步近零位移 (HARD RULE 1b/1d 连环 B⑧/#40a 豁免可见)。
真实图复算 (navigation_graph.json): 旧代价 17 边 {rotate_ccw:7,backward:3,
left:6,rotate_cw:1}; 步数代价 (rot=1.0/trans=0.45) 14 边 est 10 步, 旋转只剩
真实改朝向所需的 6 条; #45 同位转向补全路径不变 (4×rotate_ccw)。

运行: python -u test_journey_stepcost.py
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


def mk_env(ff=None):
    """v9 病灶微缩图: 旋转边 distance=0.0 (真实图 128 条如此), 平移边 0.5m。

    OLD(米数): A→R1→R2→B 经 2 条免费旋转 + 1 条 0.1m 短左移 = 0.1m 最短;
    NEW(步数): A→M1→B 经 2 条平移 = 0.9 步 < 2×1.0+0.45 = 2.45 步。
    """
    from avdb_env import AVDBEnv
    e = object.__new__(AVDBEnv)
    e._ff = {'journey_stepcost': True}
    if ff:
        e._ff.update(ff)
    nodes = {
        'A':  {'world_pos': [0.0, 0.0, 0.0], 'direction': [0, 0, 1]},
        'R1': {'world_pos': [0.0, 0.0, 0.0], 'direction': [-0.5, 0, 0.86]},
        'R2': {'world_pos': [0.0, 0.0, 0.0], 'direction': [-0.87, 0, 0.5]},
        'M1': {'world_pos': [0.0, 0.0, 0.7], 'direction': [0, 0, 1]},
        'B':  {'world_pos': [0.0, 0.0, 1.4], 'direction': [0, 0, 1]},
    }
    graph = {
        'A':  {'R1': {'edge_type': 'rotate_ccw', 'distance': 0.0},
               'M1': {'edge_type': 'forward', 'distance': 0.7}},
        'R1': {'R2': {'edge_type': 'rotate_ccw', 'distance': 0.0}},
        'R2': {'B': {'edge_type': 'left', 'distance': 0.1}},
        'M1': {'B': {'edge_type': 'forward', 'distance': 0.7}},
        'B':  {},
    }
    e.simWrapper = type('SW', (), {
        'nav_graph': {'nodes': nodes, 'graph': graph}, 'current_node': 'A'})()
    return e


def test_step_cost():
    from avdb_env import AVDBEnv
    e = mk_env()

    # 1 步数代价: 免费旋转洪泛出局, 平移路胜出 (v9 病灶方向性复现)
    p = AVDBEnv._graph_path_to(e, 'A', 'B')
    check('#46: 步数代价 → 平移路 (forward×2), 不再吃免费旋转',
          p == ['forward', 'forward'], f'{p}')

    # 2 消融 --no-journey-stepcost: 回落旧米数代价 → 旋转路回归 (旧病理)
    e2 = mk_env({'journey_stepcost': False})
    p2 = AVDBEnv._graph_path_to(e2, 'A', 'B')
    check('#46: 消融 → 旧米数代价回归旋转路 (rotate_ccw×2+left)',
          p2 == ['rotate_ccw', 'rotate_ccw', 'left'], f'{p2}')

    # 3 真需要改朝向时旋转仍在路径里 (#45 同位转向补全语义不变)
    e3 = mk_env()
    p3 = AVDBEnv._graph_path_to(e3, 'A', 'R2')
    check('#46: 同位转向补全 → 纯旋转路径保持 (#45)',
          p3 == ['rotate_ccw', 'rotate_ccw'], f'{p3}')

    # 4 _ff 缺失 (object.__new__ 环境) → 默认开 (get('journey_stepcost', True))
    e4 = mk_env()
    del e4._ff
    p4 = AVDBEnv._graph_path_to(e4, 'A', 'B')
    check('#46: _ff 缺失 → 默认步数代价 (平移路)', p4 == ['forward', 'forward'])

    # 5 不可达 → None (原语义)
    e5 = mk_env()
    e5.simWrapper.nav_graph['graph']['Z'] = {}
    p5 = AVDBEnv._graph_path_to(e5, 'A', 'Z')
    check('#46: 不可达 → None 保持', p5 is None)


def test_log_and_flags():
    import feature_flags as F
    from avdb_env import AVDBEnv

    # 6 flag 注册 + CLI 剥离
    flags, rest = F.parse_argv(['--no-journey-stepcost', 'x'])
    check('#46: DEFAULTS True + --no-journey-stepcost 注册剥离',
          F.DEFAULTS.get('journey_stepcost') is True
          and F.FLAG_ARGS.get('--no-journey-stepcost') == ('journey_stepcost', False)
          and flags == {'journey_stepcost': False} and rest == ['x'])

    # 7 救援重规划日志成分披露 (17 edges: 7 rot + 10 trans, first rotate_ccw)
    SRC = open('/home/tao_h/VLMnav/src/avdb_env.py').read()
    check('#46: replan 日志披露 rot/trans 成分',
          "n_rot = sum(1 for x in ets if 'rotate' in x)" in SRC
          and '{n_rot} rot ' in SRC)

    # 8 代价常数接线在 _graph_path_to 内 (1.0/0.45), 不动其他 Dijkstra
    seg = SRC[SRC.index('def _graph_path_to'):SRC.index('def _p6_resolve')]
    check('#46: 步数常数 (rot 1.0 / trans 0.45) 只在 _graph_path_to',
          "if 'rotate' in et else 0.45" in seg
          and SRC.count("if 'rotate' in et else 0.45") == 1)


def test_real_graph_replay():
    """真实导航图回放 v9 病例: 17 边旋转洪泛 → 14 边/旋转骤减 (取证实锤)"""
    import json
    import os
    from avdb_env import AVDBEnv
    gp = '/home/tao_h/avdb_habitat_converter/data/processed/navigation_graph.json'
    if not os.path.exists(gp):
        check('#46: 真实图回放 (跳过: 图文件不在本机)', True)
        return
    g = json.load(open(gp))
    e = object.__new__(AVDBEnv)
    e._ff = {'journey_stepcost': True}
    e.simWrapper = type('SW', (), {
        'nav_graph': g, 'current_node': '000110014690101.jpg'})()
    a, b = '000110014690101.jpg', '000110009590101.jpg'
    p_new = AVDBEnv._graph_path_to(e, a, b)
    e._ff = {'journey_stepcost': False}
    p_old = AVDBEnv._graph_path_to(e, a, b)
    rot_new = sum(1 for x in p_new if 'rotate' in x)
    rot_old = sum(1 for x in p_old if 'rotate' in x)
    check('#46: 真实图 v9 病例 — 旧代价 17 边含 7 旋转 (复现)',
          p_old is not None and len(p_old) == 17 and rot_old == 8,
          f'old={len(p_old)} rot={rot_old}')
    check('#46: 真实图 v9 病例 — 步数代价 14 边且旋转 ≤6',
          p_new is not None and len(p_new) == 14 and rot_new <= 6,
          f'new={len(p_new)} rot={rot_new}')
    # #45 同位转向对 (v8 step20 病例) 路径不变
    e._ff = {'journey_stepcost': True}
    p45 = AVDBEnv._graph_path_to(e, '000110005600101.jpg', '000110005640101.jpg')
    check('#46: 真实图 #45 同位对 → 纯旋转补全保持',
          p45 is not None and len(p45) > 0
          and all('rotate' in x for x in p45), f'{p45}')


if __name__ == '__main__':
    test_step_cost()
    test_log_and_flags()
    test_real_graph_replay()
    print('=' * 40)
    print(f"✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        for n in FAIL:
            print(f'  FAIL: {n}')
        sys.exit(1)
