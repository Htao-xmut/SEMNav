"""#44 伺服落点位移几何 + #45 扫描目击朝向补全 单测 (2026-09-13)

bmv8 实弹 (logs/ObjectNav_bm8_mahatma8/0_of_1/0_Home_001_1/, 52 步 5.54m):
两个真目击 (GT 离线取证夹角 ≤1°) 全被几何谎言害死 —
  #44 走廊节点 canonical 朝向 −37° 但 forward 边位移正西 (−90°):
     伺服 "aligned 1deg" 实偏 ~50° → 目标出画 → miss 释锁 → #37 带回
     → 再偏 → TWO-STRIKES 拉黑真目标区 (step40) → 改道西南越走越远
  #45 step20 扫描目击 0.34@240°视角: 机器人同位置 0° 视角, 归位判定
     "已到" (③c) → 配对扑空 → 真目击蒸发无人转身看

运行: python -u test_servo_geometry.py
"""
import sys
sys.path.insert(0, '/home/tao_h/VLMnav/src')
sys.path.insert(0, '/home/tao_h/avdb_habitat_converter/scripts')
import logging
import numpy as np

logging.basicConfig(level=logging.CRITICAL)

PASS, FAIL = [], []


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f'  [{info}]' if info and not cond else ''))


def mk_env():
    """走廊图复刻 bmv8 病灶: A 朝向 −37° (西北) 但 forward 边位移正西"""
    from avdb_env import AVDBEnv
    e = object.__new__(AVDBEnv)
    e._ff = {'servo_geometry': True}
    nodes = {
        'A': {'world_pos': [0.0, 0.0, 0.0],
              'direction': [-0.6, 0.0, 0.8]},      # yaw ≈ −37°
        'W': {'world_pos': [-0.5, 0.0, 0.0],        # 正西 (位移 −90°)
              'direction': [-1.0, 0.0, 0.0]},
        'NW': {'world_pos': [0.5, 0.0, 0.5],        # 西北对角 (位移 +45°)
               'direction': [0.7, 0.0, 0.7]},
        'R': {'world_pos': [0.0, 0.0, 0.0],        # 纯旋转: 同位, 朝向 +90°
              'direction': [1.0, 0.0, 0.0]},
    }
    graph = {
        'A': {'W': {'edge_type': 'forward', 'distance': 0.5},
              'NW': {'edge_type': 'left', 'distance': 0.7},
              'R': {'edge_type': 'rotate_cw', 'distance': 0}},
        'R': {'A': {'edge_type': 'rotate_ccw', 'distance': 0}},
    }
    e.simWrapper = type('SW', (), {
        'nav_graph': {'nodes': nodes, 'graph': graph}, 'current_node': 'A'})()
    return e


def test_world_bearing():
    from avdb_env import AVDBEnv
    e = mk_env()

    # 1 forward 边位移正西 → 世界 −90° (旧模型: yaw+0 = −37°, 谎言源头)
    b = AVDBEnv._option_world_bearing_deg(
        e, {'chain_type': 'forward', 'chain_count': 1, 'composite': None})
    check('#44: forward 边 → 位移方位 −90° (旧模型谎报 −37°)',
          b is not None and abs(b - (-90.0)) < 1.0, f'{b}')

    # 2 left 边 → 位移 +45° (与节点朝向无关, 纯几何)
    b = AVDBEnv._option_world_bearing_deg(
        e, {'chain_type': 'left', 'chain_count': 1, 'composite': None})
    check('#44: left 边 → 位移方位 +45° (对角)',
          b is not None and abs(b - 45.0) < 1.0, f'{b}')

    # 3 纯旋转 → 落点节点朝向 +90° (转过去面对)
    b = AVDBEnv._option_world_bearing_deg(
        e, {'chain_type': 'rotate_cw', 'chain_count': 1, 'composite': None})
    check('#44: 纯旋转 → 落点朝向 +90°',
          b is not None and abs(b - 90.0) < 1.0, f'{b}')

    # 4 复合动作 (旋转后 forward) → 两腿走完的最终位移方位
    g = e.simWrapper.nav_graph
    g['graph']['R']['W'] = {'edge_type': 'forward', 'distance': 0.5}
    b = AVDBEnv._option_world_bearing_deg(
        e, {'composite': [('rotate_cw', 1), ('forward', 1)],
            'chain_type': None, 'chain_count': 0})
    check('#44: 复合 → R 落点再 forward 的总位移方位 (−90°)',
          b is not None and abs(b - (-90.0)) < 1.0, f'{b}')

    # 5 断链 (无边类型) → None (调用方回落旧模型)
    b = AVDBEnv._option_world_bearing_deg(
        e, {'chain_type': 'backward', 'chain_count': 1, 'composite': None})
    check('#44: 断链 → None 回落旧模型', b is None)

    # 6 消融 --no-servo-geometry: _ff 关闭 → 两 picker 不走几何分支 (静态)
    SRC = open('/home/tao_h/VLMnav/src/avdb_env.py').read()
    check('#44: 三处接线 (精确伺服/对齐选边/旅程等价边) + 消融门',
          SRC.count("get('servo_geometry', True)") == 3
          and 'def _option_world_bearing_deg' in SRC)


def test_scan_hit_orientation():
    from avdb_env import AVDBEnv
    e = mk_env()
    e._ff = {'servo_geometry': True}
    e.step = 20
    e._last_returned_sighting_step = -1
    e._forced_return_path = None
    e._return_dest_node = None
    e._arrival_check_step = None
    # 同位置不同朝向的目击节点 R; cur=A (位置相同)
    e.simWrapper.target_sighting_nodes = [(18, 'R', 'scan hit conf=0.34')]

    # 7 扫描目击 → 不再 ③c 跳过, 规划旋转路径转过去 (#45)
    r = AVDBEnv._start_forced_return(e)
    check('#45: 扫描目击同位置异朝向 → 旋转补全旅程 (非扑空配对)',
          r is True and e._forced_return_path == ['rotate_cw']
          and e._return_dest_node == 'R'
          and e._arrival_check_step is None)

    # 8 活体目击 → 保持 ③c 原语义 (当前帧即目击帧, 当步配对)
    e2 = mk_env()
    e2.step = 20
    e2._last_returned_sighting_step = -1
    e2._forced_return_path = None
    e2._return_dest_node = None
    e2._arrival_check_step = None
    e2.simWrapper.target_sighting_nodes = [(19, 'R', 'YOLO confirmed')]
    e2.simWrapper._node_pos = lambda n: np.array([0.0, 0.0])
    r2 = AVDBEnv._start_forced_return(e2)
    check('#45: 活体目击 → ③c 跳过保持 (当步配对, 不烧旋转步)',
          r2 is False and e2._arrival_check_step == 20
          and not e2._forced_return_path)

    # 9 _graph_path_to 旋转边可达 (朝向补全的前提)
    e3 = mk_env()
    e3.step = 20
    e3._last_returned_sighting_step = -1
    e3.simWrapper._node_pos = lambda n: np.array(
        [0.0, 0.0])  # 同位置判定用
    e3.simWrapper.target_sighting_nodes = [(18, 'R', 'scan hit conf=0.34')]
    r3 = AVDBEnv._start_forced_return(e3)
    check('#45: _node_pos 同位判定路径也走旋转补全',
          r3 is True and e3._forced_return_path == ['rotate_cw'])


def test_flags():
    # 10 flag 注册 + 剥离
    from feature_flags import DEFAULTS, FLAG_ARGS, parse_argv
    flags, rest = parse_argv(['--no-servo-geometry', 'x'])
    check('#44: DEFAULTS True + --no-servo-geometry 注册剥离',
          DEFAULTS.get('servo_geometry') is True
          and FLAG_ARGS.get('--no-servo-geometry') == ('servo_geometry', False)
          and flags == {'servo_geometry': False} and rest == ['x'])


if __name__ == '__main__':
    test_world_bearing()
    test_scan_hit_orientation()
    test_flags()
    print('=' * 40)
    print(f"✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        for n in FAIL:
            print(f'  FAIL: {n}')
        sys.exit(1)
