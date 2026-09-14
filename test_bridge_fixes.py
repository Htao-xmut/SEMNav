"""A1/A2/B1 修复单测 — 旋转真转相机 / 地图绿格不降级 / 目击→停票桥

不启动模拟器, 不调真实 VLM (脚本化响应)。
运行: python -u test_bridge_fixes.py
"""
import sys, logging
sys.path.insert(0, '/home/tao_h/VLMnav/src')
sys.path.insert(0, '/home/tao_h/avdb_habitat_converter/scripts')
import numpy as np

logging.basicConfig(level=logging.WARNING)

from avdb_env import AVDBEnv
from avdb_sim_wrapper import AVDBSimWrapper
from avdb_depth import ExplorationMap2D
from avdb_agent import bridge_hint

PASS, FAIL = [], []


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f'  [{info}]' if info and not cond else ''))


# ---------- A1: _node_quat 旋转数学 ----------
def quat_rotate(q_wxyz, v):
    """(w,x,y,z) 四元数旋转向量 v"""
    w = q_wxyz[0]
    qv = np.array(q_wxyz[1:], float)
    v = np.asarray(v, float)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def test_a1_node_quat():
    w = object.__new__(AVDBSimWrapper)
    w.nav_graph = {'nodes': {
        'z+':  {'world_pos': [0, 0, 0],  'direction': [0, 0, 1]},
        'z-':  {'world_pos': [0, 0, 0],  'direction': [0, 0, -1]},
        'x+':  {'world_pos': [0, 0, 0],  'direction': [1, 0, 0]},
        'x-':  {'world_pos': [0, 0, 0],  'direction': [-1, 0, 0]},
        'diag': {'world_pos': [0, 0, 0], 'direction': [1, 0, 1]},
        'zero': {'world_pos': [0, 0, 0], 'direction': [0, 0, 0]},
    }}
    ok_all = True
    for name in ('z+', 'z-', 'x+', 'x-', 'diag'):
        q = w._node_quat(name)
        if q is None:
            ok_all = False
            continue
        got = quat_rotate(w._quat_wxyz(q), [0, 0, -1.0])   # 相机 forward=-z
        want = np.array(w.nav_graph['nodes'][name]['direction'], float)
        want = want / np.linalg.norm(want)
        if not np.allclose(got, want, atol=1e-5):
            ok_all = False
            print(f'   A1 mismatch {name}: got {got} want {want}')
    check('A1: _node_quat 五方向 旋转(0,0,-1) = 节点朝向', ok_all)
    check('A1: 未知节点 → None', w._node_quat('nope') is None)
    check('A1: 零向量朝向 → None', w._node_quat('zero') is None)
    w.nav_graph = None
    check('A1: 无导航图 → None', w._node_quat('z+') is None)


# ---------- A2: 地图绿格单调 ----------
def test_a2_map_monotonic():
    em = ExplorationMap2D(center=(0.0, 0.0), cell=0.05, size=600)
    # 图边前沿: (0,0)→(2,0) 整段灰 (explored_frac=0.0)
    em.update_graph_frontier([0, 0, 0], [[2, 0, 0]])
    g_far = em.regions[300, 340]                     # 2m 远端
    check('A2: 图边前沿先画灰 (region=2)', g_far == 2, f'got {g_far}')
    # 走过去: 同段画绿 (explored_frac 0.6 → 近端绿)
    em._draw_segment(0, 0, 2, 0, 0.6)
    g_near = em.regions[300, 315]                    # 0.75m 处
    check('A2: 行走后近端变绿 (region=1)', g_near == 1, f'got {g_near}')
    # 再次画图边前沿 (后到节点重复) → 绿不降级
    em.update_graph_frontier([2, 0, 0], [[0, 0, 0]])
    check('A2: 图边前沿重画不降级绿格', em.regions[300, 315] == 1,
          f'got {em.regions[300, 315]}')
    # _fill_poly region=2 覆盖绿格 → 保持绿; 覆盖未知 → 灰
    em._fill_poly([(0.5, -0.5), (0.5, 0.5), (1.5, 0.5), (1.5, -0.5)], region=2)
    check('A2: 灰四边形覆写后 绿格仍在', em.regions[300, 315] == 1)
    check('A2: 灰四边形写入未知区 → 灰', em.regions[290, 320] == 2)
    # seen 层照常累计 (决策查询用)
    em._fill_poly([(0, 0), (3.0, 0), (0, 3.0)], region=None)
    check('A2: seen 层 region=None 照常累计', bool(em.seen[300, 300]))


# ---------- B1a: wrapper 新鲜目击摘要 ----------
def test_b1_fresh_sighting():
    w = object.__new__(AVDBSimWrapper)
    w.target_sighting_nodes = [(7, 'N1', 'YOLO middle_center conf=0.80')]
    w.memory = {'step_count': 8}
    fs = w._fresh_sighting_info()
    check('B1: 新鲜目击 (1 步前) 返回摘要', fs is not None
          and fs['steps_ago'] == 1 and fs['node'] == 'N1')
    w.memory['step_count'] = 11                       # 4 步前 → 过期
    check('B1: 目击 4 步前 → None', w._fresh_sighting_info() is None)
    w.memory['step_count'] = 6                        # 未来步 (异常) → None
    check('B1: steps_ago<0 → None', w._fresh_sighting_info() is None)
    w.target_sighting_nodes = []
    check('B1: 无目击 → None', w._fresh_sighting_info() is None)


# ---------- B1b: 停票 prompt 桥 ----------
def test_b1_bridge_hint():
    h = bridge_hint({'steps_ago': 2, 'desc': 'VLM: on the table'})
    check('B1: 桥提示含 [BRIDGE] 标记', h.startswith('[BRIDGE]'))
    check('B1: 桥提示含步数与描述', '2 step(s) ago' in h and 'on the table' in h)
    check('B1: 桥提示要求 done=1 判据', 'answer done=1' in h)


# ---------- B1c: 仲裁到场配对 ----------
class FakeArrivalVLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def call_chat(self, *a, **kw):
        self.calls += 1
        return self.reply


def make_env(target='softsoap_white'):
    env = object.__new__(AVDBEnv)
    env.step = 9
    env.cfg = {'success_threshold': 1.0}
    env._sighting_frames = {'N1': np.zeros((540, 960, 3), np.uint8)}
    env._last_obs = {'color_sensor': np.zeros((540, 960, 3), np.uint8)}
    env._arrival_calls = 0

    class SW:
        target_name = target
        current_node = 'N1'
        nav_graph = {'nodes': {
            'N1': {'world_pos': [0, 0, 0], 'direction': [0, 0, 1]},
            'N2': {'world_pos': [3, 0, 0], 'direction': [0, 0, 1]}}}

        def _node_pos(self, n):
            p = self.nav_graph['nodes'][n]['world_pos']
            return np.array([p[0], p[2]])

    env.simWrapper = SW()
    return env


def test_b1_bridge_arbitration():
    # 1) 新鲜 + 在目击点 + 配对 close → 'close'
    env = make_env()
    env.simWrapper.target_sighting_nodes = [(8, 'N1', 'VLM: visually identified')]
    env.simWrapper.memory = {'step_count': 9}
    vlm = FakeArrivalVLM('{"close": 1, "where": "counter"}')
    env.agent = type('AG', (), {'actionVLM': vlm})()
    v = env._bridge_arrival_verdict(1.2)
    check('B1: 新鲜目击+在目击点+配对close → close', v == 'close')
    check('B1: 到场配对调了 1 次 VLM', vlm.calls == 1)
    # 2) 配对说 no → None (落回 far 轨道)
    env2 = make_env()
    env2.simWrapper.target_sighting_nodes = [(8, 'N1', 'x')]
    env2.simWrapper.memory = {'step_count': 9}
    vlm2 = FakeArrivalVLM('{"close": 0, "where": ""}')
    env2.agent = type('AG', (), {'actionVLM': vlm2})()
    check('B1: 配对 no → None (回无框 far 轨道)',
          env2._bridge_arrival_verdict(1.2) is None)
    # 3) 目击 4 步前 → None 且零 VLM 调用
    env3 = make_env()
    env3.simWrapper.target_sighting_nodes = [(5, 'N1', 'x')]
    env3.simWrapper.memory = {'step_count': 9}
    vlm3 = FakeArrivalVLM('{"close": 1}')
    env3.agent = type('AG', (), {'actionVLM': vlm3})()
    check('B1: 目击过期 (>3 步) → None', env3._bridge_arrival_verdict(1.2) is None)
    check('B1: 过期目击不烧 VLM 调用', vlm3.calls == 0)
    # 4) 已离开目击点 (3m) → None
    env4 = make_env()
    env4.simWrapper.current_node = 'N2'
    env4.simWrapper.target_sighting_nodes = [(8, 'N1', 'x')]
    env4.simWrapper.memory = {'step_count': 9}
    vlm4 = FakeArrivalVLM('{"close": 1}')
    env4.agent = type('AG', (), {'actionVLM': vlm4})()
    check('B1: 已离开目击点 → None', env4._bridge_arrival_verdict(1.2) is None)
    # 5) 目击点 0.3m 内 (相邻站位) + close → close
    env5 = make_env()
    env5.simWrapper.nav_graph['nodes']['N3'] = {
        'world_pos': [0.3, 0, 0], 'direction': [0, 0, 1]}
    env5.simWrapper.current_node = 'N3'
    env5.simWrapper.target_sighting_nodes = [(8, 'N1', 'x')]
    env5.simWrapper.memory = {'step_count': 9}
    vlm5 = FakeArrivalVLM('{"close": 1}')
    env5.agent = type('AG', (), {'actionVLM': vlm5})()
    check('B1: 目击点 0.3m 内 + close → close', env5._bridge_arrival_verdict(1.2) == 'close')
    # 6) 无留存帧 → None (零 VLM)
    env6 = make_env()
    env6._sighting_frames = {}
    env6.simWrapper.target_sighting_nodes = [(8, 'N1', 'x')]
    env6.simWrapper.memory = {'step_count': 9}
    vlm6 = FakeArrivalVLM('{"close": 1}')
    env6.agent = type('AG', (), {'actionVLM': vlm6})()
    check('B1: 无留存目击帧 → None', env6._bridge_arrival_verdict(1.2) is None)
    check('B1: 无帧不烧 VLM 调用', vlm6.calls == 0)


if __name__ == '__main__':
    test_a1_node_quat()
    test_a2_map_monotonic()
    test_b1_fresh_sighting()
    test_b1_bridge_hint()
    test_b1_bridge_arbitration()
    print(f"\n{'='*50}\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print('FAILED:', *FAIL, sep='\n  - ')
        sys.exit(1)
