"""阶段0 单测 — FSM 显式化 + 修复1(身份门三级化) + 修复2(停票链保护) + 修复3(房间标签中和)

设计稿: 任务/阶段0_FSM设计稿.md v1.2 §6.1
不启动模拟器, 不调真实 VLM。运行: python -u test_fsm_phase0.py
"""
import sys, logging
sys.path.insert(0, '/home/tao_h/VLMnav/src')
sys.path.insert(0, '/home/tao_h/avdb_habitat_converter/scripts')
import numpy as np
import pandas as pd
from types import SimpleNamespace as NS

logging.basicConfig(level=logging.INFO)

from avdb_env import AVDBEnv
from avdb_sim_wrapper import AVDBSimWrapper, PolarAction
import feature_flags

PASS, FAIL = [], []


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f'  [{info}]' if info and not cond else ''))


# ---------- 公共假体 ----------
class FakeSW:
    """wrapper 假体 — 只实现身份门/武装路径触碰的接口"""

    def __init__(self, current='N1'):
        self.nav_graph = {'nodes': {
            'N1': {'world_pos': [0.5, 0, 1.2], 'direction': [0, 0, 1]},
            'N2': {'world_pos': [3.0, 0, -2.5], 'direction': [0, 0, 1]},
        }, 'graph': {'N1': {}, 'N2': {}}}
        self.current_node = current
        self.target_name = 'cola'
        self.target_sighting_nodes = [(9, 'N1', 'YOLO conf=0.55')]
        self.rejected_areas = []
        self.rejections = []
        self.memory = {'step_count': 10}
        self.visited_nodes = {'N1': 1}

    def _node_pos(self, n):
        p = self.nav_graph['nodes'][n]['world_pos']
        return np.array([p[0], p[2]])

    def _node_in_rejected(self, n):
        return False, None

    def record_rejection(self, node, reason):
        self.rejections.append((node, reason))

    def _fresh_sighting_info(self):
        if not self.target_sighting_nodes:
            return None
        st = self.memory['step_count']
        s_step, s_node, s_desc = self.target_sighting_nodes[-1]
        steps_ago = st - s_step
        if not (0 <= steps_ago <= 3):
            return None
        return {'steps_ago': steps_ago, 'node': s_node, 'desc': s_desc}


def make_env(step=10, ff=None):
    env = object.__new__(AVDBEnv)
    env.step = step
    env._ff = {'identity_gate_v2': True, 'vote_fastpath': True, **(ff or {})}
    env.cfg = {'success_threshold': 1.0, 'sensor_cfg': {'fov': 131}}
    env._identity_reject = {}
    env._wrong_strikes = {}
    env._approach_bearing = 0.5
    env._approach_miss = 0
    env._approach_last_area = 0.5
    env._forced_return_path = []
    env._arrival_check_step = -99
    env._loss_strikes = []
    env._last_returned_sighting_step = -1
    env._last_identity_step = -99
    env._arbitration_close = False
    env.df = pd.DataFrame()
    env.agent = NS(stop_history=[], fastpath_armed=False,
                   last_stop_was_fastpath=False)
    env.simWrapper = FakeSW()
    return env


def obs_with(area=0.03, conf=0.55, found=True, with_options=False):
    w = max(0.001, area ** 0.5)
    det = {'class_name': 'cola', 'confidence': conf,
           'bbox_norm': [0.4, 0.4, 0.4 + w, 0.4 + w]}
    opts = [{'chain_type': 'rotate_cw', 'chain_count': 1, 'composite': None}] \
        if with_options else []
    return {'yolo_detection': {'target_found': found, 'confidence': conf,
                               'position': 'middle_center', 'area_ratio': area,
                               'all_detections': [det]},
            'edge_options': opts, 'depth_rec_deg': None,
            'color_sensor': np.zeros((480, 640, 3), np.uint8)}


class LogCap(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def with_cap(fn):
    cap = LogCap()
    root = logging.getLogger()
    root.addHandler(cap)
    try:
        fn(cap)
    finally:
        root.removeHandler(cap)


# ========== 1. FSM 状态派生 (设计稿 §2.1, 8 项) ==========
def test_fsm_derivation():
    # 1 GOTO_WAYPOINT: 回访路径非空 (即使接近锁同时在 — 让位不变量)
    e = make_env()
    e._forced_return_path = ['forward']
    e._approach_bearing, e._approach_miss = 0.3, 0
    s, r = e._fsm_state()
    check('FSM: 回访路径非空 → GOTO_WAYPOINT', s == 'GOTO_WAYPOINT')
    check('FSM: GOTO 期间带锁也判 GOTO (让位不变量)', 'forced' in r)
    # 2 CONFIRM_SIGHTING: 到场检查武装于本步 (路径已空)
    e = make_env()
    e._arrival_check_step = e.step
    check('FSM: 到场检查武装 → CONFIRM_SIGHTING',
          e._fsm_state()[0] == 'CONFIRM_SIGHTING')
    # 3 一次性消费: 上一步武装的检查步 ≠ 当前步 → 不再 CONFIRM
    e._arrival_check_step = e.step - 1
    check('FSM: 到场检查已消费 → 非 CONFIRM (落入 APPROACH)',
          e._fsm_state()[0] == 'APPROACH')
    # 4 APPROACH: 锁定 + miss≤10
    e = make_env()
    e._approach_bearing, e._approach_miss = 0.3, 5
    check('FSM: 锁定+miss≤10 → APPROACH', e._fsm_state()[0] == 'APPROACH')
    # 5 APPROACH 边界: miss=11 → 失效
    e._approach_miss = 11
    check('FSM: miss=11 (超预算) → EXPLORE', e._fsm_state()[0] == 'EXPLORE')
    # 6 EXPLORE: 无任何机制
    e = make_env()
    e._approach_bearing = None
    check('FSM: 无锁/无回访/无检查 → EXPLORE', e._fsm_state()[0] == 'EXPLORE')
    # 7 env 侧永不派生 ESCAPE (wrapper 硬规则才标记)
    check('FSM: env 派生不含 ESCAPE',
          e._fsm_state()[0] in ('GOTO_WAYPOINT', 'CONFIRM_SIGHTING',
                                'APPROACH', 'EXPLORE'))
    # 8 派生为纯函数: 连续两次调用同结果, 无属性副作用
    before = (e._forced_return_path, e._approach_bearing, e._approach_miss,
              e._arrival_check_step)
    s1, s2 = e._fsm_state()[0], e._fsm_state()[0]
    after = (e._forced_return_path, e._approach_bearing, e._approach_miss,
             e._arrival_check_step)
    check('FSM: 纯函数 (两次调用同值, 零副作用)', s1 == s2 and before == after)


# ========== 2. 修复1: 身份门三级化 (设计稿 §3, 11 项) ==========
def test_identity_gate_v2():
    # 1 L1 首次 wrong: 不拉黑 + 节点抑制窗 + 记 strike + 解锁 + 不动丢失strike
    e = make_env(step=10)
    e._penalize_proximity_wrong(obs_with(area=0.03, conf=0.55))
    sw = e.simWrapper
    h1 = e._wrong_strikes.get('N1', [])
    check('修1-L1: 首次 wrong 零 record_rejection', sw.rejections == [])
    check('修1-L1: 节点进抑制窗 (_identity_reject N1=10)',
          e._identity_reject.get('N1') == 10)
    check('修1-L1: strike 已记 (step=10, area≈0.03, conf≈0.55)',
          len(h1) == 1 and h1[0][0] == 10
          and abs(h1[0][1] - 0.03) < 1e-9 and abs(h1[0][2] - 0.55) < 1e-9)
    check('修1-L1: 锁已释放 (bearing=None)', e._approach_bearing is None)
    check('修1-L1: 丢失strike体系未被触碰 (loss_strike=False)',
          e._loss_strikes == [])

    # 2 L2 升格: 第二次 wrong, 面积≥2% + conf≥0.5 + 间隔≥10 → 永久拉黑
    #   (第一次 L1 已弹掉唯一目击 → 补一条新目击, 拉黑才有对象)
    e.step = 25   # 距首击 15 步 ≥ 冷却 10
    e.simWrapper.target_sighting_nodes = [(24, 'N1', 'YOLO conf=0.80')]
    e._penalize_proximity_wrong(obs_with(area=0.05, conf=0.80))
    check('修1-L2: 强证据二次 wrong → record_rejection 恰 1 次',
          len(sw.rejections) == 1 and sw.rejections[0][0] == 'N1')
    check('修1-L2: strike 记 2 条', len(e._wrong_strikes['N1']) == 2)

    # 3 L2 不升格 — 间隔<冷却 (连续重复错)
    e = make_env(step=10)
    e._penalize_proximity_wrong(obs_with(area=0.03, conf=0.55))
    e.step = 15   # 间隔 5 < 10
    e._penalize_proximity_wrong(obs_with(area=0.09, conf=0.90))
    check('修1-L2: 间隔<冷却 → 不升格不拉黑 (ep8 防线)',
          e.simWrapper.rejections == [])
    check('修1-L2: 间隔<冷却 → 不计新 strike',
          len(e._wrong_strikes['N1']) == 1)

    # 4 L2 不升格 — 面积<2%
    e = make_env(step=10)
    e._penalize_proximity_wrong(obs_with(area=0.03, conf=0.55))
    e.step = 30
    e._penalize_proximity_wrong(obs_with(area=0.01, conf=0.90))
    check('修1-L2: 面积<2% → 停留 L1 (小框否认不足以拉黑)',
          e.simWrapper.rejections == [] and len(e._wrong_strikes['N1']) == 2)

    # 5 L2 不升格 — conf<0.5 (防御性: 入口门隐含, 防将来放宽)
    e = make_env(step=10)
    e._penalize_proximity_wrong(obs_with(area=0.03, conf=0.55))
    e.step = 30
    e._penalize_proximity_wrong(obs_with(area=0.09, conf=0.45))
    check('修1-L2: conf<0.5 → 停留 L1 (防御性不变量)',
          e.simWrapper.rejections == [])

    # 6 L3 耗尽拉黑: 3 次 wrong 每次间隔≥冷却 (全小框也拉黑 — 防回环上界)
    e = make_env(step=0)
    e._wrong_strikes = {'N1': [(0, 0.001, 0.55), (12, 0.002, 0.55)]}
    e.step = 30   # 距第2击 18 ≥ 10

    def _l3(cap):
        e._penalize_proximity_wrong(obs_with(area=0.005, conf=0.60))
        l3 = [l for l in cap.lines if 'L3 EXHAUSTED' in l]
        check('修1-L3: 3击耗尽 → 永久拉黑',
              len(e.simWrapper.rejections) == 1)
        check('修1-L3: 全量 strike 日志 (3 条 area= 记录 + 间隔)',
              len(l3) == 1 and l3[0].count('area=') == 3
              and 'intervals [12, 18]' in l3[0])
    with_cap(_l3)

    # 7 strike 上界: 历史裁剪 ≤6 (新击入库, 最老出库)
    e = make_env(step=100)
    e._wrong_strikes = {'N1': [(i * 16, 0.001, 0.5) for i in range(6)]}
    e._penalize_proximity_wrong(obs_with(area=0.005, conf=0.6))
    h = e._wrong_strikes['N1']
    check('修1: strike 历史裁剪 ≤6 条 (新击入库最老出库)',
          len(h) == 6 and h[-1][0] == 100)

    # 8 PROXIMITY 抑制门: L1 窗内节点不触发检查
    e = make_env(step=12)
    e._identity_reject = {'N1': 10}
    check('修1: L1 窗内 (2步前) → PROXIMITY 抑制', e._proximity_suppressed())
    e.step = 25
    check('修1: 窗外 (15步前) → 不抑制', not e._proximity_suppressed())
    e._ff = {**e._ff, 'identity_gate_v2': False}
    e.step, e._identity_reject = 12, {'N1': 10}
    check('修1: 开关关 → 不抑制 (旧行为)', not e._proximity_suppressed())
    e._ff = {**e._ff, 'identity_gate_v2': True}
    e.simWrapper.current_node = None
    check('修1: 无节点 → 不抑制', not e._proximity_suppressed())

    # 9 开关关时旧行为保留 (消融对照): wrong → 立即拉黑
    e = make_env(step=10, ff={'identity_gate_v2': False})
    e._invalidate_lock('VLM rejected detection (wrong object)')
    check('修1: 开关关 → 旧行为 (一次 wrong 即拉黑)',
          len(e.simWrapper.rejections) == 1)


# ========== 3. 修复2 Layer 1: 停票链保护 (设计稿 §4, 3 项) ==========
def test_vote_protect_layer1():
    def setup(streak):
        e = make_env(step=20)
        e.agent.stop_history = [False, streak]
        calls = []

        def fake_override(obs, idx, tag, note=''):
            calls.append(tag)
            return PolarAction(0, 0)
        e._override_and_run = fake_override
        e._vlm_identity_check = lambda obs, yi: 'unsure'
        e.simWrapper._approach_active = False
        return e, calls

    # obs 带 rotate_cw 选项: CONFIRM 才有可选动作 (无选项时 pick 返回 None,
    # 非压制用例也发不出 CONFIRM)
    obs = obs_with(area=0.05, conf=0.40, with_options=True)

    # 1 streak≥1 + 有框 (分支前提) → CONFIRM 相机动作被压制
    #   (压制后落入正常伺服流, APPROACH 覆盖是正常行为 — 断言只看无 CONFIRM)
    e, calls = setup(True)

    def _run(cap):
        e._approach_step(obs)
        check('修2-L1: streak≥1+框 → CONFIRM 相机动作被压制',
              'CONFIRM' not in calls)
        check('修2-L1: 压制时打 [VOTE-PROTECT] 日志',
              any('VOTE-PROTECT' in l for l in cap.lines))
    with_cap(_run)

    # 2 streak=0 → CONFIRM 正常执行 (early return, 无后续伺服)
    e, calls = setup(False)
    e._approach_step(obs)
    check('修2-L1: streak=0 → CONFIRM 正常执行', calls == ['CONFIRM'])

    # 3 开关关 + streak≥1 → CONFIRM 不受压制 (消融对照)
    e, calls = setup(True)
    e._ff = {**e._ff, 'vote_fastpath': False}
    e._approach_step(obs)
    check('修2-L1: 开关关 → 不压制 (旧行为)', calls == ['CONFIRM'])


# ========== 4. 修复2 Layer 2: 桥接快通道 (设计稿 §4, 10 项) ==========
def test_vote_fastpath_layer2():
    # --- 武装 _arm_vote_fastpath ---
    # 1 ep2 实测场景: 桥活跃(1步前) + 框 conf 0.36 / area 0.00 → OR 语义通过
    e = make_env(step=25)
    e.simWrapper.target_sighting_nodes = [(24, 'N1', 'YOLO conf=0.36')]
    e.simWrapper.memory = {'step_count': 25}
    e._arm_vote_fastpath(obs_with(area=0.004, conf=0.36))
    ctx1 = e._fastpath_ctx or {}
    check('修2-L2: ep2 场景 (conf 0.36/area 0.00) OR 门通过 → 武装',
          e.agent.fastpath_armed is True and ctx1.get('bridge_left') == 2)
    check('修2-L2: 武装上下文 (area/conf/bridge_left)',
          abs(ctx1.get('area', -1) - 0.004) < 1e-6
          and abs(ctx1.get('conf', -1) - 0.36) < 1e-9)

    # 2 质量门双不过 (conf 0.25 + area 0.005) → 不武装
    e._arm_vote_fastpath(obs_with(area=0.005, conf=0.25))
    check('修2-L2: conf 0.25+area 0.005 双低于门 → 不武装',
          e.agent.fastpath_armed is False)

    # 3 无框 → 不武装
    e._arm_vote_fastpath(obs_with(found=False))
    check('修2-L2: 无框 → 不武装', e.agent.fastpath_armed is False)

    # 4 桥过期 (目击 4 步前) → 不武装
    e.simWrapper.target_sighting_nodes = [(21, 'N1', 'x')]
    e.simWrapper.memory = {'step_count': 25}
    e._arm_vote_fastpath(obs_with(area=0.05, conf=0.80))
    check('修2-L2: 目击 4 步前 (桥过期) → 不武装',
          e.agent.fastpath_armed is False)

    # 5 开关关 → 不武装
    e._ff = {**e._ff, 'vote_fastpath': False}
    e.simWrapper.target_sighting_nodes = [(24, 'N1', 'x')]
    e._arm_vote_fastpath(obs_with(area=0.05, conf=0.80))
    check('修2-L2: 开关关 → 不武装', e.agent.fastpath_armed is False)
    e._ff = {**e._ff, 'vote_fastpath': True}

    # --- 计量 _fastpath_metrics ---
    # 6 触发+终判 close: 计数器与五字段日志
    e2 = make_env(step=25)
    e2._fastpath_ctx = {'area': 0.004, 'conf': 0.36, 'bridge_left': 2}
    e2.agent.last_stop_was_fastpath = True
    e2._ep_stats = {'fastpath_triggered': 0, 'fastpath_final_close': 0}

    def _m1(cap):
        e2._fastpath_metrics(PolarAction.stop,
                             {'finish_status': 'success'})
        fp = [l for l in cap.lines if '[FASTPATH]' in l]
        check('修2-L2: 触发+close → triggered/final_close 各+1',
              e2._ep_stats == {'fastpath_triggered': 1,
                               'fastpath_final_close': 1})
        check('修2-L2: [FASTPATH] 日志五字段齐全',
              len(fp) >= 1 and all(k in fp[-1] for k in
                                   ('step=25', 'area=0.004', 'conf=0.36',
                                    'bridge_left=2', 'vote=1', '最终判定 close')))
    with_cap(_m1)

    # 7 终判 far (假阳性): 只计 triggered
    e2._ep_stats = {'fastpath_triggered': 0, 'fastpath_final_close': 0}
    e2._fastpath_metrics(PolarAction.stop, {'finish_status': 'running'})
    check('修2-L2: 终判 far → 只计 triggered (假阳性=差值)',
          e2._ep_stats == {'fastpath_triggered': 1,
                           'fastpath_final_close': 0})
    # 8 非快通道 stop / 非stop → 零计量
    e2._ep_stats = {'fastpath_triggered': 0, 'fastpath_final_close': 0}
    e2.agent.last_stop_was_fastpath = False
    e2._fastpath_metrics(PolarAction.stop, {'finish_status': 'success'})
    e2._fastpath_metrics(PolarAction(0, 0), {'finish_status': 'success'})
    check('修2-L2: 非快通道 stop / 非 stop → 零计量',
          e2._ep_stats['fastpath_triggered'] == 0)

    # --- 防吞票: override 内快通道 stop 不被改写为覆盖动作 ---
    import env as env_mod
    orig_step = env_mod.ObjectNavEnv._step_env
    try:
        env_mod.ObjectNavEnv._step_env = lambda self, obs: PolarAction.stop
        # 9 快通道 stop 在 override 内 → 转当步接近移动 (非覆盖动作)
        e3 = make_env(step=25)
        e3.agent.last_stop_was_fastpath = True
        veto_calls = []

        def fake_veto(obs):
            veto_calls.append(1)
            a = PolarAction(0, 0)
            a.edge_idx = 7
            return a
        e3._vetoed_stop_action = fake_veto
        e3._redraw_chosen_as_executed = lambda *a, **k: None
        act = e3._override_and_run(obs_with(), 3, 'CONFIRM', 'test')
        check('修2-L2: override 内快通道 stop 不被吞 (ep2 step25 型)',
              veto_calls == [1] and getattr(act, 'edge_idx', None) == 7)
        # 10 非快通道 stop 在 override 内 → 旧行为保留 (仍执行覆盖动作)
        e4 = make_env(step=25)
        e4.agent.last_stop_was_fastpath = False
        e4._redraw_chosen_as_executed = lambda *a, **k: None
        act = e4._override_and_run(obs_with(), 3, 'CONFIRM', 'test')
        check('修2-L2: 非快通道 stop → 旧行为 (执行覆盖动作 idx=3)',
              getattr(act, 'edge_idx', None) == 3)
    finally:
        env_mod.ObjectNavEnv._step_env = orig_step


# ========== 5. 修复3: 房间标签中和 (设计稿 §5, 7 项) ==========
ROOM_WORDS = ('kitchen', 'living room', 'hallway', 'dining',
              'bedroom', 'bathroom')


def test_neutral_rooms():
    w = object.__new__(AVDBSimWrapper)
    w.nav_graph = {'nodes': {
        'N1': {'world_pos': [0.5, 0, 1.2], 'direction': [0, 0, 1]},
        'N2': {'world_pos': [3.0, 0, -2.5], 'direction': [0, 0, 1]},
        'N3': {'world_pos': [0.0, 0, 4.0], 'direction': [0, 0, 1]},
    }, 'graph': {'N1': {}, 'N2': {}, 'N3': {}}}

    # 1 默认 (未设属性) 即中性: zone_{i}_{j} 且无房间词
    z1 = w._guess_room('N1')
    z2 = w._guess_room('N2')
    check('修3: 默认中性 — N1(0.5,1.2) → zone_0_0', z1 == 'zone_0_0', f'got {z1}')
    check('修3: N2(3.0,-2.5) → zone_1_-2 (负格保留)', z2 == 'zone_1_-2',
          f'got {z2}')
    check('修3: 中性标签零房间词',
          not any(rw in z1 or rw in z2 for rw in ROOM_WORDS))
    # 2 neutral_rooms=False → 旧阈值标签 (离线诊断侧)
    w.neutral_rooms = False
    check('修3: 开关关 → 旧标签 N3(z=4.0) → kitchen/dining',
          w._guess_room('N3') == 'kitchen/dining')
    w.neutral_rooms = True
    check('修3: 开关开 → N3 → zone_0_2 (无语义)',
          w._guess_room('N3') == 'zone_0_2', f"got {w._guess_room('N3')}")
    # 3 未知节点 → 'unknown' (两模式一致)
    check('修3: 图外节点 → unknown (两模式一致)',
          w._guess_room('nope') == 'unknown')
    w.neutral_rooms = False
    check('修3: 图外节点 → unknown (legacy 同)', w._guess_room('nope') == 'unknown')
    w.neutral_rooms = True
    # 4 重放稳定: 同一节点序列两遍 → 同一 zone 序列 (不依赖任何 GT 房间字段)
    seq = ['N1', 'N2', 'N3', 'N1', 'N2']
    r1 = [w._guess_room(n) for n in seq]
    r2 = [w._guess_room(n) for n in seq]
    check('修3: 轨迹重放 → zone 序列稳定', r1 == r2 and len(set(r1)) == 3)
    # 5 坐标源 = 图节点 world_pos (非 GT env 状态): 仅 nav_graph 即可工作
    w2 = object.__new__(AVDBSimWrapper)   # 无 sim/无 env/无 GT 任何字段
    w2.nav_graph = {'nodes': {'N1': {'world_pos': [0.5, 0, 1.2]}}}
    check('修3: 仅图节点坐标即可派生 (零 GT 依赖)',
          w2._guess_room('N1') == 'zone_0_0')
    # 6 prompt 侧不可见旧标签: get_memory_context 输出无房间词 + 含中性标签
    w.memory = {'step_count': 3, 'rooms': {}, 'short_term': [],
                'adjacent': {}, 'target_hints': {
                    'visual_features': '', 'last_seen_step': None,
                    'last_seen_room': None}}
    w.node_records = {'N1': {'state': 'empty', 'visits': 2, 'first_step': 0,
                             'last_step': 3, 'desc': ''}}
    w.current_node, w.target_name = 'N1', 'cola'
    w.rejected_areas = []
    w.target_sighting_nodes = []
    w.visited_nodes = {'N1': 2}
    w.yolo_history = []
    w.neutral_rooms = True
    try:
        ctx = w.get_memory_context()
        check('修3: get_memory_context 输出零房间词 (红线)',
              not any(rw in ctx for rw in ROOM_WORDS))
        check('修3: 输出含中性 zone 标签 (VISITED AREAS 行)',
              'zone_0_0' in ctx)
    except Exception as ex:
        check('修3: get_memory_context 输出零房间词 (红线)', False, str(ex))
        check('修3: 输出含中性 zone 标签 (VISITED AREAS 行)', False, str(ex))


# ========== 6. 优先级表 / 监控落地 / 开关 (v1.2, 6 项) ==========
def test_monitoring_and_flags():
    # 1 _step_concluded: df 尾行判定
    e = make_env()
    e.df = pd.DataFrame([{'finish_status': 'running'}])
    check('监控: 尾行 running → 未终局', e._step_concluded() is False)
    e.df = pd.DataFrame([{'finish_status': 'running'},
                         {'finish_status': 'success'}])
    check('监控: 尾行 success → 已终局 (覆盖路径防叠步)',
          e._step_concluded() is True)
    # 2 episode_stats → CSV 字段存在
    e.df = pd.DataFrame([{'finish_status': 'success', 'goal_reached': True,
                          'distance_to_goal': 0.8, 'spl': 0.3,
                          'action_number': -1}])
    e.current_episode = {'object_category': 'cola'}
    e._ep_stats = {'episode_ndx': 2, 'initial_geodesic': 5.0,
                   'vlm_calls_start': 0, 'stop_vetoed': 0, 'votes_cleared': 0,
                   'fastpath_triggered': 1, 'fastpath_final_close': 1}
    e.agent = NS(stopping_calls=[-2, 7])
    e.episode_stats_list = []
    e._collect_episode_stats()
    row = e.episode_stats_list[-1]
    check('监控: fastpath_triggered/final_close 进 episode_stats (CSV 源)',
          row.get('fastpath_triggered') == 1
          and row.get('fastpath_final_close') == 1)
    # 3 feature_flags: 三开关默认开 + 质量门 config 可配
    ff = feature_flags.resolve({})
    check('开关: identity_gate_v2/vote_fastpath/neutral_rooms 默认全开',
          ff['identity_gate_v2'] and ff['vote_fastpath']
          and ff['neutral_rooms'])
    check('开关: 质量门默认 OR 语义参数 (conf 0.30 / area 0.01)',
          ff['vote_fastpath_min_conf'] == 0.30
          and ff['vote_fastpath_min_area'] == 0.01)
    ff2 = feature_flags.resolve(
        {'feature_flags': {'vote_fastpath': False,
                           'vote_fastpath_min_conf': 0.5}})
    check('开关: cfg 可独立关 fastpath 且调质量门',
          ff2['vote_fastpath'] is False and ff2['vote_fastpath_min_conf'] == 0.5)
    # 4 CLI: --no-vote-fastpath 等剥离
    flags, rest = feature_flags.parse_argv(
        ['17', 'cola', '1', '--no-vote-fastpath', '--no-neutral-rooms'])
    check('开关: CLI --no-vote-fastpath/--no-neutral-rooms 生效',
          flags.get('vote_fastpath') is False
          and flags.get('neutral_rooms') is False and rest[:1] == ['17'])
    # 5 wrapper 硬规则 ESCAPE 标记齐全 (静态检查 ≥6 处)
    src = open('/home/tao_h/avdb_habitat_converter/scripts/'
               'avdb_sim_wrapper.py').read()
    check('FSM: wrapper 硬规则 ESCAPE 标记 ≥6 处 (1a/1b/1c/1d/2/3)',
          src.count('[FSM] state=ESCAPE') >= 6)
    # 6 agent 快通道条件静态存在: step_ndx≥3 地板 + last_stop_was_fastpath 标志
    asrc = open('/home/tao_h/VLMnav/src/agent.py').read()
    check('FSM: agent 快通道含 step_ndx≥3 地板 + 标志位复位',
          'self.step_ndx >= 3' in asrc
          and 'last_stop_was_fastpath = False' in asrc
          and 'last_stop_was_fastpath = True' in asrc)


if __name__ == '__main__':
    test_fsm_derivation()
    test_identity_gate_v2()
    test_vote_protect_layer1()
    test_vote_fastpath_layer2()
    test_neutral_rooms()
    test_monitoring_and_flags()
    print(f"\n{'='*50}\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print('FAILED:', *FAIL, sep='\n  - ')
        sys.exit(1)
