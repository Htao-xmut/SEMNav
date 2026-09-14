"""阶段0.1 单测 — 问题驱动三修复 (ep1 方位错认 / ep4 零检出转圈)

修2A hr_180_cooldown : 硬规则 1a/1d 的 180° 掉头冷却 (wrapper)
修2B zero_detect_crop: 零检出感知兜底 (env, 静态断言 + 计数语义)
修3A first_dir_gate   : FIRST_DIRECTION 单图证据门 (env)

实弹背景:
  ep1 (coca_cola): 六图联合分析把 120° 照片的可乐瓶归属到 240° (该照片
    全是百叶窗) → 开局背对目标 1.94m → 11.32m max_steps
  ep4 (softsoap): 全程零 YOLO 检出 → 无线索纯扫视 → 1b/1d 窗口惯性
    交替对抗 (掉头 5 次) → 区域内转圈 10.28m max_steps

运行: python -u test_ep_fixes.py
"""
import sys, re, logging
sys.path.insert(0, '/home/tao_h/VLMnav/src')
sys.path.insert(0, '/home/tao_h/avdb_habitat_converter/scripts')
import numpy as np

logging.basicConfig(level=logging.INFO)

from avdb_env import AVDBEnv
from avdb_sim_wrapper import AVDBSimWrapper
import feature_flags

PASS, FAIL = [], []


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f'  [{info}]' if info and not cond else ''))


ENV_SRC = open('/home/tao_h/VLMnav/src/avdb_env.py').read()
WRAP_SRC = open('/home/tao_h/avdb_habitat_converter/scripts/'
                'avdb_sim_wrapper.py').read()

FD_RE = re.compile(r'FIRST_DIRECTION:\s*(\d+)°')   # 下游两处解析正则


def make_env(ff=None):
    e = object.__new__(AVDBEnv)
    e._ff = {'first_dir_gate': True, 'zero_detect_crop': True, **(ff or {})}
    return e


def scan_frame(angle, img=None, yolo_found=False):
    return {'direction': f'{angle}°', 'angle': angle,
            'yolo_found': yolo_found, 'yolo_info': None,
            'img': img if img is not None else np.zeros((4, 4, 3), np.uint8)}


# ========== 修3A: FIRST_DIRECTION 单图证据门 (6 项) ==========
def test_first_dir_gate():
    # ep1 实弹措辞 (含目标词干 — 声明类型判据要求目标词紧邻 visible)
    resp = ("Analyzing... the Coca-Cola glass bottle is visible in the "
            "240° image. FIRST_DIRECTION: 240°")

    # 1 单图验证 present=0 → 方位作废 + response 改写 + 下游正则解析不到
    e = make_env()
    e._vlm_grounding_detect = lambda rgb, t: (False, {'target_found': False})
    a, r = e._verify_first_direction(resp, 240,
                                     [scan_frame(240)], 'coca_cola_glass_bottle')
    check('修3A: present=0 → 方位作废 (None)', a is None)
    check('修3A: response 改写含 REJECTED 标记', 'REJECTED-240deg' in r)
    check('修3A: 改写后下游正则解析不到 (双处 _init/_re_scan 同正则)',
          FD_RE.search(r) is None)

    # 2 present=1 → 保留方位, response 原样
    e = make_env()
    e._vlm_grounding_detect = lambda rgb, t: (True, {'confidence': 0.8})
    a, r = e._verify_first_direction(resp, 240,
                                     [scan_frame(240)], 'coca_cola_glass_bottle')
    check('修3A: present=1 → 方位保留 + response 原样',
          a == 240 and 'REJECTED' not in r)

    # 3 开关关 → 不调 VLM 直接采信 (旧行为)
    e = make_env(ff={'first_dir_gate': False})
    calls = []
    e._vlm_grounding_detect = \
        lambda rgb, t: (calls.append(1), (False, {}))[1]
    a, r = e._verify_first_direction(resp, 240,
                                     [scan_frame(240)], 't')
    check('修3A: 开关关 → 零 VLM 调用 + 直接采信 (消融)',
          a == 240 and calls == [])

    # 4 无对应帧 (角度对不上) → 保守放行, 不调 VLM
    e = make_env()
    calls = []
    e._vlm_grounding_detect = \
        lambda rgb, t: (calls.append(1), (False, {}))[1]
    a, _ = e._verify_first_direction(resp, 90, [scan_frame(240)], 't')
    check('修3A: 无对应帧 → 保守放行 (无法验证不惩罚)', a == 90 and calls == [])

    # 5 YOLO 已在该帧检出 → target_priority 路径, 不经门
    e = make_env()
    calls = []
    e._vlm_grounding_detect = \
        lambda rgb, t: (calls.append(1), (False, {}))[1]
    a, _ = e._verify_first_direction(resp, 240,
                                     [scan_frame(240, yolo_found=True)], 't')
    check('修3A: YOLO 已检出帧 → 不调 VLM (无串扰路径)', a == 240 and calls == [])

    # 6 angle=None → 直通
    a, r = e._verify_first_direction(resp, None, [scan_frame(240)], 't')
    check('修3A: angle=None → 直通', a is None and r == resp)


# ========== 修2A: 硬规则掉头冷却 (5 项) ==========
def test_hr_180_cooldown():
    w = object.__new__(AVDBSimWrapper)
    w.memory = {'step_count': 100}
    check('修2A: 首次掉头 → 允许 (并记账)', w._can_force_180() is True)
    w.memory['step_count'] = 111      # 11 步后 < 12
    check('修2A: 冷却内 (11步) → 拒绝', w._can_force_180() is False)
    w.memory['step_count'] = 112      # 12 步后 ≥ 冷却
    check('修2A: 冷却期满 (12步) → 允许', w._can_force_180() is True)
    # 开关关: 旧行为 — 连续掉头无限制
    w2 = object.__new__(AVDBSimWrapper)
    w2.memory = {'step_count': 0}
    w2.hr_180_cooldown = False
    ok = all(w2._can_force_180() for _ in range(3))
    check('修2A: 开关关 → 连续掉头全放行 (旧行为保留)', ok)
    # 挂接点静态断言: 1a/1d 触发条件已挂冷却 + 冷却日志
    check('修2A: 1a/1d 触发点已挂 _can_force_180 (静态)',
          'same_dir_streak >= 5 and self._can_force_180()' in WRAP_SRC
          and 'rotate_count >= 6 and self._can_force_180()' in WRAP_SRC
          and 'let VLM scan finish' in WRAP_SRC)


# ========== 修2B: 零检出感知兜底 (4 项) ==========
def test_zero_detect_crop():
    # 1 触发块存在 + 挂开关 + force stop 路径 (静态)
    check('修2B: 触发块挂 zero_detect_crop 开关 (静态)',
          "(_ff or {}).get('zero_detect_crop', True)" in ENV_SRC
          and 'zero-detect' in ENV_SRC
          and '[CROP-GATE] zero-detect CLOSE → force stop' in ENV_SRC)
    # 2 计数器更新语义: 未见 +1 / 见到清零 (源码行)
    check('修2B: 计数器更新 (未见+1 / 见到清零) (静态)',
          "= 0 if yolo_now.get('target_found')" in ENV_SRC
          and "getattr(self, '_no_detect_streak', 0) + 1" in ENV_SRC)
    # 3 每 episode 复位 + 节流间隔字段
    check('修2B: _initialize_episode 复位 (静态)',
          'self._no_detect_streak = 0' in ENV_SRC
          and 'self._last_zerodetect_crop = -99' in ENV_SRC)
    # 4 触发门槛: ≥12 步 + 间隔 ≥6 + 共用 _crop_calls ≤10 限流
    seg = ENV_SRC.split('修2B (ep4 实弹)')[1][:1200]
    check('修2B: 门槛 12 步 / 间隔 6 / 限流共用 (静态)',
          '>= 12' in seg and '>= 6' in seg
          and "_crop_calls', 0) < 10" in seg)


# ========== 开关系统 (2 项) ==========
def test_flags():
    ff = feature_flags.resolve({})
    check('开关: hr_180_cooldown/zero_detect_crop/first_dir_gate 默认全开',
          ff['hr_180_cooldown'] and ff['zero_detect_crop']
          and ff['first_dir_gate'])
    check('开关: 冷却步数默认 12 + 三个 CLI 剥离参数',
          ff['hr_180_cooldown_steps'] == 12
          and '--no-hr-180-cooldown' in feature_flags.FLAG_ARGS
          and '--no-zero-detect-crop' in feature_flags.FLAG_ARGS
          and '--no-first-dir-gate' in feature_flags.FLAG_ARGS)


# ========== 集成: 门在 _warmup_scan 调用点 (1 项) ==========
def test_gate_wired():
    check('修3A: 门已接入 _warmup_scan (静态)',
          'self._verify_first_direction(' in ENV_SRC
          and "'img': img," in ENV_SRC)


# ========== 修3A-b: 方位提取 fallback + 声明类型三态 (ep6 复测实弹) ==========
def test_first_dir_fallback():
    e = make_env()
    check('fallback: 规定行优先 (不走 action 字段)',
          e._extract_first_direction(
              'x FIRST_DIRECTION: 180° y ... "action": 60') == 180)
    # ep6 复测实弹: 无规定行, 角度锁死在 "action": 60 (格式漂移)
    check('fallback: 规定行缺失 → action 字段 60 (ep6 实弹)',
          e._extract_first_direction(
              '{"reasoning": "... kitchen ... 60° image", "action": 60}') == 60)
    check('fallback: 非法方位 45 / 字符串动作 → None',
          e._extract_first_direction('"action": 45') is None
          and e._extract_first_direction('"action": "rotate_cw"') is None)
    check('fallback: 两处都无 → None',
          e._extract_first_direction('no angle here') is None)
    # 过门放行后的补写: fallback 路径 response 无规定行 → 注入前补一行
    # (否则下游 _initialize_episode/_re_scan 正则解析不到, 链路仍断)
    check('fallback: 过门放行后补写规定行 (静态)',
          'if first_dir_angle is not None and not re.search(' in ENV_SRC
          and "response += f'\\nFIRST_DIRECTION: {first_dir_angle}°'" in ENV_SRC)


def test_claim_type_triage():
    # ep6 实弹措辞 (推测声明): kitchen 推测方位与真方位差 56° — 注入猜测
    # = 放大错误 → 不注入; 且无需 VLM 验证 (零成本路径)
    resp_ep6 = ('The mahatma rice is likely located in the kitchen area '
                'visible in the 60° image.\nFIRST_DIRECTION: 60°')
    e = make_env()
    calls = []
    e._vlm_grounding_detect = lambda rgb, t: (calls.append(1), (False, {}))[1]
    a, r = e._verify_first_direction(resp_ep6, 60,
                                     [scan_frame(60)], 'mahatma_rice')
    check('三态: ep6 推测声明 (likely located) → 不注入 + 零 VLM 调用',
          a is None and calls == [])
    check('三态: 推测声明含规定行 → 同步改写防下游注入',
          'REJECTED-60deg' in r and FD_RE.search(r) is None)
    # ep6 变体: 无规定行 (纯 action 字段漂移) → 原 response 无行, 天然安全
    a2, r2 = e._verify_first_direction(
        '{"reasoning": "mahatma rice likely in kitchen", "action": 60}',
        60, [scan_frame(60)], 'mahatma_rice')
    check('三态: 推测声明无规定行 → 不注入 + response 原样 (天然安全)',
          a2 is None and FD_RE.search(r2) is None and calls == [])
    # ep1 实弹措辞 (可见性声明): 可证伪 → 单图验证
    resp_ep1 = ('The Coca-Cola glass bottle is visible in the 240° image. '
                'FIRST_DIRECTION: 240°')
    e2 = make_env()
    e2._vlm_grounding_detect = lambda rgb, t: (False, {'target_found': False})
    a3, r3 = e2._verify_first_direction(resp_ep1, 240,
                                        [scan_frame(240)],
                                        'coca_cola_glass_bottle')
    check('三态: ep1 可见性声明 → 走单图验证, present=0 拒',
          a3 is None and 'REJECTED-240deg' in r3)


# ========== 问题1-A: 支撑面深度交叉 (4 项) ==========
def test_support_depth():
    from types import SimpleNamespace as NS

    def mk_env():
        e = object.__new__(AVDBEnv)
        e._ff = {'support_depth': True, 'parallax_gate': True}
        e._bbox_area_hist = []
        e.simWrapper = NS(target_name='cola', _node_pos=lambda n: None)
        return e

    # 1 _xcheck_depth 纯逻辑: 一致→网格值 / 分歧→保守 max / sup=None→原值
    e = mk_env()
    check('1A: 两源一致 (Δ≤0.6) → 用网格中位数',
          e._xcheck_depth(1.5, 1.9) == 1.5)
    check('1A: 强分歧 → 保守取远 max (遮挡与穿透都不产生新 fp)',
          e._xcheck_depth(0.8, 3.0) == 3.0
          and e._xcheck_depth(3.0, 0.8) == 3.0)
    check('1A: 支撑面无效 (None) → 原行为', e._xcheck_depth(1.5, None) == 1.5)

    # 2 _support_depth 像素逻辑: bbox 底边下方窗口取到台面深度 (3.0)
    d = np.full((100, 100), 0.8, np.float32)
    d[44:, :] = 3.0                      # 底边下方 = 远台面 (玻璃瓶穿透场景)
    sup = e._support_depth(d, 0.45, 0.50, 0.45, 100, 100)
    check('1A: 采样窗落在 bbox 底边下方 (取到 3.0m 台面)',
          sup is not None and abs(sup - 3.0) < 1e-6)


# ========== 问题1-B: 运动视差测距 (6 项) ==========
def test_parallax():
    from types import SimpleNamespace as NS
    POS = {'A': np.array([0.0, 0.0]), 'B': np.array([1.0, 0.0]),
           'C': np.array([0.1, 0.0])}

    def mk_env(hist):
        e = object.__new__(AVDBEnv)
        e._ff = {'support_depth': True, 'parallax_gate': True}
        e._bbox_area_hist = hist
        e.simWrapper = NS(target_name='cola', _node_pos=lambda n: POS[n])
        return e

    # 1 针孔模型数学: Δ=1m, 面积比 4 (d1/d2=2) → d2=1.0m
    e = mk_env([(10, 0.01, 'A'), (12, 0.04, 'B')])
    pd = e._parallax_distance()
    check('1B: 针孔模型 Δ1m×面积比4 → d2≈1.0m',
          pd is not None and abs(pd - 1.0) < 1e-6, f'got {pd}')
    # 2 四个无效情形: 不逼近/同节点/比率暴增/位移过小
    check('1B: 面积未增 (不逼近) → None',
          mk_env([(10, 0.04, 'A'), (12, 0.02, 'B')])._parallax_distance() is None)
    check('1B: 同节点 (无位移) → None',
          mk_env([(10, 0.01, 'A'), (12, 0.04, 'A')])._parallax_distance() is None)
    check('1B: 比率>25 (检测跳变) → None',
          mk_env([(10, 0.002, 'A'), (12, 0.06, 'B')])._parallax_distance() is None)
    check('1B: 位移<0.25m (噪声主导) → None',
          mk_env([(10, 0.01, 'A'), (12, 0.04, 'C')])._parallax_distance() is None)


# ========== 问题1 A+B: _stop_evidence 集成 (4 项) ==========
def _mk_stop_env(depth_img, hist, ff=None):
    from types import SimpleNamespace as NS
    POS = {'A': np.array([0.0, 0.0]), 'B': np.array([1.0, 0.0])}
    e = object.__new__(AVDBEnv)
    e._ff = {'support_depth': True, 'parallax_gate': True, **(ff or {})}
    e._bbox_area_hist = hist
    e.simWrapper = NS(target_name='cola', _node_pos=lambda n: POS[n])
    det = {'class_name': 'cola', 'bbox_norm': [0.45, 0.40, 0.50, 0.45]}
    e._obs = {'yolo_detection': {'target_found': True,
                                 'all_detections': [det]},
              'depth_sensor': depth_img}
    return e


def test_stop_evidence_ab():
    # 1 支撑面分歧否决: 框内网格 0.8m (会 close) + 底边下方台面 3.0m
    #   → 保守取远 3.0 → far (玻璃瓶穿透修复)
    d = np.full((100, 100), 0.8, np.float32)
    d[44:, :] = 3.0
    ev = _mk_stop_env(d, [])._stop_evidence(
        _mk_stop_env(d, [])._obs)
    check('A+B: 网格 0.8 + 台面 3.0 → 保守 far (穿透否决)',
          ev is not None and ev[0] == 'far')

    # 2 视差否决: 深度全 0.8 (close 候选) + 视差测距 2.5m → 降级 far
    d2 = np.full((100, 100), 0.8, np.float32)
    e2 = _mk_stop_env(d2, [(10, 0.02, 'A'), (12, 0.0392, 'B')])
    ev2 = e2._stop_evidence(e2._obs)
    check('A+B: 深度 0.8 close 候选 + 视差 2.5m → 降级 far',
          ev2 is not None and ev2[0] == 'far')

    # 3 视差一致 (0.9m ≤1.2) → close 保留
    e3 = _mk_stop_env(d2, [(10, 0.02, 'A'), (12, 0.0891, 'B')])
    ev3 = e3._stop_evidence(e3._obs)
    check('A+B: 视差 0.9m 与深度一致 → close 保留',
          ev3 is not None and ev3[0] == 'close')

    # 4 开关关 → 视差否决不生效 (旧行为)
    e4 = _mk_stop_env(d2, [(10, 0.02, 'A'), (12, 0.0392, 'B')],
                      ff={'parallax_gate': False})
    ev4 = e4._stop_evidence(e4._obs)
    check('A+B: 开关关 → 无视差否决 (旧行为 close)',
          ev4 is not None and ev4[0] == 'close')


# ========== 修4 (扫描覆盖 bug): warmup/rescan 帧目录隔离 (2 项) ==========
def test_scan_dir_isolation():
    # 用户实测: 每次rescan 重写 warmup_scan_XX 同名目录 → 初始扫描帧
    # 被覆盖丢失, 事后取证只能看到最后一次 rescan 的画面
    check('修4: 扫描帧目录带 step_tag (初始 warmup 固定名, rescan 独立名)',
          "def _warmup_scan(self, step_tag='warmup_scan')" in ENV_SRC
          and "step_dir = f'{ep_dir}/{step_tag}_{fi:02d}'" in ENV_SRC
          and 'warmup_scan_{fi:02d}' not in ENV_SRC)
    check('修4: rescan 传独立 tag + 删除 warmup_scan 中转搬移',
          "tag = f'step{step_num}_rescan'" in ENV_SRC
          and '_warmup_scan(step_tag=tag)' in ENV_SRC
          and 'shutil.move(scan_src, rescan_dst)' not in ENV_SRC)


# ========== 阶段1: 区域看尽→前沿跳跃 (转圈根治) ==========
def test_frontier_hop():
    from types import SimpleNamespace as NS

    # 小导航图: A→B→C→D 直线链 (0,0)→(0,3), 全图已知
    nodes = {n: {'world_pos': [0.0, 0.0, float(i)], 'direction': [0, 0, 1]}
             for i, n in enumerate('ABCD')}
    graph = {}
    for i, u in enumerate('ABCD'):
        graph[u] = {}
        if i > 0:
            v = 'ABCD'[i - 1]
            graph[u][v] = {'edge_type': 'forward', 'distance': 1.0}
        if i < 3:
            v = 'ABCD'[i + 1]
            graph[u][v] = {'edge_type': 'forward', 'distance': 1.0}

    class _EM:  # exploration_map 假地图 (只喂前沿格)
        def __init__(self, frontier):
            self.regions = np.zeros((600, 600), np.int8)
            for gx, gz in frontier:
                self.regions[gz, gx] = 2
        def _to_world(self, gz, gx):   # 与 ExplorationMap2D 反推一致
            return (gx - 300) * 0.05, (gz - 300) * 0.05

    def mk_env(em, cur='A'):
        e = object.__new__(AVDBEnv)
        e._ff = {'frontier_hop': True}
        e.step = 20
        # mock 契约保真: 生产 _node_pos 返回 2 维 (x, z) — 曾用 3 维
        # world_pos 直接当 _node_pos, ext[2]/cur_pos[2] 越界 crash 在
        # 测试里全绿、实弹 ep4 step19 才炸 (mock 失真 = 假绿灯)
        e.simWrapper = NS(
            current_node=cur,
            nav_graph={'nodes': nodes, 'graph': graph},
            visited_nodes=set(nodes), exploration_map=em,
            _node_pos=lambda n: np.array(nodes[n]['world_pos'])[[0, 2]],
            _area_key=lambda n: (0, 0))
        e._forced_return_path = []
        return e

    # 1 前沿格在 D 外 0.5m (3.5m 远) → 跳跃规划成功 + 路径写入 GOTO 通道
    e = mk_env(_EM([(300, 370)]))     # 世界坐标 (0, 3.5)
    ok = e._plan_frontier_hop()
    check('前沿: 3.5m 前沿格 → 规划成功', ok and e._forced_return_path)
    if ok:
        # 2 前沿格距 D 仅 2.0m (< min_dist 2.5) → 太近不跳
        check('前沿: 前沿格 <2.5m (太近) → 不跳',
              mk_env(_EM([(300, 320)]), cur='D')._plan_frontier_hop() is False)
    # 3 无前沿格 → False
    check('前沿: 地图无前沿 → 不跳',
          mk_env(_EM([]))._plan_frontier_hop() is False)
    # 4 _area_exhausted: 10 步困小区域 → True / 直线散布 → False / 步数不足 → False
    #    (POS 用 2 维 (x,z), 与生产 _node_pos 契约一致 — 见 mk_env 注释)
    #    P6 判尽门满足前提: 本 zone (0,0) 已有 2 个扫描机位 (门本身在
    #    test_scan_inquiry 单测)
    POS = {'P1': [0, 0], 'P2': [0, 0.8], 'P3': [0.6, 0.4],
           **{f'L{i}': [0, float(i)] for i in range(10)}}
    e4 = mk_env(_EM([]))
    e4.simWrapper._node_pos = lambda n: np.array(POS[n])
    e4._scan_positions_by_cell = {(0, 0): {'S1', 'S2'}}
    e4._all_nodes = ['P1', 'P2', 'P3', 'P1', 'P2', 'P3', 'P1', 'P2',
                     'P3', 'P1']
    check('前沿: 10 步困 <2.5m 小区域 (转圈) → 看尽',
          e4._area_exhausted() is True)
    e4._all_nodes = [f'L{i}' for i in range(10)]   # 直线 9m 长条
    check('前沿: 直线搜索 (对角线 9m) → 不触发', e4._area_exhausted() is False)
    e4._all_nodes = ['P1', 'P2', 'P3', 'P1']
    check('前沿: 步数不足 (4<10) → 不触发', e4._area_exhausted() is False)
    # 5 挂接静态断言: 触发块 + 轨迹记录 (含 override 步)
    check('前沿: 触发块挂 frontier_hop 开关 + _all_nodes 全路径记录 (静态)',
          "_plan_frontier_hop()" in ENV_SRC
          and "get('frontier_hop', True)" in ENV_SRC
          and 'self._all_nodes = (getattr' in ENV_SRC
          and "'--no-frontier-hop'" in open(
              '/home/tao_h/VLMnav/src/feature_flags.py').read())


# ========== 阶段1: 低置信误检换机位 (ep1 灯下黑) ==========
def test_reposition():
    from types import SimpleNamespace as NS
    e = object.__new__(AVDBEnv)
    e._ff = {'low_conf_reposition': True}
    e._reposition_calls = 0
    # 1 选边: 平移 (left/right) 优先于 forward; 纯旋转不算换机位
    opts = [{'chain_type': 'rotate_cw'}, {'chain_type': 'left'},
            {'chain_type': 'forward'}]
    obs = {'edge_options': opts}
    check('换机位: 平移优先 (left 先于 forward)',
          e._pick_reposition_option(obs) == 1)
    obs2 = {'edge_options': [
        {'chain_type': 'rotate_cw'},
        {'chain_type': 'forward'},
        {'chain_type': 'rotate_ccw'}]}
    check('换机位: 无平移 → forward; 全旋转 → None',
          e._pick_reposition_option(obs2) == 1
          and e._pick_reposition_option(
              {'edge_options': [{'chain_type': 'rotate_cw'}]}) is None)
    # 2 预算: ≤4 次 (静态断言) + 两处 IDENTITY no 分支挂接
    check('换机位: 预算 ≤4/回合 + 双 no 分支挂接 (静态)',
          ">= 4" in ENV_SRC.split('def _reposition_after_reject')[1][:600]
          and ENV_SRC.count("_reposition_after_reject(obs, conf") == 2)
    # 3 开关默认开
    ff = feature_flags.resolve({})
    check('换机位: 开关默认开', ff['low_conf_reposition'])


# ========== 修5 (记忆表示红线): 文件名不进 prompt ==========
def test_memory_representation():
    # 用户红线: 照片文件名/节点 ID = 数据集先验, 真实机器人没有 "照片库
    # 编号"; VISITED AREAS/SIGHTING/黑名单全部改 zone 标签 + bearing/dist
    check('修5: VISITED AREAS 行用 zone+bearing 表示 (无文件名)',
          'f"  - [{rec[\'visits\']}x] {self._guess_room(name)} "' in WRAP_SRC
          and 'f"bearing {ang:+.0f}°, ~{dist:.1f}m from you"' in WRAP_SRC
          and 'f"  - {name} ({room})' not in WRAP_SRC)
    check('修5: 目击/黑名单行同口径 (自身里程计相对方位)',
          'at {_rel_loc(sight_node)}' in WRAP_SRC
          and 'at {sight_node}' not in WRAP_SRC
          and 'f"  - {a[\'node\']} {loc}' not in WRAP_SRC)


# ========== 批次1: P1 停票距离交叉核 (7 项) ==========
def test_dist_xcheck():
    from types import SimpleNamespace as NS
    # mock 契约保真: _node_pos 返回 2 维 (x, z) — 同 test_frontier_hop
    POS = {'A': np.array([0.0, 0.0]), 'B': np.array([1.0, 0.0])}

    def mk_env(ff=None, hist=None):
        e = object.__new__(AVDBEnv)
        e._ff = {'dist_xcheck': True, 'support_depth': True,
                 'parallax_gate': True, **(ff or {})}
        # mock 契约保真: 生产 _bbox_area_hist 尾 2 条 (step, area, node)
        e._bbox_area_hist = hist if hist is not None else []
        e.simWrapper = NS(target_name='cola', _node_pos=lambda n: POS[n])
        return e

    def obs_with(depth=None, bbox=None):
        obs = {}
        if depth is not None:
            obs['depth_sensor'] = depth
        if bbox is not None:
            obs['yolo_detection'] = {'target_found': True, 'all_detections': [
                {'class_name': 'cola', 'bbox_norm': bbox,
                 'confidence': 0.9}]}
        return obs

    # 1 视差优先否决: Δ1m 面积比 1.44 (√r=1.2) → d2=5.0m > 1.5 → veto
    e = mk_env(hist=[(10, 0.01, 'A'), (12, 0.0144, 'B')])
    v, src = e._distance_xcheck(
        obs_with(depth=np.full((100, 100), 0.8, np.float32)), 'crop-close')
    check('P1: 视差 5.0m → veto (优先级最高)',
          v == 'veto' and src[0] == 'parallax'
          and abs(src[1] - 5.0) < 1e-6, f'got {v},{src}')
    # 2 支撑面否决 (视差 None): 框内/前向 0.8 近 + 底边下方 3.0 台面
    d = np.full((100, 100), 0.8, np.float32)
    d[44:, :] = 3.0
    e2 = mk_env()
    v2, src2 = e2._distance_xcheck(
        obs_with(depth=d, bbox=[0.45, 0.40, 0.50, 0.45]), 'crop-close')
    check('P1: 支撑面 3.0m (视差 None) → veto',
          v2 == 'veto' and src2[0] == 'support'
          and abs(src2[1] - 3.0) < 1e-6, f'got {v2},{src2}')
    # 3 前向带否决 (无框 crop close 场景): 全图 2.8m 远
    e3 = mk_env()
    v3, src3 = e3._distance_xcheck(
        obs_with(depth=np.full((100, 100), 2.8, np.float32)), 'crop-close')
    check('P1: 无框 + 前向带 2.8m → veto (ep6 step35 形态)',
          v3 == 'veto' and src3[0] == 'forward-band', f'got {v3},{src3}')
    # 4 全源无证据 → 维持 VLM 判定 (防死锁)
    e4 = mk_env()
    v4, _ = e4._distance_xcheck({}, 'crop-close')
    check('P1: 无深度无视差 (G1/G4 近距形态) → 维持 (none)',
          v4 == 'none')
    # 5 近距场景全源 ≤1.5 → 不否决
    e5 = mk_env()
    v5, _ = e5._distance_xcheck(
        obs_with(depth=np.full((100, 100), 0.8, np.float32),
                 bbox=[0.45, 0.40, 0.50, 0.45]), 'crop-close')
    check('P1: 前向 0.8m + 支撑面 0.8m → 放行 (不停真 close)',
          v5 == 'none')
    # 6 开关关 → 永不否决 (消融)
    e6 = mk_env(ff={'dist_xcheck': False})
    v6, _ = e6._distance_xcheck(
        obs_with(depth=np.full((100, 100), 2.8, np.float32)), 'crop-close')
    check('P1: 开关关 → 无否决 (旧行为)', v6 == 'none')
    # 7 挂接静态断言: 三个 crop 停票点 + _stop_evidence 第三源
    check('P1: 3 处 crop 停票点挂 _distance_xcheck + '
          '_stop_evidence 前向带降级 (静态)',
          ENV_SRC.count('self._distance_xcheck(') == 3
          and '→ demote CLOSE to FAR (P1 xcheck)' in ENV_SRC
          and '[DIST-XCHECK]' in ENV_SRC)
    # 8 方位守卫: 侧向近目标 (左半 0.9m + 右半 5m 远墙, 框在左) —
    #   前向带测视轴 (右侧远) 不在适用域 → 不否决 (旧契约保留)
    d8 = np.full((100, 100), 5.0, np.float32)
    d8[:, :50] = 0.9
    e8 = mk_env()
    v8, src8 = e8._distance_xcheck(
        obs_with(depth=d8, bbox=[0.10, 0.45, 0.20, 0.55]), 'crop-close')
    check('P1: 侧向近目标 (视轴远墙) → 前向带跳过, 不否决',
          v8 == 'none', f'got {v8},{src8}')
    # 9 居中框 + 全场 2.2m → veto (ep6 step37 形态; 支撑面/前向带按
    #   优先级任一触发 — 两源此处同距, 拒绝哪个都拦住误停)
    e9 = mk_env()
    v9, src9 = e9._distance_xcheck(
        obs_with(depth=np.full((100, 100), 2.2, np.float32),
                 bbox=[0.45, 0.40, 0.55, 0.50]), 'crop-close')
    check('P1: 居中框 + 远场 2.2m → veto (ep6 形态, 源按优先级)',
          v9 == 'veto' and src9[0] in ('support', 'forward-band'),
          f'got {v9},{src9}')


# ========== 批次1: P2 MUST-DO 条件化 (2 项) ==========
def test_mustdo_gate():
    # 1 结构: INCONCLUSIVE 分支 (门拒 + YOLO 无发现) 与 MUST-DO else 分支并存
    check('P2: 门拒时 MUST-DO 换推测注记 (静态)',
          'first-direction INCONCLUSIVE' in ENV_SRC
          and 'not target_priority and first_dir_angle is None' in ENV_SRC
          and 'do NOT treat it as a MUST' in ENV_SRC
          and 'MUST-DO: Go toward the warmup-identified direction NOW'
          in ENV_SRC)
    # 2 开关默认开 + 消融参数
    ff = feature_flags.resolve({})
    check('P2: mustdo_gate 默认开 + CLI 剥离参数',
          ff['mustdo_gate']
          and '--no-mustdo-gate' in feature_flags.FLAG_ARGS)


# ========== 批次1: P3 硬规则 1b/1c 伺服旁路 (4 项) ==========
def test_hr_servo_bypass():
    def mk_w(active, miss, flag=True):
        w = object.__new__(AVDBSimWrapper)
        w.hr_servo_bypass = flag
        w._approach_active = active
        w._approach_miss = miss
        return w

    # 1 精确伺服中 (APPROACH + miss=0) → 旁路生效
    check('P3: APPROACH miss=0 → 伺服中 (旁路 1b/1c)',
          mk_w(True, 0)._in_active_servo() is True)
    # 2 跟丢计数中 (miss>0) → 恢复硬规则 (防圈不失守)
    check('P3: APPROACH miss=2 (跟丢中) → 不旁路',
          mk_w(True, 2)._in_active_servo() is False)
    check('P3: 非 APPROACH → 不旁路 (旧行为)',
          mk_w(False, 0)._in_active_servo() is False)
    # 3 开关关 → 永不旁路 (消融)
    check('P3: 开关关 → 永不旁路 (旧行为)',
          mk_w(True, 0, flag=False)._in_active_servo() is False)
    # 4 挂接静态: 1b/1c 双挂 + env 侧 miss 同步 + flag 注入
    check('P3: 1b/1c 挂 _in_active_servo + env 同步 _approach_miss (静态)',
          WRAP_SRC.count('not hijacking (P3)') == 2
          and 'def _in_active_servo' in WRAP_SRC
          and 'self.simWrapper._approach_miss = getattr(' in ENV_SRC
          and 'hr_servo_bypass' in ENV_SRC
          and '--no-hr-servo-bypass' in open(
              '/home/tao_h/VLMnav/src/feature_flags.py').read())


# ========== 批次1: P5 确认问询 bbox 裁剪放大 (4 项) ==========
def test_zoom_confirm():
    from types import SimpleNamespace as NS

    def mk_env():
        e = object.__new__(AVDBEnv)
        e._ff = {'zoom_confirm': True}
        e.simWrapper = NS(target_name='cola')
        return e

    obs = {'color_sensor': np.zeros((100, 100, 3), np.uint8)}

    def yolo_with(bbox):
        return {'target_found': True, 'confidence': 0.9,
                'all_detections': [
                    {'class_name': 'cola', 'bbox_norm': bbox,
                     'confidence': 0.9}]}

    # 1 小框 (4px) 裁剪放大: #43 收紧 margin 2.5→1.0 → crop 44..56 (12px)
    #   ×4 → 48px, 框坐标同步换算 (旧 24px 窗: 0.3% 画面框只占 1/6 看不清)
    z = mk_env()._zoom_bbox_frame(obs, yolo_with([0.48, 0.48, 0.52, 0.52]))
    check('P5: 小框 4px → 裁剪 12px×4 放大 48px (#43 窗收紧) + 框坐标换算',
          z is not None and z[0].shape[:2] == (48, 48)
          and abs(z[1][0] - 16) < 2 and abs(z[1][2] - 32) < 2,
          f'got {None if z is None else (z[0].shape, z[1])}')
    # 2 无框 → None (走全幅旧路径)
    check('P5: 无框 → None (全幅旧路径)',
          mk_env()._zoom_bbox_frame(
              obs, {'all_detections': []}) is None)
    # 3 大框 (80px, margin 200px > 画面) → 整幅 ×4
    z3 = mk_env()._zoom_bbox_frame(obs, yolo_with([0.1, 0.1, 0.9, 0.9]))
    check('P5: 大框 → 近整幅裁剪 (上下文保留)',
          z3 is not None and z3[0].shape[:2] == (400, 400))
    # 4 挂接静态: 两个确认问询挂 zoom_confirm + zoom_note 提示
    seg_id = ENV_SRC.split('def _vlm_identity_check')[1][:3000]
    seg_px = ENV_SRC.split('def _vlm_proximity_check')[1][:3000]
    check('P5: IDENTITY/PROXIMITY 双挂 zoom (静态)',
          "zoom_confirm" in seg_id and "zoom_confirm" in seg_px
          and 'ZOOMED-IN crop' in seg_id and 'ZOOMED-IN crop' in seg_px
          and '--no-zoom-confirm' in open(
              '/home/tao_h/VLMnav/src/feature_flags.py').read())


# ========== 批次2: P4 释锁即六向重扫 (4 项) ==========
def test_release_rescan():
    from types import SimpleNamespace as NS

    # 1 释锁路径统一置 pending (blacklist 型 — 无目击/无 strike 依赖)
    e = object.__new__(AVDBEnv)
    e._ff = {'release_rescan': True}
    e.step = 10
    e.simWrapper = NS(current_node='A')   # 无 target_sighting_nodes → []
    e._invalidate_lock('unit: wrong object', blacklist=True)
    check('P4: _invalidate_lock → 置 pending 重扫标志',
          getattr(e, '_pending_release_rescan', False) is True)
    # 2 消费点: flag 门 + 预算 ≤2 + 与周期重扫互斥 (静态)
    check('P4: 消费点挂 flag + 预算 <2 + 周期互斥 (静态)',
          "_ff or {}).get('release_rescan', True)" in ENV_SRC
          and "'_release_rescans', 0) < 2" in ENV_SRC
          and 'lock release re-orientation' in ENV_SRC
          and '_periodic_scanned_at.add(self.step)' in ENV_SRC)
    # 3 每 episode 复位 (静态)
    check('P4: _initialize_episode 复位预算与 pending (静态)',
          'self._release_rescans = 0' in ENV_SRC
          and 'self._pending_release_rescan = False' in ENV_SRC)
    # 4 开关默认开 + CLI 剥离参数
    ff = feature_flags.resolve({})
    check('P4: release_rescan 默认开 + --no-release-rescan',
          ff['release_rescan']
          and '--no-release-rescan' in feature_flags.FLAG_ARGS)


# ========== 批次2: P6 扫描语义问询层 (10 项) ==========
def test_scan_inquiry():
    from types import SimpleNamespace as NS

    def mk_env(ff=None, node='A', cell=(1, 2), spots=None, positions=None):
        e = object.__new__(AVDBEnv)
        e._ff = {'scan_inquiry': True, **(ff or {})}
        e.step = 7
        e.simWrapper = NS(current_node=node, _area_key=lambda n: cell)
        if spots is not None:
            e._suspicious_spots = spots
        if positions is not None:
            e._scan_positions_by_cell = positions
        return e

    # 1 SCAN_SUSPICIOUS: 角度解析 + 机位登记 + 可疑点入队 + 注记文案
    e = mk_env()
    ang, note = e._parse_scan_inquiry(
        'rooms...\nSCAN_SUSPICIOUS: 120°\nFIRST_DIRECTION: REJECTED-60deg',
        'cola')
    check('P6: SCAN_SUSPICIOUS 120° → 角度 + "先查再走" 注记',
          ang == 120 and 'suspicious spot' in note and '120°' in note)
    check('P6: 本次扫描节点记为该 zone 机位',
          e._scan_positions_by_cell == {(1, 2): {'A'}})
    check('P6: 可疑点入队 (step, angle)',
          e._suspicious_spots == {(1, 2): [(7, 120)]})
    # 2 SCAN_CLEAR: 可疑点回写清空 + 去新区注记
    e2 = mk_env(spots={(1, 2): [(1, 10), (2, 20)]},
                positions={(1, 2): {'A'}})
    ang2, note2 = e2._parse_scan_inquiry(
        'nothing suspicious here\nSCAN_CLEAR', 'cola')
    check('P6: SCAN_CLEAR → 可疑点清空 (查过无藏匿) + 去新区注记',
          ang2 is None and e2._suspicious_spots == {(1, 2): []}
          and 'NEW zone' in note2)
    # 3 ≤2/区域: 已有 2 条 → 最旧先出, 新的入队
    e3 = mk_env(spots={(1, 2): [(1, 10), (2, 20)]})
    _ang3, _n3 = e3._parse_scan_inquiry('SCAN_SUSPICIOUS: 300°', 'cola')
    check('P6: 可疑点上限 2 (旧的先出)',
          e3._suspicious_spots[(1, 2)] == [(2, 20), (7, 300)])
    # 4 开关关 → 不解析不登记 (消融, 旧行为)
    e4 = mk_env(ff={'scan_inquiry': False})
    ang4, note4 = e4._parse_scan_inquiry('SCAN_SUSPICIOUS: 120°', 'cola')
    check('P4x: P6 开关关 → 零登记零注记 (旧行为)',
          ang4 is None and note4 == ''
          and not hasattr(e4, '_scan_positions_by_cell'))
    # 5 覆盖摘要: 机位数 + 未查可疑方位进 prompt (红线口径: zone 标签)
    e5 = mk_env(spots={(1, 2): [(3, 240)]}, positions={(1, 2): {'A', 'B'}})
    cov = e5._scan_coverage_note()
    check('P6: 覆盖摘要 — zone 标签 + 机位数 + 未查方位 (无节点 ID)',
          'zone 1_2' in cov and '2 distinct' in cov
          and '240°' in cov and 'A' not in cov.split('zone')[1])
    # 6 判尽门: 区域看尽须 ≥2 扫描机位 (灯下黑防护)
    def mk_traj_env(positions, flag=True):
        e = object.__new__(AVDBEnv)
        e._ff = {'frontier_hop': True, 'scan_inquiry': flag}
        POS = {'P1': [0, 0], 'P2': [0, 0.8], 'P3': [0.6, 0.4]}
        e.simWrapper = NS(_node_pos=lambda n: np.array(POS[n]),
                          _area_key=lambda n: (0, 0))
        e._all_nodes = ['P1', 'P2', 'P3', 'P1', 'P2', 'P3', 'P1', 'P2',
                        'P3', 'P1']
        e._scan_positions_by_cell = positions
        return e

    check('P6: 判尽门 — 2 机位 → 允许看尽 (可跳走)',
          mk_traj_env({(0, 0): {'S1', 'S2'}})._area_exhausted() is True)
    check('P6: 判尽门 — 1 机位 → 不判尽 (先换机位补扫)',
          mk_traj_env({(0, 0): {'S1'}})._area_exhausted() is False)
    check('P6: 判尽门 — 0 机位 → 不判尽 (从没 360° 扫过)',
          mk_traj_env({})._area_exhausted() is False)
    check('P6: 判尽门 — 开关关 → 无门 (旧行为直判看尽)',
          mk_traj_env({}, flag=False)._area_exhausted() is True)
    # 7 挂接静态: prompt 问询行 + _re_scan 兜底转身 + episode 复位
    check('P6: prompt 覆盖检查 + _re_scan 兜底 + 复位 (静态)',
          'SCAN_SUSPICIOUS: <angle>°' in ENV_SRC
          and 'SCAN_CLEAR' in ENV_SRC
          and 'P6 inquiry: suspicious spot at' in ENV_SRC
          and '_scan_coverage_note' in ENV_SRC
          and 'self._suspicious_spots = {}' in ENV_SRC
          and 'self._scan_positions_by_cell = {}' in ENV_SRC
          and '--no-scan-inquiry' in open(
              '/home/tao_h/VLMnav/src/feature_flags.py').read())


# ========== #25: 拒停不失忆 (目击方位锁 / P4 兜底, 7 项) ==========
def test_reject_sight_lock():
    from types import SimpleNamespace as NS
    # mock 契约保真: _node_pos 返回 2 维 (x, z); 目击记录
    # [(step, node_name, description)] 末条最新 (wrapper 89 行)
    POS = {'A': np.array([0.0, 0.0]), 'S': np.array([1.0, 1.0])}

    def mk_env(sights, cur='A', ff=None):
        e = object.__new__(AVDBEnv)
        e._ff = {'reject_sight_lock': True, **(ff or {})}
        e.step = 30
        e.simWrapper = NS(current_node=cur,
                          target_sighting_nodes=sights,
                          _node_pos=lambda n: POS[n])
        return e

    # 1 新鲜目击 (异节点, 5 步前) → 方位锁对准目击节点 (世界 45°)
    e = mk_env([(25, 'S', 'rice bag on counter')])
    e._reject_stop_and_approach('[T] unit', vote_level=True)
    check('#25: 新鲜目击异节点 → 目击方位锁 45° + miss=0 (不盲走)',
          getattr(e, '_approach_bearing', None) is not None
          and abs(e._approach_bearing - float(np.arctan2(1.0, 1.0))) < 1e-6
          and e._approach_miss == 0
          and not getattr(e, '_pending_release_rescan', False))
    # 2 站在目击点上 (方位无定义) → 无锁 + P4 释锁重扫兜底
    e2 = mk_env([(29, 'A', 'at my feet')])
    e2._reject_stop_and_approach('[T]', vote_level=True)
    check('#25: 站在目击点 → 无锁 + P4 重扫兜底 (原地重定向)',
          getattr(e2, '_approach_bearing', None) is None
          and e2._pending_release_rescan is True)
    # 3 无目击记录 → 同兜底
    e3 = mk_env([])
    e3._reject_stop_and_approach('[T]', vote_level=True)
    check('#25: 无目击记忆 → P4 重扫兜底',
          e3._pending_release_rescan is True)
    # 4 陈旧目击 (9 步前 > 8 步窗) → 视为无记忆 (不锁旧方位)
    e4 = mk_env([(21, 'S', 'stale')])
    e4._reject_stop_and_approach('[T]', vote_level=True)
    check('#25: 目击过期 (>8 步) → 不锁旧方位',
          getattr(e4, '_approach_bearing', None) is None
          and e4._pending_release_rescan is True)
    # 5 开关关 → 旧行为 (无锁无兜底, 消融)
    e5 = mk_env([(25, 'S', 'x')], ff={'reject_sight_lock': False})
    e5._reject_stop_and_approach('[T]', vote_level=True)
    check('#25: 开关关 → 旧行为 (无锁无重扫)',
          getattr(e5, '_approach_bearing', None) is None
          and not getattr(e5, '_pending_release_rescan', False))
    # 6 开关默认开 + CLI 剥离参数
    ff = feature_flags.resolve({})
    check('#25: reject_sight_lock 默认开 + --no-reject-sight-lock',
          ff['reject_sight_lock']
          and '--no-reject-sight-lock' in feature_flags.FLAG_ARGS)
    # 7 挂接静态: 拒停链 (force-stop 拒 → guard → 无框分支) 进 #25
    check('#25: 拒停链挂接 (静态)',
          'sight-memory lock' in ENV_SRC
          and "'[PROXIMITY] vote rejected" in ENV_SRC
          and 'no fresh sighting memory' in ENV_SRC)


# ========== #26: 判尽门跨格聚合 + 换机位补扫 + SCAN 解析兜底 (11 项) ==========
def test_viewpoint_rescan():
    from types import SimpleNamespace as NS

    def mk_traj_env(positions, cur_cell=(0, 0), ff=None):
        e = object.__new__(AVDBEnv)
        e._ff = {'frontier_hop': True, 'scan_inquiry': True,
                 'viewpoint_rescan': True, **(ff or {})}
        POS = {'P1': [0, 0], 'P2': [0, 0.8], 'P3': [0.6, 0.4]}
        e.simWrapper = NS(_node_pos=lambda n: np.array(POS[n]),
                          _area_key=lambda n: cur_cell)
        e._all_nodes = ['P1', 'P2', 'P3', 'P1', 'P2', 'P3', 'P1', 'P2',
                        'P3', 'P1']
        e._scan_positions_by_cell = positions
        return e

    # 1 #26a 聚合: 机位在邻格 (±1 格内) → 算数 (旧单格逻辑永 False)
    check('#26a: 机位在邻格 → 聚合计数判尽 (跨格修复)',
          mk_traj_env({(1, 1): {'S1'}, (2, 1): {'S2'}},
                      cur_cell=(2, 2))._area_exhausted() is True)
    # 2 #26a 远格 (>±1) 不算 + 挡下置换机位补扫 pending (#26b)
    e2 = mk_traj_env({(10, 10): {'S1', 'S2'}})
    check('#26a: 远格机位不算 → 不判尽',
          e2._area_exhausted() is False)
    # 3 #26b: 门挡下 → 置 _pending_viewpoint_shift (引导去补扫, 不干等)
    check('#26b: 门挡下 → 换机位补扫 pending 置位',
          getattr(e2, '_pending_viewpoint_shift', False) is True)
    # 4 #26b 预算满 (2/回合) → 不再置位 (日志走旧文案分支)
    e4 = mk_traj_env({})
    e4._viewpoint_rescans = 2
    e4._area_exhausted()
    check('#26b: 预算满 → 不置位 (≤2/回合封顶)',
          getattr(e4, '_pending_viewpoint_shift', False) is False)
    # 5 #26b 消费: 有平移边 → override 平移一步 + 下一步重扫 pending + 预算+1
    e5 = mk_traj_env({})
    calls = []
    e5._override_and_run = lambda obs, i, tag, note: \
        (calls.append((i, tag)) or 'OVERRIDE_ACT')
    e5._pending_viewpoint_shift = True
    obs5 = {'edge_options': [{'chain_type': 'rotate_cw'},
                             {'chain_type': 'left'}]}
    ret = e5._consume_viewpoint_shift(obs5)
    check('#26b: 平移边消费 → override(left=idx1) + rescan pending + 预算1',
          ret == 'OVERRIDE_ACT' and calls == [(1, 'VIEWPOINT-SHIFT')]
          and e5._pending_viewpoint_rescan is True
          and e5._viewpoint_rescans == 1)
    # 6 #26b 消费: 无平移边 → None + 不耗预算 (等周期重扫)
    e6 = mk_traj_env({})
    e6._override_and_run = lambda *a: 'SHOULD_NOT_RUN'
    e6._pending_viewpoint_shift = True
    ret6 = e6._consume_viewpoint_shift(
        {'edge_options': [{'chain_type': 'rotate_ccw'}]})
    check('#26b: 无平移边 → 不动预算不置 rescan (原地白扫防护)',
          ret6 is None and getattr(e6, '_viewpoint_rescans', 0) == 0
          and getattr(e6, '_pending_viewpoint_rescan', False) is False)
    # 7 #26b flag 关 → 不动作 (消融)
    e7 = mk_traj_env({}, ff={'viewpoint_rescan': False})
    e7._override_and_run = lambda *a: 'SHOULD_NOT_RUN'
    e7._pending_viewpoint_shift = True
    ret7 = e7._consume_viewpoint_shift({'edge_options': [
        {'chain_type': 'left'}]})
    check('#26b: 开关关 → 不平移不置位 (旧行为)',
          ret7 is None and getattr(e7, '_viewpoint_rescans', 0) == 0)

    def mk_scan_env(spots=None, ff=None):
        e = object.__new__(AVDBEnv)
        e._ff = {'scan_inquiry': True, **(ff or {})}
        e.step = 9
        e.simWrapper = NS(current_node='A', _area_key=lambda n: (3, 4))
        if spots is not None:
            e._suspicious_spots = spots
        return e

    # 8 #26c JSON 字段: scan_suspicious 90 → 入队 (规定行缺失兜底)
    e8 = mk_scan_env(spots={})
    ang8, _ = e8._parse_scan_inquiry(
        '{"reasoning": "shelf corner unchecked", "scan_suspicious": 90}',
        'cola')
    check('#26c: JSON scan_suspicious 90 → 解析入队',
          ang8 == 90 and e8._suspicious_spots == {(3, 4): [(9, 90)]})
    # 9 #26c JSON 字段: scan_clear true → 清空 + 去新区注记
    e9 = mk_scan_env(spots={(3, 4): [(2, 45)]})
    ang9, note9 = e9._parse_scan_inquiry(
        '{"reasoning": "all six photos checked", "scan_clear": true}',
        'cola')
    check('#26c: JSON scan_clear → 可疑点清空 + NEW zone 注记',
          ang9 is None and e9._suspicious_spots == {(3, 4): []}
          and 'NEW zone' in note9)
    # 10 #26c 关键词兜底: nothing suspicious → CLEAR; 无角度 suspicious 不误报
    e10 = mk_scan_env(spots={(3, 4): [(2, 45)]})
    _a10, note10 = e10._parse_scan_inquiry(
        'Checked all 6 photos — nothing suspicious anywhere here.', 'cola')
    e10b = mk_scan_env(spots={})
    a10b, _ = e10b._parse_scan_inquiry(
        'the shelf looks suspicious but no angle given', 'cola')
    check('#26c: 关键词 CLEAR 兜底 + SUSPICIOUS 无角度不误报 (宁漏勿误)',
          e10._suspicious_spots == {(3, 4): []} and 'NEW zone' in note10
          and a10b is None and e10b._suspicious_spots == {})
    # 11 复位 + 挂接静态: plain_text 根因修复 + 双 pending 消费 + 复位
    VLM_SRC = open('/home/tao_h/VLMnav/src/vlm.py').read()
    check('#26c: plain_text 参数 + warmup 传参 + 消费/复位挂接 (静态)',
          'plain_text: bool' in VLM_SRC
          and 'and not plain_text' in VLM_SRC
          and 'plain_text=True)' in ENV_SRC
          and 'self._viewpoint_rescans = 0' in ENV_SRC
          and '_consume_viewpoint_shift(obs)' in ENV_SRC
          and 'viewpoint coverage re-scan (#26' in ENV_SRC
          and '--no-viewpoint-rescan' in open(
              '/home/tao_h/VLMnav/src/feature_flags.py').read())


if __name__ == '__main__':
    test_first_dir_gate()
    test_first_dir_fallback()
    test_claim_type_triage()
    test_hr_180_cooldown()
    test_zero_detect_crop()
    test_flags()
    test_gate_wired()
    test_support_depth()
    test_parallax()
    test_stop_evidence_ab()
    test_scan_dir_isolation()
    test_frontier_hop()
    test_reposition()
    test_memory_representation()
    test_dist_xcheck()
    test_mustdo_gate()
    test_hr_servo_bypass()
    test_zoom_confirm()
    test_release_rescan()
    test_scan_inquiry()
    test_reject_sight_lock()
    test_viewpoint_rescan()
    print(f"\n{'='*50}\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print('FAILED:', *FAIL, sep='\n  - ')
        sys.exit(1)
