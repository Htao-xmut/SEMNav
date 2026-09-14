"""D2/③/④ 修复单测: 停票证据分级 / 拒停转接近 / crop 门 / 覆盖完成停

不启动模拟器, 不调用真实 VLM (脚本化响应)。
运行: python -u test_stop_gate.py
"""
import sys, logging
sys.path.insert(0, '/home/tao_h/VLMnav/src')
sys.path.insert(0, '/home/tao_h/avdb_habitat_converter/scripts')
import numpy as np

logging.basicConfig(level=logging.WARNING)

from avdb_env import AVDBEnv
from avdb_sim_wrapper import PolarAction

PASS, FAIL = [], []


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f'  [{info}]' if info and not cond else ''))


class FakeVLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def call_chat(self, i, imgs, prompt):
        self.calls.append((prompt, imgs))
        return self.reply


def make_env(target='aunt_jemima_original_syrup'):
    env = object.__new__(AVDBEnv)
    env.step = 7
    env.cfg = {'sensor_cfg': {'fov': 131}, 'success_threshold': 1.0}

    class SW:
        target_name = target
        current_node = 'N1'
        nav_graph = {'nodes': {'N1': {'world_pos': [0, 0, 0],
                                      'direction': [0, 0, 1]}}}

    env.simWrapper = SW()

    class AG:
        stop_history = [False, True, True, False, True]

    env.agent = AG()
    env._approach_bearing = None
    env._approach_miss = 99
    env._last_returned_sighting_step = -1
    env._forced_return_path = []
    env._arrival_check_step = -99
    env._sighting_frames = {}
    env._arrival_calls = 0
    return env


def depth_frame(val, H=540, W=960):
    return np.full((H, W), val, dtype=np.float32)


def det(bbox, cls='aunt_jemima_original_syrup', conf=0.8):
    return {'target_found': True, 'confidence': conf,
            'all_detections': [{'class_name': cls, 'bbox_norm': bbox,
                                'confidence': conf}]}


# ---------- ② _stop_evidence ----------
def test_stop_evidence():
    env = make_env()
    # 框面积 5% → close (无需深度)
    obs = {'yolo_detection': det([0.40, 0.40, 0.60, 0.60]),
           'depth_sensor': depth_frame(9.0)}
    check('证据: 框面积≥2% → close', env._stop_evidence(obs)[0] == 'close')
    # 小框 + bbox 中心深度 4m → far
    obs = {'yolo_detection': det([0.48, 0.48, 0.52, 0.52]),
           'depth_sensor': depth_frame(4.0)}
    check('证据: 小框深探 4m → far (G2 拦截点)', env._stop_evidence(obs)[0] == 'far')
    # 1.3 收紧分级: ≤1.0m 硬判据 close; (1.0, 2.2]m 软接近 far (拒停
    # 转接近锁, 原旧判据 1.5m 会 close — bm18 曾停 1.68m 即其后果)
    for d_, tag in ((0.95, 'close'), (1.01, 'far'), (1.5, 'far'),
                    (2.19, 'far'), (2.21, 'far')):
        obs = {'yolo_detection': det([0.48, 0.48, 0.52, 0.52]),
               'depth_sensor': depth_frame(d_)}
        check(f'证据: 1.3 分级 小框深探 {d_}m → {tag}',
              env._stop_evidence(obs)[0] == tag)
    # bm10/bm14 回归: 小框 + 极近 = 遮挡物击中探测 → 小框即远证据 → far
    # (bm14: 1.6% 框 + 0.06m 探深, 真距 5.43m; 旧版 defer crop 后放大
    #  裁剪抹掉尺寸线索被判 close → fp 停)
    obs = {'yolo_detection': det([0.32, 0.46, 0.44, 0.59]),
           'depth_sensor': depth_frame(0.47)}
    check('证据: 小框+极近 物理矛盾 → far 拒停转接近 (bm10 回归)',
          env._stop_evidence(obs) == ('far', obs['yolo_detection']))
    obs = {'yolo_detection': det([0.40, 0.35, 0.48, 0.55]),   # 0.08×0.20=1.6%
           'depth_sensor': depth_frame(0.06)}
    ev = env._stop_evidence(obs)
    check('证据: bm14 实弹参数 (1.6%框 + 0.06m探深) → far',
          isinstance(ev, tuple) and ev[0] == 'far'
          and ev[1] is obs['yolo_detection'],
          f'ev={ev[0] if isinstance(ev, tuple) else ev}')
    # 深度探测位置取 bbox 中心而非全图: 左侧近墙 (0.9m) 右侧远 (5m)
    d = depth_frame(5.0)
    d[:, :480] = 0.9                      # 左半近 (≤1.0 硬判据)
    obs = {'yolo_detection': det([0.10, 0.45, 0.20, 0.55]),   # 中心 x=0.15 → 左
           'depth_sensor': d}
    check('证据: 探测取 bbox 中心局部 (左近框 → close)',
          env._stop_evidence(obs)[0] == 'close')
    # 无检测 / 无框 / 深度无效 → None
    check('证据: 无检测 → None', env._stop_evidence({'yolo_detection': {}}) is None)
    obs = {'yolo_detection': {'target_found': True, 'confidence': 0.9,
                              'all_detections': [{'class_name':
                                                  'aunt_jemima_original_syrup'}]},
           'depth_sensor': depth_frame(0.0)}
    check('证据: 有检测无框 → None', env._stop_evidence(obs) is None)
    obs = {'yolo_detection': det([0.48, 0.48, 0.52, 0.52]),
           'depth_sensor': depth_frame(0.0)}
    check('证据: 深度全无效 → None', env._stop_evidence(obs) is None)
    check('证据: obs 空 → None', env._stop_evidence(None) is None)


# ---------- ② _reject_stop_and_approach ----------
def test_reject_and_approach():
    env = make_env()
    env._reject_stop_and_approach('test: reset only')
    h = env.agent.stop_history
    check('拒停: 尾部连票 True 清零 (计数条件失活)',
          h == [False, True, True, False, False], f'{h}')
    # 带框 → 建立接近锁
    env2 = make_env()
    env2._reject_stop_and_approach('test: aim',
                                   yolo_info=det([0.60, 0.40, 0.80, 0.60]))
    rel_expect = (0.70 - 0.5) * 131.0   # cx=0.70 → 右偏 26.2°
    expect = np.radians(rel_expect)      # yaw_world = 0 (direction [0,0,1])
    check('拒停: 框方位 → 接近锁 (下步 AUTO-STEER)',
          env2._approach_bearing is not None
          and abs(env2._approach_bearing - expect) < 0.01
          and env2._approach_miss == 0,
          f'bearing={env2._approach_bearing} expect={expect}')


# ---------- bm17 绕门修复: _proximity_rejected_stop_guard ----------
def test_proximity_guard():
    # ① 连票被拦转接近锁 (bm17 Step12 复刻: 小框 4.95m 远距误判 close)
    env = make_env()
    env._last_obs = {'yolo_detection': det([0.48, 0.48, 0.52, 0.52]),
                     'depth_sensor': depth_frame(4.95)}
    m = {'done': True, 'finish_status': 'fp', 'goal_reached': False}
    env._proximity_rejected_stop_guard(PolarAction.stop, m, 4.95)
    check('守卫①: 连票 stop 被拦 → done 翻回 False 继续走',
          m['done'] is False and m['finish_status'] == 'running')
    check('守卫①: 清连票 + 建接近锁 (走过去不停)',
          env.agent.stop_history[-1] is False
          and env._approach_bearing is not None
          and env._arbitration_close is False,
          f'h={env.agent.stop_history} bearing={env._approach_bearing}')

    # ② 连票过门 (米制 CLOSE → 尊重停止, 交给 fp-救援)
    env2 = make_env()
    env2._last_obs = {'yolo_detection': det([0.40, 0.40, 0.60, 0.60]),   # 4% ≥ 2%
                      'depth_sensor': depth_frame(4.95)}
    m2 = {'done': True, 'finish_status': 'fp', 'goal_reached': False}
    env2._proximity_rejected_stop_guard(PolarAction.stop, m2, 3.00)
    check('守卫②: 米制 CLOSE → 尊重停止 (done 不翻, 连票不清)',
          m2['done'] is True and env2._arbitration_close is True
          and env2.agent.stop_history[-1] is True)

    # ③ 无框连票清零 (无米制证据 → far, 不建锁)
    env3 = make_env()
    env3._last_obs = {'yolo_detection': {}}
    m3 = {'done': True, 'finish_status': 'fp'}
    env3._proximity_rejected_stop_guard(PolarAction.stop, m3, 4.95)
    check('守卫③: 无框 → 拒停 + 清票 + 不建锁',
          m3['done'] is False and env3.agent.stop_history[-1] is False
          and env3._approach_bearing is None)

    # ④ 投票级: 被拒的票就地清零 (动作不是 stop → metrics 不动)
    env4 = make_env()
    m4 = {'done': False, 'finish_status': 'running'}
    env4._proximity_rejected_stop_guard(None, m4, 5.16)
    check('守卫④: 投票级拒绝 → 只清票不动 metrics',
          m4 == {'done': False, 'finish_status': 'running'}
          and env4.agent.stop_history == [False, True, True, False, False])


# ---------- ③ _crop_confirm ----------
def test_crop_confirm():
    env = make_env()
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    obs = {'color_sensor': rgb}
    env.agent.actionVLM = FakeVLM('{"present": 1, "close": 1, "reasoning": "x"}')
    check('crop: present+close → close', env._crop_confirm(obs, 't') == 'close')
    env.agent.actionVLM = FakeVLM('{"present": 1, "close": 0, "reasoning": "x"}')
    check('crop: present+far → far', env._crop_confirm(obs, 't') == 'far')
    env.agent.actionVLM = FakeVLM('{"present": 0, "close": 0, "reasoning": "x"}')
    check('crop: absent → no', env._crop_confirm(obs, 't') == 'no')
    # 限流: 已 10 次 → None
    env._crop_calls = 10
    env.agent.actionVLM = FakeVLM('{"present": 1, "close": 1, "reasoning": "x"}')
    check('crop: ≥10 次限流 → None', env._crop_confirm(obs, 't') is None)
    # 调用格式: 放大裁剪 (非原图直传)
    env2 = make_env()
    env2._crop_calls = 0
    f = FakeVLM('{"present": 0, "close": 0, "reasoning": "x"}')
    env2.agent.actionVLM = f
    env2._crop_confirm({'color_sensor': rgb}, 't')
    img = f.calls[0][1][0]
    check('crop: 送 VLM 的是放大裁剪 (2x 下 2/3 区域)',
          img.shape[0] == int(120 * 0.70) * 2 and img.shape[1] == int(160 * 0.76) * 2,
          f'{img.shape}')


# ---------- ③c _arrival_confirm + 到场武装 ----------
def test_arrival_confirm():
    env = make_env()
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    env.agent.actionVLM = FakeVLM('{"close": 1, "where": "left frame, on table"}')
    check('到场: 左帧近景 → close',
          env._arrival_confirm({'color_sensor': rgb}, rgb) == 'close')
    env.agent.actionVLM = FakeVLM('{"close": 0, "where": "far side of room"}')
    check('到场: 左帧远景/不可辨 → no',
          env._arrival_confirm({'color_sensor': rgb}, rgb) == 'no')
    # 限流: 已 6 次 → None
    env._arrival_calls = 6
    env.agent.actionVLM = FakeVLM('{"close": 1, "where": "x"}')
    check('到场: ≥6 次限流 → None',
          env._arrival_confirm({'color_sensor': rgb}, rgb) is None)
    # 无当前帧 → None
    env2 = make_env()
    env2._arrival_calls = 0
    check('到场: 无 color_sensor → None', env2._arrival_confirm({}, rgb) is None)
    # 调用格式: 左右并排双帧 (各 448 宽 → 896×336)
    env3 = make_env()
    f = FakeVLM('{"close": 0, "where": "x"}')
    env3.agent.actionVLM = f
    env3._arrival_confirm({'color_sensor': rgb}, rgb)
    img = f.calls[0][1][0]
    check('到场: 送 VLM 的是并排双帧 896×336',
          img.shape[0] == 336 and img.shape[1] == 896, f'{img.shape}')


def test_arrival_arm():
    # 同位目击 (朝向变体节点) → skip 分支就地武装当步确认
    env = make_env()
    env._graph_path_to = lambda a, b: []
    sw = env.simWrapper
    sw.target_sighting_nodes = [(env.step - 1, 'N1_dir2', 'i see it')]
    sw._node_pos = lambda n: [0.0, 0.0, 0.0]
    env._start_forced_return()
    check('到场: 同位目击 skip → 当步武装 (_arrival_check_step=step)',
          env._arrival_check_step == env.step and env._forced_return_path == [],
          f'armed={env._arrival_check_step} step={env.step}')
    # 回访路径末边执行 → 武装下一步
    env2 = make_env()
    env2._forced_return_path = ['backward']
    env2._override_and_run = lambda obs, idx, tag, note: 0
    env2.outer_run_name = env2.inner_run_name = env2.curr_run_name = 'x'
    obs = {'edge_options': [{'chain_type': 'backward', 'chain_count': 1,
                             'composite': None}]}
    env2._execute_forced_return(obs)
    check('到场: 末边执行 → 下一步武装 (step+1)',
          env2._arrival_check_step == env2.step + 1
          and env2._forced_return_path == [],
          f'armed={env2._arrival_check_step} step={env2.step}')
    # 目击帧留存 hook 逻辑 (独立验证 dict 更新与容量)
    env3 = make_env()
    env3._last_obs = {'color_sensor': np.full((60, 80, 3), 7, np.uint8)}
    env3.simWrapper.target_sighting_nodes = [(9, 'NA', 'x')]
    env3.step = 10
    # 直接复刻 hook 主体
    sights_f = env3.simWrapper.target_sighting_nodes
    if sights_f and sights_f[-1][0] > getattr(env3, '_sight_cap_step', -1):
        env3._sight_cap_step = sights_f[-1][0]
        rgb_f = env3._last_obs.get('color_sensor')
        if rgb_f is not None:
            env3._sighting_frames[sights_f[-1][1]] = np.asarray(rgb_f).copy()
    check('到场: 目击帧按节点留存可取回',
          env3._sighting_frames.get('NA') is not None
          and env3._sighting_frames['NA'].shape == (60, 80, 3))


# ---------- bm15 修复: IDENTITY yes 建接近锁 ----------
def test_identity_yes_lock():
    env = make_env()
    env._identity_reject = {}
    env._last_identity_step = -99
    env.step = 7
    env._vlm_identity_check = lambda obs, y: 'yes'
    rgb = np.zeros((60, 80, 3), np.uint8)
    yolo = {'target_found': True, 'confidence': 0.39, 'area_ratio': 0.01,
            'all_detections': [{'class_name':
                                'aunt_jemima_original_syrup',
                                'bbox_norm': [0.60, 0.40, 0.80, 0.60]}]}
    # 复刻 1839 分支主体 (conf 0.3-0.6 未武装 → identity 检查)
    conf_u = yolo.get('confidence', 0)
    area_u = float(yolo.get('area_ratio', 0) or 0)
    assert 0.30 <= conf_u < 0.60 and area_u < 0.10
    verdict = env._vlm_identity_check({'color_sensor': rgb}, yolo)
    if verdict == 'yes':
        rel_yes = env._detection_bearing_deg(yolo)
        if rel_yes is not None:
            env._approach_bearing = env._yaw_world() + np.radians(rel_yes)
            env._approach_miss = 0
    expect = np.radians((0.70 - 0.5) * 131.0)   # cx=0.70 → +26.2°, yaw=0
    check('身份确认: IDENTITY yes → 接近锁 (bm15: conf0.39 真目标被丢)',
          env._approach_bearing is not None
          and abs(env._approach_bearing - expect) < 0.01
          and env._approach_miss == 0,
          f'bearing={env._approach_bearing} expect={expect}')


# ---------- bm11/bm16 修复: 无框停票判据 ----------
def test_no_box_verdict():
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    # 无框 = 无米制证据 → 一律 far, 且零 VLM 调用 (bm11/14/16 三连 fp:
    # VLM 看图判 close 在 4.4-5.4m 全错)
    env = make_env()
    f = FakeVLM('{"present": 1, "close": 1, "reasoning": "x"}')
    env.agent.actionVLM = f
    check('无框判: 无框停票 → far, 零 VLM 调用 (bm16 拦截点)',
          env._no_box_stop_verdict({'color_sensor': rgb}) == 'far'
          and len(f.calls) == 0, f'calls={len(f.calls)}')
    check('无框判: crop 已限流不影响 → far',
          env._no_box_stop_verdict({'color_sensor': rgb}) == 'far')


# ---------- ④ _coverage_exhausted ----------
def test_coverage_exhausted():
    class EM:
        pass
    env = make_env()

    em = EM()
    em.regions = np.ones((600, 600), dtype=np.int8)          # 全绿 已探索
    em.seen = np.ones((600, 600), dtype=bool)
    env.simWrapper.exploration_map = em
    check('覆盖停: 无前沿+覆盖足 → True', env._coverage_exhausted() is True)

    em2 = EM()
    r = np.ones((600, 600), dtype=np.int8); r[50:60, 50:60] = 2   # 有前沿
    em2.regions = r
    em2.seen = np.ones((600, 600), dtype=bool)
    env.simWrapper.exploration_map = em2
    check('覆盖停: 还有前沿 → False', env._coverage_exhausted() is False)

    em3 = EM()
    r3 = np.zeros((600, 600), dtype=np.int8); r3[290:310, 290:310] = 1  # 覆盖太小
    em3.regions = r3
    em3.seen = np.zeros((600, 600), dtype=bool)
    env.simWrapper.exploration_map = em3
    check('覆盖停: 地图未画开 → False', env._coverage_exhausted() is False)

    env.simWrapper.exploration_map = None
    check('覆盖停: 无地图 → False', env._coverage_exhausted() is False)


if __name__ == '__main__':
    test_stop_evidence()
    test_reject_and_approach()
    test_proximity_guard()
    test_crop_confirm()
    test_arrival_confirm()
    test_arrival_arm()
    test_identity_yes_lock()
    test_no_box_verdict()
    test_coverage_exhausted()
    print(f"\n{'='*40}\n✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        print('Failed:', *FAIL, sep='\n  - ')
        sys.exit(1)
