"""#42/#43 低置信 identity=no 二次确认 + zoom 窗收紧 单测 (2026-09-13)

用户实弹指控 (bmv7 step27, logs/ObjectNav_bmv7_mahatmav7/0_of_1/0_Home_001_1/):
  "YOLO都看到了，不过去确认下吗？！！！！ 另外，YOLO的置信框大小目前
   会不会设大了？其实不用那么大。大概画面的3%其实已经非常大了！"
  "其实应该去前进二次确认下啊！"
— conf 0.41 真米袋被 identity no → 记 loss strike → 与接近链丢失凑
  TWO-STRIKES 拉黑 → step28 扫描重检被 `[BLACKLIST] ... suppressed` →
  语义改道 → 4.67m max_steps。

修复:
  #42a conf<0.50 的 no ≈ unsure → CONFIRM 前进二次确认 (≤3/节点)
  #42b wrong 判决不混入丢失两击 (_invalidate_lock loss_strike=False)
  #43  _zoom_bbox_frame margin_mult 2.5→1.0 (窗口 36×→9× bbox 面积)

运行: python -u test_identity_confirm.py
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


OBS = {
    'yolo_detection': {
        'target_found': True, 'confidence': 0.41, 'area_ratio': 0.05,
        'position': 'middle_right',
        'all_detections': [{'class_name': 'mahatma_rice',
                            'bbox_norm': [0.6, 0.5, 0.68, 0.58],
                            'confidence': 0.41}],
    },
    'edge_options': [{'chain_type': 'forward', 'chain_count': 1,
                      'composite': None}],
}


def mk_env(conf=0.41, verdict='no', confirm_idx=0, flag=True,
           budget_used=0):
    from avdb_env import AVDBEnv
    e = object.__new__(AVDBEnv)
    e._ff = {'identity_lowconf_confirm': flag, 'low_conf_reposition': False}
    e.step = 27
    e.simWrapper = type('SW', (), {'current_node': 'N1'})()
    e._identity_reject = {}
    e._last_identity_step = -99
    e._lowconf_confirm_used = {'N1': budget_used}
    e._detection_bearing_deg = lambda yi: 10.0
    e._vlm_identity_check = lambda o, y: verdict
    e._pick_confirm_option = lambda o, y: confirm_idx
    e.invalidated = []

    def _inv(reason, **kw):
        e.invalidated.append((reason, kw))

    e._invalidate_lock = _inv
    e.ran = []

    def _ov(obs, idx, tag, note=''):
        e.ran.append((idx, tag, note))
        return ('ov', idx)

    e._override_and_run = _ov
    obs = dict(OBS)
    obs['yolo_detection'] = dict(OBS['yolo_detection'],
                                 confidence=conf,
                                 all_detections=[{'class_name': 'mahatma_rice',
                                                  'bbox_norm': [0.6, 0.5, 0.68, 0.58],
                                                  'confidence': conf}])
    return e, obs


def test_lowconf_confirm():
    from avdb_env import AVDBEnv

    # 1 conf 0.41 < 0.50 的 no → CONFIRM 前进二次确认, 不拒绝不解锁
    e, obs = mk_env()
    r = AVDBEnv._approach_step(e, obs)
    check('#42a: 低置信 no → CONFIRM 二次确认 (不走拒绝路径)',
          r is not None and e.ran and e.ran[0][1] == 'CONFIRM'
          and not e.invalidated and not e._identity_reject
          and e._lowconf_confirm_used['N1'] == 1)

    # 2 预算耗尽 (3/3) → 回落原拒绝路径
    e, obs = mk_env(budget_used=3)
    r = AVDBEnv._approach_step(e, obs)
    check('#42a: 确认预算 3/3 耗尽 → 拒绝 (保 ep8 类真误报止损)',
          r is None and e._identity_reject.get('N1') == 27
          and len(e.invalidated) == 1)

    # 3 高置信 (≥0.50) no → 维持原行为: 立即拒绝, 不二次确认
    e, obs = mk_env(conf=0.55)
    r = AVDBEnv._approach_step(e, obs)
    check('#42a: conf≥0.50 的 no → 立即拒绝 (高置信否认可信)',
          r is None and not e.ran
          and e._identity_reject.get('N1') == 27)

    # 4 #42b: 拒绝路径 _invalidate_lock 必须传 loss_strike=False
    #   (wrong 判决不与丢失 strike 混算 — bmv7 TWO-STRIKES 误升级主因)
    e, obs = mk_env(conf=0.55)
    AVDBEnv._approach_step(e, obs)
    check('#42b: identity-no 释锁传 loss_strike=False (不混丢失两击)',
          e.invalidated and e.invalidated[0][1].get('loss_strike') is False)

    # 5 无前进确认选项 → 回落拒绝 (视角受限时不死等)
    e, obs = mk_env(confirm_idx=None)
    r = AVDBEnv._approach_step(e, obs)
    check('#42a: 无 confirm 选项 → 回落拒绝路径',
          r is None and e._identity_reject.get('N1') == 27
          and len(e.invalidated) == 1)

    # 6 消融 --no-identity-lowconf-confirm → 低置信 no 直接拒绝 (原行为)
    e, obs = mk_env(flag=False)
    r = AVDBEnv._approach_step(e, obs)
    check('#42: 消融 --no-identity-lowconf-confirm → 原拒绝行为',
          r is None and not e.ran and e._identity_reject.get('N1') == 27)


def test_zoom_tighter():
    from avdb_env import AVDBEnv
    e = object.__new__(AVDBEnv)
    e.simWrapper = type('SW', (), {'target_name': 'mahatma_rice'})()
    # 480x640 帧, bbox 6.4% 宽 (≈2% 画面面积 — 用户: 3% 已经非常大)
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    obs = {'color_sensor': rgb}
    yi = {'all_detections': [{'class_name': 'mahatma_rice',
                              'bbox_norm': [0.50, 0.50, 0.564, 0.564],
                              'confidence': 0.41}]}
    z, zbox = e._zoom_bbox_frame(obs, yi)
    # 7 窗口面积 ≈ 9× bbox (margin_mult 1.0: 每边留 1× bbox 边长)
    #   放大后尺寸变了 → 用 bbox 在窗中的边长占比反推窗/bbox 面积比
    fx = (zbox[2] - zbox[0]) / z.shape[1]
    fy = (zbox[3] - zbox[1]) / z.shape[0]
    ratio = 1.0 / (fx * fy)
    check('#43: zoom 窗 ≈ 9× bbox 面积 (旧 2.5 倍边距时 ≈ 36×)',
          7.5 <= ratio <= 11.0, f'ratio={ratio:.1f}')
    # 8 目标在放大窗里占 ~1/3 边长 (~11% 面积) — VLM 真正看得清
    frac = (zbox[2] - zbox[0]) / z.shape[1]
    check('#43: bbox 占放大窗 ~1/3 边长 (目标可辨识, 不再是 1/6)',
          0.28 <= frac <= 0.38, f'frac={frac:.2f}')


def test_static():
    SRC = open('/home/tao_h/VLMnav/src/avdb_env.py').read()

    # 9 静态: 分支接线 + loss_strike=False + 默认 margin_mult=1.0 + flag 注册
    from feature_flags import DEFAULTS, FLAG_ARGS, parse_argv
    check('#42/#43: 源码接线 (二次确认分支/loss_strike/margin_mult/flag)',
          'identity_lowconf_confirm' in SRC
          and '_lowconf_confirm_used' in SRC
          and 'second confirmation' in SRC
          and 'loss_strike=False)' in SRC
          and 'margin_mult=1.0, min_out=448' in SRC)
    flags, rest = parse_argv(['--no-identity-lowconf-confirm', 'x'])
    check('#42: DEFAULTS True + --no-identity-lowconf-confirm 注册剥离',
          DEFAULTS.get('identity_lowconf_confirm') is True
          and FLAG_ARGS.get('--no-identity-lowconf-confirm')
          == ('identity_lowconf_confirm', False)
          and flags == {'identity_lowconf_confirm': False} and rest == ['x'])


if __name__ == '__main__':
    test_lowconf_confirm()
    test_zoom_tighter()
    test_static()
    print('=' * 40)
    print(f"✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        for n in FAIL:
            print(f'  FAIL: {n}')
        sys.exit(1)
