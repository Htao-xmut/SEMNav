"""任务三 warmup 反应式抢占单测: 强证据判定 + 消融开关 + 正则兼容

不启动模拟器, 不调用真实 VLM。
运行: python -u test_warmup_preempt.py
"""
import sys, logging, re
sys.path.insert(0, '/home/tao_h/VLMnav/src')
sys.path.insert(0, '/home/tao_h/avdb_habitat_converter/scripts')

logging.basicConfig(level=logging.WARNING)

from avdb_env import AVDBEnv

PASS, FAIL = [], []


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f'  [{info}]' if info and not cond else ''))


def make_env(preempt=True):
    env = object.__new__(AVDBEnv)
    env._ff = {'warmup_preempt': preempt}
    return env


def hit(conf, bbox, angle=120, target='aunt_jemima_original_syrup'):
    """构造 scan_data 里的一个 YOLO 命中条目"""
    return {
        'direction': f'{angle}°', 'angle': angle, 'yolo_found': True,
        'yolo_info': {'confidence': conf, 'target_found': True,
                      'all_detections': [
                          {'class_name': target, 'bbox_norm': bbox,
                           'confidence': conf}]},
    }


# ---------- 强证据判定 ----------
def test_preempt_check():
    # ① 强证据: conf 0.8 + 面积 0.10×0.06=0.6% ≥0.5% → 抢占
    env = make_env()
    ok, msg = env._warmup_preempt_check(
        hit(0.8, [0.40, 0.40, 0.50, 0.46]), 'aunt_jemima_original_syrup')
    check('抢占: conf≥0.5 且面积≥0.5% → 抢占 + 置标志',
          ok and env._warmup_preempted is True and 'PREEMPT' in msg)

    # ② caller 桥接正则兼容: (\d+°) 与 confidence **x%**
    a = re.search(r'\((\d+)°\)', msg)
    c = re.search(r'confidence \*\*([\d.]+)%', msg)
    check('抢占: 消息保持 [APPROACH-LOCK] warmup 桥接正则兼容',
          a is not None and a.group(1) == '120'
          and c is not None and float(c.group(1)) == 80.0,
          f'a={a.group(1) if a else None} c={c.group(1) if c else None}')

    # ③ conf 不足 (0.45) → 不抢占
    env2 = make_env()
    ok2, _ = env2._warmup_preempt_check(
        hit(0.45, [0.30, 0.30, 0.50, 0.50]), 'aunt_jemima_original_syrup')
    check('抢占: conf 0.45 <0.5 → 不抢占 (走 VLM 分析路径)',
          ok2 is False and not hasattr(env2, '_warmup_preempted'))

    # ④ 面积不足 (0.08×0.045≈0.36% <0.5%) → 不抢占
    env3 = make_env()
    ok3, _ = env3._warmup_preempt_check(
        hit(0.8, [0.40, 0.40, 0.48, 0.445]), 'aunt_jemima_original_syrup')
    check('抢占: conf 0.8 但面积 ~0.36% → 不抢占 (远距小目标仍需语义层)',
          ok3 is False)

    # ⑤ 消融开关: --no-warmup-preempt
    env4 = make_env(preempt=False)
    ok4, _ = env4._warmup_preempt_check(
        hit(0.9, [0.20, 0.20, 0.60, 0.60]), 'aunt_jemima_original_syrup')
    check('抢占: 消融开关关闭 → 强证据也不抢占 (W 档基线)',
          ok4 is False)

    # ⑥ 面积取目标类别最大框 (非目标检测不参与)
    env5 = make_env()
    mixed = hit(0.8, [0.40, 0.40, 0.50, 0.46])
    mixed['yolo_info']['all_detections'].append(
        {'class_name': 'other_thing', 'bbox_norm': [0.0, 0.0, 1.0, 1.0],
         'confidence': 0.95})
    ok6, _ = env5._warmup_preempt_check(mixed, 'aunt_jemima_original_syrup')
    check('抢占: 面积只统计目标类别框 (他类大框不误触发)', ok6 is True)

    # ⑦ 面积略高于阈值 (0.10×0.055≈0.55%) → 抢占 (远离浮点边界)
    env6 = make_env()
    ok7, _ = env6._warmup_preempt_check(
        hit(0.6, [0.40, 0.40, 0.50, 0.455]), 'aunt_jemima_original_syrup')
    check('抢占: 面积 ~0.55% 略超阈值 → 抢占', ok7 is True)


if __name__ == '__main__':
    test_preempt_check()
    print(f"\n{'='*40}\n✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        print('Failed:', *FAIL, sep='\n  - ')
        sys.exit(1)
