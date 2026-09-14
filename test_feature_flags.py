"""特性开关系统单测: CLI 解析 / cfg 合成 / 档位记名

运行: python -u test_feature_flags.py
"""
import sys
sys.path.insert(0, '/home/tao_h/VLMnav/src')
sys.path.insert(0, '/home/tao_h/avdb_habitat_converter/scripts')

from feature_flags import DEFAULTS, parse_argv, resolve, describe

PASS, FAIL = [], []


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f'  [{info}]' if info and not cond else ''))


def test_parse_argv():
    flags, rest = parse_argv(['17', 'cat_name', '1', '--no-metric-gate'])
    check('CLI: 开关剥离且位置参数保留',
          flags == {'metric_gate': False} and rest == ['17', 'cat_name', '1'])
    flags2, rest2 = parse_argv(['17', 'cat', '1',
                                '--no-cascade', '--no-warmup-preempt',
                                '--gt-detect'])
    check('CLI: 多开关组合',
          flags2 == {'cascade': False, 'warmup_preempt': False,
                     'gt_detect': True} and rest2 == ['17', 'cat', '1'])
    flags3, rest3 = parse_argv(['17', 'cat', '1'])
    check('CLI: 无开关 → 空	flags + 原参', flags3 == {} and rest3 == ['17', 'cat', '1'])


def test_resolve():
    ff = resolve()
    check('合成: 无参 → 全默认', ff == DEFAULTS)
    ff2 = resolve({'feature_flags': {'metric_gate': False}})
    check('合成: cfg 注入覆盖单一开关, 其余保持默认',
          ff2['metric_gate'] is False and ff2['cascade'] is True
          and ff2['warmup_preempt'] is True)
    ff3 = resolve({'feature_flags': {'bogus_flag': 1}})
    check('合成: 未知键被丢弃 (防拼写错误静默生效)',
          'bogus_flag' not in ff3)
    ff4 = resolve({'feature_flags': {'cascade': False}}, {'cascade': True})
    check('合成: overrides 优先于 cfg', ff4['cascade'] is True)


def test_describe():
    d = describe(resolve())
    check('记名: 全默认 → +C+G+S+W', d == '+C+G+S+W', d)
    d2 = describe(resolve({'feature_flags': {
        'cascade': False, 'grid_hub': False,
        'metric_gate': False, 'warmup_preempt': False}}))
    check('记名: 全关 → base', d2 == 'base', d2)
    d3 = describe(resolve({'feature_flags': {
        'grid_hub': False, 'metric_gate': False, 'warmup_preempt': False}}))
    check('记名: 只开级联 → +C', d3 == '+C', d3)
    d4 = describe(resolve({'feature_flags': {
        'metric_gate': False, 'warmup_preempt': False}}))
    check('记名: C+G → +C+G', d4 == '+C+G', d4)
    d5 = describe(resolve({'feature_flags': {'warmup_preempt': False}}))
    check('记名: C+G+S → +C+G+S', d5 == '+C+G+S', d5)
    d6 = describe(resolve({'feature_flags': {'hard_depth_filter': True}}))
    check('记名: 硬剔除对照 → +C+G+S+W+HD', d6 == '+C+G+S+W+HD', d6)
    d6b = describe(resolve({'feature_flags': {'gt_detect': True}}))
    check('记名: GT 上限行 → +C+G+S+W+GT', d6b == '+C+G+S+W+GT', d6b)
    d6c = describe(resolve({'feature_flags': {
        'cascade': False, 'grid_hub': False, 'metric_gate': False,
        'warmup_preempt': False, 'single_thresh': True}}))
    check('记名: base+ST (表 IV 单阈行)', d6c == 'base+ST', d6c)
    d7 = describe(resolve({'feature_flags': {
        'cascade': True, 'grid_hub': False}}))
    check('记名: 非标准组合 → custom', d7 == 'custom', d7)


if __name__ == '__main__':
    test_parse_argv()
    test_resolve()
    test_describe()
    print(f"\n{'='*40}\n✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        print('Failed:', *FAIL, sep='\n  - ')
        sys.exit(1)
