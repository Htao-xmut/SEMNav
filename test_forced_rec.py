"""forced exploration 跟随 DEPTH 推荐单测 (bm18 实弹修复)

bm18: 27 次 forced 0 次跟随 — agent 正则找 "Recommended: ±Ndeg" 但
obs['depth_trace'] 文本里没有该字样 (只在给 VLM 的 prompt 里)。
修: wrapper 注入结构化 obs['depth_rec_deg'], agent 优先读它。
运行: python -u test_forced_rec.py
"""
import sys, logging
sys.path.insert(0, '/home/tao_h/VLMnav/src')

logging.basicConfig(level=logging.WARNING)

PASS, FAIL = [], []


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f'  [{info}]' if info and not cond else ''))


def opt(et, cnt=1):
    return {'chain_type': et, 'chain_count': cnt, 'composite': None}


def pick_forced_by_rec(options, valid_nonzero, obs):
    """复刻 agent.py forced 块主体: 结构化优先, 正则兜底, 角差选优"""
    rec_deg = obs.get('depth_rec_deg')
    if rec_deg is None:
        import re as _re
        m = _re.search(r'Recommended: ([+-]\d+)deg', obs.get('depth_trace') or '')
        if m:
            rec_deg = float(m.group(1))
    if rec_deg is None:
        return None, None
    rec_deg = float(rec_deg)

    def _opt_bearing(o):
        if o.get('composite'):
            et, cnt = o['composite'][0]
        else:
            et, cnt = (o.get('chain_type'), o.get('chain_count', 1))
        base = {'rotate_cw': 30.0, 'rotate_ccw': -30.0, 'forward': 0.0,
                'backward': 180.0, 'left': -90.0, 'right': 90.0}.get(et, None)
        if base is None:
            return None
        return base * (cnt or 1)

    best, best_d = None, 1e9
    for i in valid_nonzero:
        b = _opt_bearing(options[i])
        if b is None:
            continue
        d = abs((b - rec_deg + 540) % 360 - 180)
        if d < best_d:
            best_d, best = d, i
    if best is not None and best_d <= 100:
        return best, rec_deg
    return None, rec_deg


def test_structured_rec():
    opts = [opt('forward'), opt('rotate_ccw'), opt('rotate_cw'), opt('right')]
    idx, deg = pick_forced_by_rec(opts, [0, 1, 2, 3],
                                  {'depth_rec_deg': -51.0})
    check('跟随: rec -51° → rotate_ccw(-30°) 选项 [1]',
          idx == 1, f'idx={idx}')
    idx, _ = pick_forced_by_rec(opts, [0, 1, 2, 3], {'depth_rec_deg': 88.0})
    check('跟随: rec +88° → right(+90°) 选项 [3]', idx == 3, f'idx={idx}')
    idx, _ = pick_forced_by_rec(opts, [0, 1, 2, 3], {'depth_rec_deg': 175.0})
    check('跟随: rec 175° 无接近选项 (min 差 95>…边界) 按 100° 阈值',
          idx == 3, f'idx={idx}')   # right 90 距 175 差 85 ≤ 100 → 仍选


def test_regex_fallback():
    opts = [opt('forward'), opt('left')]
    # 无结构化字段 → 正则兜底 (老格式兼容)
    idx, deg = pick_forced_by_rec(opts, [0, 1],
                                  {'depth_trace': '... Recommended: -85deg ...'})
    check('兜底: 无 depth_rec_deg 时正则解析 -85° → left(-90°) [1]',
          idx == 1 and deg == -85.0, f'idx={idx} deg={deg}')


def test_bm18_no_match_fixed():
    # bm18 实弹文本: depth_trace 里根本没有 Recommended 字样 →
    # 旧正则 0 命中; 结构化字段存在时必须走结构化
    trace = ('Status: ok\nDepth sectors: ...\n'
             'DEPTH SCAN recommends turning right (~29deg)')
    opts = [opt('forward'), opt('rotate_cw')]
    idx, deg = pick_forced_by_rec(opts, [0, 1], {'depth_trace': trace})
    check('bm18 文本: 无结构化+无正则命中 → 不改选 (None)',
          idx is None and deg is None, f'idx={idx}')
    idx, deg = pick_forced_by_rec(
        opts, [0, 1], {'depth_trace': trace, 'depth_rec_deg': 29.0})
    check('bm18 文本: 有结构化字段 → 跟随 +29° 选 rotate_cw [1]',
          idx == 1, f'idx={idx}')


if __name__ == '__main__':
    test_structured_rec()
    test_regex_fallback()
    test_bm18_no_match_fixed()
    print(f"\n{'='*40}\n✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        print('Failed:', *FAIL, sep='\n  - ')
        sys.exit(1)
