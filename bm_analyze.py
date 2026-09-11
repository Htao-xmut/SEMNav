#!/usr/bin/env python3
"""基准测试分析 — 每组: 轨迹/结局/停止链/缺陷信号 (A/B/C)

用法: python -u bm_analyze.py <name> ...   (name 如 bm1_bumblebee1)
"""
import re, os, sys, glob, json

def analyze(name):
    base = f'/home/tao_h/VLMnav/logs/ObjectNav_{name}/0_of_1'
    runs = sorted(glob.glob(f'{base}/*/'))
    if not runs:
        return {'name': name, 'error': 'no logs'}
    D = runs[-1].rstrip('/')
    log = f'/tmp/bm_{name}.log'
    text = open(log, errors='ignore').read() if os.path.exists(log) else ''
    # fallback: scan all per-run logs? assume caller saved each to /tmp/bm_<name>.log

    r = {'name': name}
    steps = sorted([d for d in glob.glob(f'{D}/step*')
                    if re.fullmatch(r'step\d+', os.path.basename(d))],
                   key=lambda p: int(re.findall(r'\d+', p)[-1]))
    dists = {}
    for d in steps:
        try:
            t = open(f'{d}/details.txt').read()
            m = re.search(r'DISTANCE_TO_GOAL\n([\d.]+)', t)
            if m:
                dists[int(os.path.basename(d)[4:])] = float(m.group(1))
        except Exception:
            pass
    last_step = max(steps, key=lambda p: int(re.findall(r'\d+', p)[-1])) if steps else None
    if last_step:
        try:
            t = open(f'{last_step}/details.txt').read()
            r['status'] = (re.search(r'FINISH_STATUS\n(\w+)', t) or [None, '?'])[1]
            r['goal'] = (re.search(r'GOAL_REACHED\n(\w+)', t) or [None, '?'])[1]
            stop_r = re.search(r'STOPPING RESPONSE.*?"reasoning":\s*"([^"]{0,180})', t, re.S)
            r['stop_reason'] = stop_r.group(1).strip() if stop_r else ''
        except Exception:
            pass
    if dists:
        best_step = min(dists, key=dists.get)
        r['d0'] = dists.get(0)
        r['best'] = f"{dists[best_step]:.2f}@s{best_step}"
        r['final'] = f"{dists[max(dists)]:.2f}@s{max(dists)}"
        r['n_steps'] = max(dists)
    # 事件计数 (来自运行日志)
    def cnt(pat):
        return len(re.findall(pat, text))
    r['ev'] = {
        'identity': cnt(r'\[IDENTITY\] conf=([\d.]+) → VLM verdict: (\w+)'),
        'id_no': cnt(r'verdict: no'), 'id_unsure': cnt(r'verdict: unsure'),
        'id_unarmed_reject': cnt(r'unarmed sighting.*rejected'),
        'area_flip': cnt(r'\[AREA\] rec'),
        'rule3': cnt(r'HARD RULE 3'),
        'steer': cnt(r'AUTO-STEER\] step'),
        'approach': cnt(r'\[APPROACH\] step'),
        'confirm': cnt(r"'CONFIRM'"),
        'arb_close': cnt(r'ARBITRATION\].*HONORED'),
        'arb_reject': cnt(r'ARBITRATION\].*rejected'),
        'blacklist_add': cnt(r'\[BLACKLIST\] \+'),
        'suppressed': re.findall(r'detection at blacklisted node (\S+) suppressed \(conf=([\d.]+)\)', text),
        'visual_confirm': re.findall(r'Visual confirm: (.+)', text),
        'consec_stop': re.findall(r'Stop by consecutive stops: (\d)', text),
    }
    # 缺陷信号
    sig = []
    if r.get('status') == 'fp':
        sig.append(f"A? 停止被接受但距离 {r.get('final')} (fp)")
    stops2 = [int(x) for x in r['ev']['consec_stop'] if int(x) >= 2]
    if stops2:
        sig.append(f"A: 连续{max(stops2)}票触发stop")
    if r['ev']['consec_stop'] and not r['ev']['arb_reject'] and not r['ev']['arb_close']:
        sig.append("B: stop未走仲裁 (无新鲜目击) 直通")
    hi_sup = [f"{n[-10:-4]}:{c}" for n, c in r['ev']['suppressed'] if float(c) >= 0.7]
    if hi_sup:
        sig.append(f"C: 高置信检测被黑名单抑制 {hi_sup}")
    r['defects'] = sig
    return r


if __name__ == '__main__':
    names = sys.argv[1:]
    out = [analyze(n) for n in names]
    print(json.dumps(out, ensure_ascii=False, indent=1))
