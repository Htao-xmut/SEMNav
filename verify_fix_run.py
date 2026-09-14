#!/usr/bin/env python3
"""修复验证: A1 (旋转真转相机) / A2 (绿格不降级) 逐帧核账

用法: python -u verify_fix_run.py <run_dir>
  <run_dir> 例: logs/ObjectNav_batch_+C+G+S+W_s2_n1/0_of_1/2_Home_001_1

输出:
  1. 红点位置对账 (节点世界坐标预测 vs PNG 实测, 应 0px)
  2. 绿像素单调性 (A2: 不应有净减 >50px)
  3. 旋转步 vs 平移步 的绿增量 (A1: 修复后旋转步应显著 >0)
  4. 前沿格方位扇区集中度 (A1: 修复后 top-4/12 桶占比应远离 73%)
"""
import sys, os, re, json
import numpy as np
from PIL import Image

GRAPH = '/home/tao_h/avdb_habitat_converter/data/processed/navigation_graph.json'
PAL = {'unknown': (30, 30, 30), 'green': (0, 180, 0),
       'gray': (150, 150, 150), 'blue': (60, 60, 160), 'red': (255, 60, 60)}


def node_of(details_path):
    try:
        t = open(details_path).read()
    except OSError:
        return None, None
    m = re.search(r'(\d+\.jpg) \([^)]*\) \[\d+x\][^\n]*← YOU ARE HERE', t)
    node = m.group(1) if m else None
    ma = re.search(r'ACTION_NUMBER\n(-?\d+)', t)
    # 动作类型: 从 AVAILABLE DIRECTIONS 里找 [N] 行
    at = None
    if ma:
        n = int(ma.group(1))
        mm = re.search(rf'^\s*\[{n}\] ([^\n]+)$', t, re.M)
        if mm:
            d = mm.group(1).lower()
            if 'turn' in d or 'sidestep' not in d and ('left' in d.split('then')[0] or 'right' in d.split('then')[0]):
                at = 'rotate' if 'then' not in d else 'composite'
            elif 'then' in d:
                at = 'composite'
            elif 'forward' in d or 'backward' in d or 'sidestep' in d:
                at = 'walk'
            else:
                at = '?'
    return node, at


def main(run_dir):
    G = json.load(open(GRAPH))
    steps = sorted((d for d in os.listdir(run_dir)
                    if re.fullmatch(r'step\d+', d)),
                   key=lambda d: int(d[4:]))
    assert steps, f'no step dirs in {run_dir}'
    rows = []
    prev = None
    for s in steps:
        p = f'{run_dir}/{s}'
        node, atype = node_of(f'{p}/details.txt')
        img = Image.open(f'{p}/topdown_depth_map.png').convert('RGB')
        a = np.asarray(img, dtype=np.int16)
        cnt = {k: int((np.abs(a - np.array(c)).max(axis=2) <= 12).sum())
               for k, c in PAL.items()}
        m = (np.abs(a - np.array(PAL['red'])).max(axis=2) <= 60)
        dot = (int(np.nonzero(m)[0].mean()), int(np.nonzero(m)[1].mean())) \
            if m.sum() >= 5 else None
        rows.append({'step': int(s[4:]), 'node': node, 'atype': atype,
                     'green': cnt['green'], 'gray': cnt['gray'],
                     'blue': cnt['blue'], 'dot': dot})
    # 1. 红点对账
    base = next(r for r in rows if r['node'])
    c0 = G['nodes'][base['node']]['world_pos']
    errs, skipped = [], []
    for r in rows:
        if r['node'] is None:
            skipped.append(r['step'])
            continue
        wp = G['nodes'][r['node']]['world_pos']
        pred = (int((wp[2] - c0[2]) / 0.05 + 300), int((wp[0] - c0[0]) / 0.05 + 300))
        if r['dot']:
            errs.append(((r['dot'][0] - pred[0]) ** 2 +
                         (r['dot'][1] - pred[1]) ** 2) ** 0.5)
    # 2. 绿单调性
    dec = [(rows[i - 1]['step'], rows[i - 1]['green'] - rows[i]['green'])
           for i in range(1, len(rows))
           if rows[i]['green'] < rows[i - 1]['green'] - 50]
    # 3. 旋转步 vs 平移步绿增量
    growth = {}
    for i in range(1, len(rows)):
        d = rows[i]['green'] - rows[i - 1]['green']
        at = rows[i]['atype'] or '?'
        growth.setdefault(at, []).append(max(0, d))
    # 4. 前沿扇区集中度 (末帧)
    last = f'{run_dir}/{steps[-1]}/topdown_depth_map.png'
    a = np.asarray(Image.open(last).convert('RGB'), dtype=np.int16)
    gray = (np.abs(a - np.array(PAL['gray'])).max(axis=2) <= 12)
    dots = np.array([[(G['nodes'][r['node']]['world_pos'][0] - c0[0]) / 0.05 + 300,
                      (G['nodes'][r['node']]['world_pos'][2] - c0[2]) / 0.05 + 300]
                     for r in rows if r['node']])
    rr, cc = np.nonzero(gray)
    D = np.hypot(rr[:, None] - dots[:, 1], cc[:, None] - dots[:, 0])
    near = D.argmin(1)
    keep = D.min(1) > 5
    bear = np.degrees(np.arctan2(cc - dots[near, 0], rr - dots[near, 1])) % 360
    bear = bear[keep]
    hist, _ = np.histogram(bear, bins=12, range=(0, 360))
    top4 = int(np.sort(hist)[-4:].sum() / max(1, hist.sum()) * 100)

    print(f'=== {run_dir} ({len(rows)} steps) ===')
    if skipped:
        print(f'    (跳过无节点信息的步: {skipped})')
    print(f"[1] 红点对账: {len(errs)}/{len(rows)} 步有红点, "
          f"误差 max {max(errs):.1f}px / mean {np.mean(errs):.1f}px"
          + ('  ✓' if max(errs) < 2 else '  ✗ 异常!'))
    print(f"[2] A2 绿单调: 净减>50px 的步数 = {len(dec)}"
          + (f'  ✗ {dec}' if dec else '  ✓ 单调不降'))
    for at in ('rotate', 'walk', 'composite'):
        if at in growth:
            g = growth[at]
            print(f"[3] A1 {at:9s} 步 ({len(g):2d} 步): 绿增量 median "
                  f"{np.median(g):6.0f}px  mean {np.mean(g):6.0f}px")
    print(f"[4] A1 前沿扇区: top-4/12 桶占 {top4}% (基线 ep5=73%, 均匀≈33%)")
    return max(errs) < 2, not dec, top4


if __name__ == '__main__':
    main(sys.argv[1])
