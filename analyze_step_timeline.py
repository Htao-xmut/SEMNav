"""逐步还原 coca run: 每步节点/距离/检测/动作 + 关键事件对齐"""
import os, re, glob, json
import numpy as np

R = 'logs/ObjectNav_avdb_depth_ep51_coca_50step/0_of_1/0_Home_001_1'
ng = json.load(open('/home/tao_h/avdb_habitat_converter/data/processed/navigation_graph.json'))
goal = np.array([3.0854, -4.7841])

# 主日志事件按时间轴
events = []
with open('/tmp/coca_blacklist_run.log') as f:
    for line in f:
        m = re.match(r'(\d+:\d+:\d+)', line)
        for tag in ['PROXIMITY', 'CORRECT', 'BLACKLIST', 'APPROACH', 'FORCED-RETURN',
                    'AUTO', 'RE-SCAN', 'DETOUR', 'STOP-BLOCK', 'HARD RULE', 'APPROACH-LOCK',
                    'YOLO: TARGET FOUND']:
            if tag in line:
                events.append((m.group(1) if m else '?', tag, line.strip()[:150]))
                break

for s in range(0, 50):
    d = f'{R}/step{s}'
    if not os.path.isdir(d):
        continue
    dt = open(f'{d}/details.txt').read() if os.path.exists(f'{d}/details.txt') else ''
    # 节点 (从 depth trace 或文件名)
    node = re.search(r'Node (\S+)', dt)
    dist = None
    m = re.search(r'distance_to_goal[=: ]+([\d.]+)', dt)
    if m:
        dist = float(m.group(1))
    yolo_t = 'TARGET' if re.search(r'target_found.{0,20}True|TARGET FOUND', dt, re.I) else ''
    act = re.search(r'Executing action (\[?\d+\]?)', dt)
    chosen = re.search(r'Chosen action[^\n]*', dt)
    print(f'--- step {s} ---')
    if dist:
        print(f'  dist={dist}')
    for mm in re.finditer(r'\[(PROXIMITY|CORRECT|BLACKLIST|APPROACH|FORCED-RETURN|AUTO|DETOUR|STOP-BLOCK|APPROACH-LOCK|RE-SCAN)\][^\n]*', dt):
        print('  EVT:', mm.group(0)[:140])
    # VLM reasoning 摘要
    r = re.search(r'"reasoning": "([^"]{0,180})', dt)
    if r:
        print('  VLM:', r.group(1))
