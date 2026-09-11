import json, gzip, glob, numpy as np
dd = 'data/datasets/objectnav_avdb_home001_1/val/content'
eps = []
for fn in sorted(glob.glob(dd + '/*.json.gz')):
    data = json.load(gzip.open(fn, 'rt'))
    eps += list(data.values()) if isinstance(data, dict) else data.get('episodes', [])
coca = [e for e in eps if e.get('object_category') == 'coca_cola_glass_bottle']
ep = coca[5]
goal = np.array(ep['goals'][0]['position'])
print('goal xyz:', goal)
ng = json.load(open('/home/tao_h/avdb_habitat_converter/data/processed/navigation_graph.json'))
black = [(-2.4, 2.1), (2.6, 3.4), (2.2, 1.8), (2.3, 0.2)]
for c in black:
    print(f'blacklist center {c}: dist to goal(xz) = {np.linalg.norm(np.array(c) - goal[[0, 2]]):.2f}m')
for n in ['000110008970101.jpg', '000110008980101.jpg', '000110006140101.jpg']:
    p = np.array(ng['nodes'][n]['world_pos'])
    print(f'{n}: pos ({p[0]:.1f},{p[2]:.1f}), dist to goal = {np.linalg.norm(p[[0, 2]] - goal[[0, 2]]):.2f}m')
