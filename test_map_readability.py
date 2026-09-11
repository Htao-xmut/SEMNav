"""地图可读性单测: 障碍配色可区分 / 远处地面进图 (bm15 核对修复)

bm15 用户核对发现: ① OCCUPIED (12,12,12) 与 UNKNOWN (30,30,30) 人眼
不可分, 看图以为"走过的区域又消失了"; ② update_live max_r=3.5 (照片
射线时代参数) 使 live 深度 (10m 精确) 画面 3.5m 外的地面全不进图,
50 步只画 10m² 绿。
运行: python -u test_map_readability.py
"""
import sys, inspect, logging
sys.path.insert(0, '/home/tao_h/avdb_habitat_converter/scripts')
import numpy as np

logging.basicConfig(level=logging.WARNING)

from avdb_depth import (ExplorationMap2D, MAP_UNKNOWN, MAP_EXPLORED,
                        MAP_FRONTIER, MAP_OCCUPIED, MAP_AGENT)

PASS, FAIL = [], []


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f'  [{info}]' if info and not cond else ''))


def color_dist(a, b):
    return float(np.hypot(*(np.array(a[:2]) - np.array(b[:2]))
                          if len(a) == 2 else
                          np.linalg.norm(np.array(a, float) - np.array(b, float))))


def test_colors_distinguishable():
    # 障碍色必须与未知黑拉开距离 (旧 12,12,12 距 30,30,30 仅 20.8)
    d_unknown = np.linalg.norm(np.array(MAP_OCCUPIED, float)
                               - np.array(MAP_UNKNOWN, float))
    check('配色: OCCUPIED vs UNKNOWN 距离>60 (bm15 旧 20.8 不可分)',
          d_unknown > 60, f'd={d_unknown:.0f}')
    d_frontier = np.linalg.norm(np.array(MAP_OCCUPIED, float)
                                - np.array(MAP_FRONTIER, float))
    check('配色: OCCUPIED vs FRONTIER 距离>60', d_frontier > 60,
          f'd={d_frontier:.0f}')
    # 五色两两可分 sanity (除 UNKNOWN-EXPLORED 外都应 >60)
    cols = {'UNKNOWN': MAP_UNKNOWN, 'EXPLORED': MAP_EXPLORED,
            'FRONTIER': MAP_FRONTIER, 'OCCUPIED': MAP_OCCUPIED,
            'AGENT': MAP_AGENT}
    bad = []
    names = list(cols)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = np.linalg.norm(np.array(cols[names[i]], float)
                               - np.array(cols[names[j]], float))
            if d < 60:
                bad.append(f'{names[i]}-{names[j]}={d:.0f}')
    check('配色: 五色两两距离>60', not bad, f'{bad}')


def test_render_mapping():
    # render 必须按常量渲染 (防将来改 regions 语义不动颜色)
    em = ExplorationMap2D(center=(0.0, 0.0), cell=0.05, size=600)
    em.regions[:] = 0
    em.regions[100, 100] = 1
    em.regions[200, 200] = 2
    em.regions[300, 300] = 3
    img = em.render()
    check('渲染: regions 0/1/2/3 → 对应四色',
          tuple(img[0, 0]) == MAP_UNKNOWN
          and tuple(img[100, 100]) == MAP_EXPLORED
          and tuple(img[200, 200]) == MAP_FRONTIER
          and tuple(img[300, 300]) == MAP_OCCUPIED,
          f'{tuple(img[300, 300])} vs {MAP_OCCUPIED}')


def test_far_floor_enters_map():
    # 合成几何: 深度帧全像素同值 d → 球壳。地面交点在水平距离
    # sqrt(d²-sensor_h²)。d=4.6, h=1.5 → 4.35m 处一圈"地面"。
    # 旧 max_r=3.5 (保留 <4.2m) 画不到; 新 5.0 (保留 <6.0m) 能进图。
    H, W = 540, 960
    d = 4.6
    em = ExplorationMap2D(center=(0.0, 0.0), cell=0.05, size=600)
    n_rays = em.update_live([0.0, 0.0, 0.0], (1, 0, 0, 0),
                            np.full((H, W), d, dtype=np.float32),
                            fov_deg=131.0)
    check('远地: 4.35m 地面产生射线 (旧 3.5 半径为 0)', n_rays > 0,
          f'n_rays={n_rays}')
    # 绿楔画到段长 60%, 其后 40% 是灰前沿 → 4.35m 真值处是灰格。
    # 断言"画出来的格子 (绿+灰) 覆盖到 4.0m 外" = 远处地面进图
    painted = np.isin(em.regions, [1, 2])
    gy, gx = np.where(painted)
    if len(gx) == 0:
        check('远地: 绿+灰格覆盖到 4.0m 外', False, '无格子')
        return
    r_cells = np.hypot(gx - 300, gy - 300) * 0.05
    far = (r_cells > 4.0).sum()
    check('远地: 绿+灰格覆盖到 4.0m 外 (真值 4.35m)',
          far > 50, f'far={far}, r范围={r_cells.min():.2f}-'
          f'{r_cells.max():.2f}m')
    # 旧参数对照: 3.5 时同样输入 0 射线 (复现 bm15 "房间进不了图")
    em_old = ExplorationMap2D(center=(0.0, 0.0), cell=0.05, size=600)
    n_old = em_old.update_live([0.0, 0.0, 0.0], (1, 0, 0, 0),
                               np.full((H, W), d, dtype=np.float32),
                               fov_deg=131.0, max_r=3.5)
    check('远地: 旧 max_r=3.5 同输入 0 射线 (复现 bm15 缺图)', n_old == 0,
          f'n_old={n_old}')


def test_default_max_r():
    sig = inspect.signature(ExplorationMap2D.update_live)
    check('参数: update_live 默认 max_r=5.0 (bm15 核对)',
          sig.parameters['max_r'].default == 5.0,
          f"got {sig.parameters['max_r'].default}")


if __name__ == '__main__':
    test_colors_distinguishable()
    test_render_mapping()
    test_far_floor_enters_map()
    test_default_max_r()
    print(f"\n{'='*40}\n✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        print('Failed:', *FAIL, sep='\n  - ')
        sys.exit(1)
