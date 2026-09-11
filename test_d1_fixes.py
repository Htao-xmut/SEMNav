"""D1 修复单测: seen 层 / corridor_prospect / 区域分走廊打折 / 停止票转化 / crop 门 / 覆盖停

不启动模拟器。合成深度帧 (平地 + 前墙) 喂 update_live。
运行: python -u test_d1_fixes.py
"""
import sys, os, types, logging
sys.path.insert(0, '/home/tao_h/VLMnav/src')
sys.path.insert(0, '/home/tao_h/avdb_habitat_converter/scripts')
import numpy as np

logging.basicConfig(level=logging.WARNING)

from avdb_depth import ExplorationMap2D
from avdb_sim_wrapper import AVDBSimWrapper

PASS, FAIL = [], []


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f'  [{info}]' if info and not cond else ''))


# ---------- 合成深度帧: 平地 + 前方 wall_z 米处墙, agent 原点, 单位朝向 ----------
def synth_depth(wall_z=3.0, H=540, W=960, fov=131.0, pitch=-0.45, cam_h=1.5):
    """identity yaw: 相机朝世界 -z。返回 (depth, )
    每像素: 地面命中 dd_floor / 墙命中 dd_wall 取近者; 都没有 → 0 (无效)
    """
    f = W / (2 * np.tan(np.radians(fov / 2)))
    jj, ii = np.meshgrid(np.arange(W), np.arange(H))
    ux = (jj - W / 2) / f
    uy = -(ii - H / 2) / f
    uz = -np.ones_like(ux)
    cp, sp = np.cos(pitch), np.sin(pitch)
    # pitch 旋转 (与 update_live 相同): body = Rx(pitch) @ cam
    vx = ux
    vy = cp * uy - sp * uz
    vz = sp * uy + cp * uz
    dd = np.zeros((H, W), np.float32)
    down = vy < -1e-6
    dd_floor = np.where(down, -cam_h / np.where(down, vy, -1), np.inf)
    fwd = vz < -1e-6
    dd_wall = np.where(fwd, wall_z / np.where(fwd, -vz, 1), np.inf)
    dd = np.minimum(dd_floor, dd_wall)
    dd[~np.isfinite(dd)] = 0.0
    dd[dd > 7] = 0.0
    return dd


FWD = np.pi  # identity yaw → 前向 = 世界 -z → bearing 180°


def test_update_live_seen():
    em = ExplorationMap2D(center=(0, 0))
    n = em.update_live((0.0, 0.0, 0.0), (1, 0, 0, 0), synth_depth(3.0))
    check('live: 射线数 > 10', n > 10, f'n={n}')
    check('live: 近处绿色已探索', em.cell_state(0.0, -1.0) == 1,
          f"state={em.cell_state(0.0, -1.0)}")
    check('live: 中段灰色前沿', em.cell_state(0.0, -2.5) == 2,
          f"state={em.cell_state(0.0, -2.5)}")
    check('live: 3m 处墙 = OCCUPIED', em.cell_state(0.0, -2.95) == 3,
          f"state={em.cell_state(0.0, -2.95)}")
    gx, gz = em._to_grid(0.0, -2.0)
    check('seen: 走廊中段已看过', em.seen[gz, gx])
    pr = em.corridor_prospect(0.0, 0.0, FWD)
    check('prospect: 视过的 3m 死胡同 = exhausted (内容看光, 墙封底)',
          pr['verdict'] == 'exhausted', f"{pr}")
    # 换个没看过的方向 (侧面, 90°) → fresh
    check('prospect: 未看过的侧向 = fresh',
          em.corridor_prospect(0.0, 0.0, np.pi / 2)['verdict'] == 'fresh')
    # 前进 1m 再看同方向 → 仍全程看过+墙 → exhausted
    em.update_live((0.0, 0.0, -1.0), (1, 0, 0, 0), synth_depth(3.0))
    pr2 = em.corridor_prospect(0.0, -1.0, FWD)
    check('prospect: 二次看同走廊 = exhausted (看光了)',
          pr2['verdict'] == 'exhausted', f"{pr2}")


def test_prospect_cases():
    em = ExplorationMap2D(center=(0, 0))
    # (a) 全 UNKNOWN 走廊 → fresh
    check('prospect: 全未知走廊 = fresh',
          em.corridor_prospect(0, 0, FWD)['verdict'] == 'fresh')
    # (b) 灰色前沿走廊 → frontier: 近场涂绿 + 0.5m 起灰
    for z in np.arange(-0.3, -0.5, -0.05):
        for dx in (-0.25, 0, 0.25):
            gx2, gz2 = em._to_grid(dx, z)
            em.regions[gz2, gx2] = 1
    for z in np.arange(-0.5, -3.5, -0.05):
        for dx in (-0.25, 0, 0.25):
            gx2, gz2 = em._to_grid(dx, z)
            em.regions[gz2, gx2] = 2
    check('prospect: 灰前沿走廊 = frontier',
          em.corridor_prospect(0, 0, FWD)['verdict'] == 'frontier')
    # (c) 看光 + 1m 处墙 → exhausted (G6 死角)
    em2 = ExplorationMap2D(center=(0, 0))
    for z in np.arange(-0.3, -0.9, -0.05):
        for x in np.arange(-0.5, 0.5, 0.05):
            gx, gz = em2._to_grid(x, z)
            em2.regions[gz, gx] = 1
            em2.seen[gz, gx] = True
    for x in np.arange(-0.5, 0.5, 0.05):        # 1m 处一排墙
        gx, gz = em2._to_grid(x, -1.0)
        em2.regions[gz, gx] = 3
    pr = em2.corridor_prospect(0, 0, FWD)
    check('prospect: 看光+贴脸墙 = exhausted (G6 step1 死角)',
          pr['verdict'] == 'exhausted', f"{pr}")
    # (d) seen 但未涂 region (沙发后地面) → 不算 fresh
    em3 = ExplorationMap2D(center=(0, 0))
    for z in np.arange(-0.3, -3.5, -0.05):
        for x in np.arange(-0.3, 0.3, 0.05):
            gx, gz = em3._to_grid(x, z)
            em3.seen[gz, gx] = True             # 只 seen, region 仍 0
    check('prospect: seen-only 走廊不算 fresh (沙发后看过)',
          em3.corridor_prospect(0, 0, FWD)['verdict'] == 'exhausted')


def make_wrapper():
    """复用 test_nav_fixes 的骨架 + exploration_map"""
    w = object.__new__(AVDBSimWrapper)
    w.nav_graph = {'nodes': {
        'N1': {'world_pos': [0.0, 0.0, 0.0], 'direction': [0, 0, 1]},
        'F':  {'world_pos': [0.0, 0.0, -0.65], 'direction': [0, 0, 1]},   # 前 (-z)
        'R':  {'world_pos': [0.65, 0.0, 0.0], 'direction': [0, 0, 1]},   # 右
    }, 'graph': {
        'N1': {'F': {'edge_type': 'forward', 'distance': 0.65},
               'R': {'edge_type': 'right', 'distance': 0.65}},
        'F': {'N1': {'edge_type': 'backward', 'distance': 0.65}},
        'R': {'N1': {'edge_type': 'left', 'distance': 0.65}},
    }}
    w.visited_nodes = {}
    w.visited_area_cells = set()
    w.rejected_areas = []
    w.exploration_map = None
    w.depth_cfg = None
    w.current_node = 'N1'
    w._walk_hist = []

    class CamStub:
        cam_R = staticmethod(lambda d: np.eye(3))
    w.depth_cam = CamStub()
    return w


def test_area_adjust_corridor():
    # 前向 (F, bearing 180°) 走廊看光+墙; 右向 (R, bearing 90°) 全未知
    w = make_wrapper()
    em = ExplorationMap2D(center=(0, 0))
    for z in np.arange(-0.3, -0.9, -0.05):
        for x in np.arange(-0.4, 0.4, 0.05):
            gx, gz = em._to_grid(x, z)
            em.regions[gz, gx] = 1
            em.seen[gz, gx] = True
    for x in np.arange(-0.4, 0.4, 0.05):
        gx, gz = em._to_grid(x, -0.9)
        em.regions[gz, gx] = 3
    w.exploration_map = em
    w.visited_area_cells = {(0, 0)}    # F/R 都在别格? F(0,-0.65)→格(0,-1), R(0.65,0)→格(0,0)
    # R 与 N1 同格 (0,0) 已访问 → FRESH 不触发; 挪 R 到 2.2m 外新格
    w.nav_graph['nodes']['R']['world_pos'] = [2.2, 0.0, 0.0]
    w.nav_graph['graph']['N1']['R']['distance'] = 2.2
    scored = {'recommended': {'rep_deg': 180.0, 'score': 0.9, 'safe_dist_m': 0.6,
                              'confidence': 0.9, 'source': 'graph'},
              'alternatives': [{'rep_deg': 90.0, 'score': 0.85, 'safe_dist_m': 2.2,
                                'confidence': 0.9, 'source': 'graph'}]}
    out = w._area_exploration_adjust(scored, 'N1')
    rec = out['recommended']
    check('区域分: 前向死角 (SEEN-DEADEND) 零加分 → 推荐翻转到右侧 fresh',
          abs(rec['rep_deg'] - 90.0) < 1e-6,
          f"rec={rec.get('rep_deg')}, tag={rec.get('area_tag')}")
    fwd = [a for a in out['alternatives'] if abs(a['rep_deg'] - 180.0) < 1e-6][0]
    check('区域分: 死角候选带 SEEN-DEADEND 标签且不加弹',
          fwd.get('area_tag') == 'SEEN-DEADEND'
          and fwd['score'] <= 0.9 + 1e-9, f"tag={fwd.get('area_tag')} score={fwd['score']}")
    # 地图缺失 → 保守旧行为 (全额)
    w2 = make_wrapper()
    w2.exploration_map = None
    w2.visited_area_cells = {(0, 0)}
    w2.nav_graph['nodes']['R']['world_pos'] = [2.2, 0.0, 0.0]
    w2.nav_graph['graph']['N1']['R']['distance'] = 2.2
    out2 = w2._area_exploration_adjust(scored, 'N1')
    r2 = [a for a in out2['alternatives'] if abs(a['rep_deg'] - 90.0) < 1e-6][0]
    check('区域分: 无地图时保守全额 (旧行为)',
          r2['score'] > 0.85 and r2.get('area_tag', '').startswith('FRESH'),
          f"score={r2['score']} tag={r2.get('area_tag')}")


if __name__ == '__main__':
    test_update_live_seen()
    test_prospect_cases()
    test_area_adjust_corridor()
    print(f"\n{'='*40}\n✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        print('Failed:', *FAIL, sep='\n  - ')
        sys.exit(1)
