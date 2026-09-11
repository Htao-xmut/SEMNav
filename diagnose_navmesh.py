#!/usr/bin/env python3
"""
诊断 NavMesh 和路径查找问题
"""

import habitat_sim
import numpy as np
import gzip
import json

def diagnose_navmesh():
    # 加载 episode 数据
    with gzip.open('data/datasets/objectnav_hm3d_v2/val_mini/content/TEEsavR23oF.json.gz', 'rt') as f:
        data = json.load(f)
    
    episode = data['episodes'][0]
    scene_path = f"data/scene_datasets/hm3d/minival/{episode['scene_id']}/{episode['scene_id']}.glb"
    
    print(f"📍 Scene: {scene_path}")
    print(f"🚀 Start: {episode['start_position']}")
    print(f"🎯 Goal: {episode['goals'][0]['position']}")
    print(f"👀 View points: {episode['goals'][0]['view_points'][:2]}...")  # 只显示前2个
    
    # 初始化模拟器
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    backend_cfg.gpu_device_id = -1  # 软件渲染
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = 1.5
    agent_cfg.radius = 0.17
    
    sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(sim_cfg)
    
    print("\n✅ Simulator initialized")
    
    # 重建 NavMesh
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    if sim.recompute_navmesh(sim.pathfinder, navmesh_settings):
        print("✅ NavMesh recomputed")
    else:
        print("❌ NavMesh recomputation failed")
    
    # 检查起始点和目标点是否可通行
    start_pos = np.array(episode['start_position'], dtype=np.float32)
    goal_pos = np.array(episode['goals'][0]['position'], dtype=np.float32)
    
    start_snapped = sim.pathfinder.snap_point(start_pos)
    goal_snapped = sim.pathfinder.snap_point(goal_pos)
    
    print(f"\n🔍 Start snapped: {start_snapped} (valid: {not np.isnan(start_snapped).any()})")
    print(f"🔍 Goal snapped: {goal_snapped} (valid: {not np.isnan(goal_snapped).any()})")
    
    # 尝试计算路径
    path = habitat_sim.ShortestPath()
    path.requested_start = start_snapped
    path.requested_end = goal_snapped
    
    if sim.pathfinder.find_path(path):
        print(f"\n✅ Path found! Distance: {path.geodesic_distance:.2f}m")
        print(f"   Path length: {len(path.points)} points")
    else:
        print(f"\n❌ NO PATH FOUND!")
        print(f"   Start is navigable: {sim.pathfinder.is_navigable(start_snapped)}")
        print(f"   Goal is navigable: {sim.pathfinder.is_navigable(goal_snapped)}")
        
        # 检查是否在同一个连通组件
        start_island = sim.pathfinder.get_island(start_snapped)
        goal_island = sim.pathfinder.get_island(goal_snapped)
        print(f"   Start island: {start_island}")
        print(f"   Goal island: {goal_island}")
        if start_island != goal_island:
            print("   ⚠️  Start and goal are on different islands (not connected)!")
    
    sim.close()

if __name__ == "__main__":
    diagnose_navmesh()
