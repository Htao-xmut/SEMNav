#!/usr/bin/env python3
"""
快速浏览所有 HM3D 场景 - 为每个场景生成一张缩略图
"""

import os
import sys
import numpy as np
from PIL import Image
import habitat_sim


def quick_scene_preview(scene_path, output_path, resolution=(320, 240)):
    """
    快速生成场景预览图
    
    Args:
        scene_path: 场景文件路径
        output_path: 输出图片路径
        resolution: 图像分辨率
    """
    
    # 配置模拟器
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.gpu_device_id = -1  # 软件渲染
    
    # RGB 传感器
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = list(resolution)
    rgb_sensor_spec.position = [0.0, 1.5, 0.0]
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor_spec]
    
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    
    try:
        sim = habitat_sim.Simulator(cfg)
        
        # 设置初始位置
        state = habitat_sim.AgentState()
        state.position = [0.0, 0.0, 0.0]
        state.rotation = [1.0, 0.0, 0.0, 0.0]
        sim.get_agent(0).set_state(state)
        
        # 渲染
        obs = sim.get_sensor_observations()
        rgb = obs['color_sensor'][:, :, :3]
        
        # 保存图片
        image = Image.fromarray(rgb.astype(np.uint8))
        image.save(output_path)
        
        sim.close()
        return True
        
    except Exception as e:
        print(f"   ❌ 错误: {e}")
        return False


def main():
    base_dir = "/home/tao_h/VLMnav/data/scene_datasets/hm3d/minival"
    output_dir = "/home/tao_h/VLMnav/scene_visualization/previews"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("🖼️  HM3D Minival 场景快速预览")
    print("=" * 70)
    print()
    
    scenes_generated = 0
    
    for scene_folder in sorted(os.listdir(base_dir)):
        scene_path_full = os.path.join(base_dir, scene_folder)
        if not os.path.isdir(scene_path_full):
            continue
        
        # 查找 .glb 文件
        glb_file = None
        for file in os.listdir(scene_path_full):
            if file.endswith('.glb'):
                glb_file = file
                break
        
        if not glb_file:
            continue
        
        scene_glb = os.path.join(scene_path_full, glb_file)
        output_path = os.path.join(output_dir, f"{scene_folder}.png")
        
        print(f"[{scenes_generated + 1}] {scene_folder}", end=" ... ")
        
        if quick_scene_preview(scene_glb, output_path):
            size_kb = os.path.getsize(output_path) / 1024
            print(f"✅ ({size_kb:.1f} KB)")
            scenes_generated += 1
        else:
            print("❌")
    
    print()
    print("=" * 70)
    print(f"✅ 成功生成 {scenes_generated} 个场景预览")
    print(f"📁 预览图目录: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
