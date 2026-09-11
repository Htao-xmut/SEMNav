#!/usr/bin/env python3
"""
HM3D 场景可视化工具
使用 Habitat-Sim 加载并渲染 HM3D 场景，生成全景图和漫游视频
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import habitat_sim


def create_scene_viewer(scene_path, output_dir="scene_visualization", num_frames=60):
    """
    创建场景可视化：渲染多个角度的视图并生成漫游动画
    
    Args:
        scene_path: 场景文件路径 (.glb)
        output_dir: 输出目录
        num_frames: 生成的帧数（用于动画）
    """
    
    print("=" * 60)
    print("🎬 HM3D 场景可视化工具")
    print("=" * 60)
    print(f"\n📂 场景文件: {scene_path}")
    
    if not os.path.exists(scene_path):
        print(f"❌ 场景文件不存在: {scene_path}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    scene_name = os.path.basename(os.path.dirname(scene_path))
    print(f"📁 输出目录: {output_dir}/{scene_name}")
    
    # 配置模拟器
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.gpu_device_id = -1  # 使用软件渲染
    
    # 配置传感器
    sensor_specs = []
    
    # RGB 传感器
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [480, 640]
    rgb_sensor_spec.position = [0.0, 1.5, 0.0]  # 眼睛高度 1.5m
    rgb_sensor_spec.orientation = [0.0, 0.0, 0.0]
    sensor_specs.append(rgb_sensor_spec)
    
    # Depth 传感器
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [480, 640]
    depth_sensor_spec.position = [0.0, 1.5, 0.0]
    depth_sensor_spec.orientation = [0.0, 0.0, 0.0]
    sensor_specs.append(depth_sensor_spec)
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    
    print("\n🔧 初始化 Habitat-Sim...")
    sim = habitat_sim.Simulator(cfg)
    print("✅ 模拟器初始化成功")
    
    # 获取场景信息
    print(f"\n📊 场景信息:")
    print(f"   - 场景名称: {scene_name}")
    
    # 尝试获取边界框
    try:
        bbox = sim.get_agent(0).scene.get_bounding_box()
        print(f"   - 场景边界: min={bbox.min}, max={bbox.max}")
        scene_size = bbox.max - bbox.min
        print(f"   - 场景尺寸: {scene_size[0]:.1f}m x {scene_size[1]:.1f}m x {scene_size[2]:.1f}m")
    except Exception as e:
        print(f"   - 场景边界: 无法获取 ({e})")
    
    # 设置初始位置（场景中心附近）
    initial_state = habitat_sim.AgentState()
    initial_state.position = [0.0, 0.0, 0.0]
    initial_state.rotation = [1.0, 0.0, 0.0, 0.0]  # 单位四元数
    sim.get_agent(0).set_state(initial_state)
    
    print("\n📸 开始渲染场景...")
    
    # 渲染第一帧（初始视角）
    observations = sim.get_sensor_observations()
    rgb = observations['color_sensor'][:, :, :3]
    depth = observations['depth_sensor']
    
    # 保存 RGB 图像
    rgb_image = Image.fromarray(rgb.astype(np.uint8))
    rgb_path = f"{output_dir}/{scene_name}/initial_view.png"
    os.makedirs(os.path.dirname(rgb_path), exist_ok=True)
    rgb_image.save(rgb_path)
    print(f"✅ 保存初始视角: {rgb_path}")
    
    # 保存深度图（归一化到 0-255）
    depth_normalized = (depth / depth.max() * 255).astype(np.uint8)
    depth_image = Image.fromarray(depth_normalized)
    depth_path = f"{output_dir}/{scene_name}/depth_map.png"
    depth_image.save(depth_path)
    print(f"✅ 保存深度图: {depth_path}")
    
    # 生成环绕动画（相机绕 Y 轴旋转）
    print(f"\n🎥 生成环绕动画 ({num_frames} 帧)...")
    frames = []
    
    for i in range(num_frames):
        # 计算旋转角度
        angle = 2 * np.pi * i / num_frames
        
        # 设置相机位置（圆形轨道）
        radius = 5.0  # 半径 5 米
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        
        state = habitat_sim.AgentState()
        state.position = [x, 1.5, z]
        
        # 相机朝向原点
        import quaternion
        target_direction = np.array([-x, 0, -z])
        target_direction = target_direction / np.linalg.norm(target_direction)
        
        # 计算朝向的四元数
        yaw = np.arctan2(-x, -z)
        rot = quaternion.from_euler_angles([0, yaw, 0])
        state.rotation = quaternion.as_float_array(rot)
        
        sim.get_agent(0).set_state(state)
        
        # 渲染
        obs = sim.get_sensor_observations()
        rgb_frame = obs['color_sensor'][:, :, :3]
        frames.append(rgb_frame)
        
        if (i + 1) % 10 == 0:
            print(f"   进度: {i+1}/{num_frames} 帧")
    
    # 保存所有帧为单独的图片
    print(f"\n💾 保存动画帧...")
    frames_dir = f"{output_dir}/{scene_name}/frames"
    os.makedirs(frames_dir, exist_ok=True)
    
    for i, frame in enumerate(frames):
        frame_image = Image.fromarray(frame.astype(np.uint8))
        frame_path = f"{frames_dir}/frame_{i:03d}.png"
        frame_image.save(frame_path)
    
    print(f"✅ 保存了 {len(frames)} 帧到: {frames_dir}")
    
    # 尝试生成 GIF（如果安装了 imageio）
    try:
        import imageio
        gif_path = f"{output_dir}/{scene_name}/orbit_animation.gif"
        imageio.mimsave(gif_path, frames, duration=0.1, loop=0)
        print(f"✅ 生成 GIF 动画: {gif_path}")
    except ImportError:
        print("⚠️  未安装 imageio，跳过 GIF 生成")
        print("   安装命令: pip install imageio")
    
    # 生成场景统计信息
    print(f"\n📈 场景统计:")
    print(f"   - RGB 图像尺寸: {rgb.shape}")
    print(f"   - 深度图范围: [{depth.min():.2f}, {depth.max():.2f}] 米")
    print(f"   - 平均深度: {depth.mean():.2f} 米")
    
    # 清理
    sim.close()
    
    print("\n" + "=" * 60)
    print("🎉 场景可视化完成！")
    print("=" * 60)
    print(f"\n查看结果:")
    print(f"  - 初始视角: {rgb_path}")
    print(f"  - 深度图: {depth_path}")
    if 'gif_path' in locals():
        print(f"  - 环绕动画: {gif_path}")
    print(f"  - 动画帧目录: {frames_dir}")


def list_available_scenes(base_dir="/home/tao_h/VLMnav/data/scene_datasets/hm3d/minival"):
    """列出可用的场景"""
    print("\n📋 可用的 HM3D Minival 场景:")
    print("-" * 60)
    
    if not os.path.exists(base_dir):
        print(f"❌ 目录不存在: {base_dir}")
        return []
    
    scenes = []
    for scene_folder in sorted(os.listdir(base_dir)):
        scene_path = os.path.join(base_dir, scene_folder)
        if os.path.isdir(scene_path):
            # 查找 .glb 文件
            for file in os.listdir(scene_path):
                if file.endswith('.glb'):
                    glb_path = os.path.join(scene_path, file)
                    size_mb = os.path.getsize(glb_path) / (1024 * 1024)
                    scenes.append((scene_folder, glb_path, size_mb))
                    print(f"  [{len(scenes):2d}] {scene_folder:30s} ({size_mb:.1f} MB)")
                    break
    
    print("-" * 60)
    print(f"总计: {len(scenes)} 个场景\n")
    return scenes


def main():
    parser = argparse.ArgumentParser(description='HM3D 场景可视化工具')
    parser.add_argument('--scene', type=int, default=None, 
                       help='选择要可视化的场景编号（使用 --list 查看）')
    parser.add_argument('--path', type=str, default=None,
                       help='直接指定场景文件路径')
    parser.add_argument('--output', type=str, default='scene_visualization',
                       help='输出目录')
    parser.add_argument('--frames', type=int, default=60,
                       help='动画帧数')
    parser.add_argument('--list', action='store_true',
                       help='列出所有可用场景')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_scenes()
        return
    
    # 确定场景路径
    if args.path:
        scene_path = args.path
    elif args.scene is not None:
        scenes = list_available_scenes()
        if 0 < args.scene <= len(scenes):
            scene_path = scenes[args.scene - 1][1]
        else:
            print(f"❌ 无效的场景编号: {args.scene}")
            return
    else:
        # 默认使用第一个场景
        scenes = list_available_scenes()
        if scenes:
            scene_path = scenes[0][1]
            print(f"使用默认场景: {scenes[0][0]}")
        else:
            print("❌ 没有找到可用的场景")
            return
    
    # 创建可视化
    create_scene_viewer(scene_path, args.output, args.frames)


if __name__ == "__main__":
    main()
