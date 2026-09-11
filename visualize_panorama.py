#!/usr/bin/env python3
"""
HM3D 全景可视化工具
生成真正的 360° 全景图和立方体贴图
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import habitat_sim
import math


def render_equirectangular(sim, resolution=(512, 1024)):
    """
    渲染等距柱状投影全景图 (Equirectangular Projection)
    
    Args:
        sim: Habitat 模拟器实例
        resolution: (height, width) 分辨率
    
    Returns:
        rgb_panorama: RGB 全景图
        depth_panorama: 深度全景图
    """
    height, width = resolution
    
    # 创建全景图数组
    rgb_panorama = np.zeros((height, width, 3), dtype=np.uint8)
    depth_panorama = np.zeros((height, width), dtype=np.float32)
    
    agent = sim.get_agent(0)
    initial_state = agent.get_state()
    
    print(f"   渲染全景图 {width}x{height}...")
    
    # 遍历每个像素，计算对应的视角方向
    for v in range(height):
        for u in range(width):
            # 将球面坐标转换为相机方向
            # theta: 水平角度 [0, 2π]
            # phi: 垂直角度 [-π/2, π/2]
            theta = 2 * math.pi * u / width
            phi = math.pi * v / height - math.pi / 2
            
            # 计算观察方向向量
            x = math.cos(phi) * math.sin(theta)
            y = math.sin(phi)
            z = math.cos(phi) * math.cos(theta)
            
            # 归一化
            norm = math.sqrt(x*x + y*y + z*z)
            x, y, z = x/norm, y/norm, z/norm
            
            # 计算欧拉角（yaw, pitch, roll）
            yaw = math.atan2(x, z)
            pitch = math.asin(y)
            
            # 设置相机朝向
            from scipy.spatial.transform import Rotation as R
            rot = R.from_euler('yxz', [pitch, yaw, 0], degrees=False)
            quat = rot.as_quat()  # [x, y, z, w]
            
            # Habitat 使用 [w, x, y, z] 格式
            quat_habitat = [quat[3], quat[0], quat[1], quat[2]]
            
            state = agent.get_state()
            state.rotation = quat_habitat
            agent.set_state(state)
            
            # 渲染
            obs = sim.get_sensor_observations()
            rgb_panorama[v, u] = obs['color_sensor'][0, 0, :3]
            depth_panorama[v, u] = obs['depth_sensor'][0, 0]
        
        if (v + 1) % 50 == 0:
            print(f"   进度: {v+1}/{height} 行")
    
    # 恢复初始状态
    agent.set_state(initial_state)
    
    return rgb_panorama, depth_panorama


def render_cube_faces(sim, resolution=512):
    """
    渲染立方体贴图的 6 个面
    
    Args:
        sim: Habitat 模拟器实例
        resolution: 每个面的分辨率
    
    Returns:
        faces: dict with keys ['front', 'back', 'left', 'right', 'top', 'bottom']
    """
    agent = sim.get_agent(0)
    initial_state = agent.get_state()
    
    # 定义 6 个面的朝向（欧拉角：pitch, yaw, roll）
    face_orientations = {
        'front':  (0, 0, 0),              # 前
        'back':   (0, math.pi, 0),        # 后
        'left':   (0, math.pi/2, 0),      # 左
        'right':  (0, -math.pi/2, 0),     # 右
        'top':    (-math.pi/2, 0, 0),     # 上
        'bottom': (math.pi/2, 0, 0),      # 下
    }
    
    faces = {}
    
    print(f"   渲染立方体贴图 6 个面 ({resolution}x{resolution})...")
    
    for face_name, (pitch, yaw, roll) in face_orientations.items():
        # 计算四元数
        from scipy.spatial.transform import Rotation as R
        rot = R.from_euler('yxz', [pitch, yaw, roll], degrees=False)
        quat = rot.as_quat()  # [x, y, z, w]
        quat_habitat = [quat[3], quat[0], quat[1], quat[2]]
        
        # 设置相机朝向
        state = agent.get_state()
        state.rotation = quat_habitat
        agent.set_state(state)
        
        # 渲染
        obs = sim.get_sensor_observations()
        rgb = obs['color_sensor'][:, :, :3]
        depth = obs['depth_sensor']
        
        faces[face_name] = {
            'rgb': rgb.astype(np.uint8),
            'depth': depth
        }
        
        print(f"   ✅ {face_name:8s} 完成")
    
    # 恢复初始状态
    agent.set_state(initial_state)
    
    return faces


def create_panorama_viewer(scene_path, output_dir="panorama_visualization", 
                          pano_res=(512, 1024), cube_res=512):
    """
    创建全景图查看器
    
    Args:
        scene_path: 场景文件路径
        output_dir: 输出目录
        pano_res: 全景图分辨率 (height, width)
        cube_res: 立方体贴图每个面的分辨率
    """
    
    print("=" * 70)
    print("🌐 HM3D 全景可视化工具")
    print("=" * 70)
    print(f"\n📂 场景文件: {scene_path}")
    
    if not os.path.exists(scene_path):
        print(f"❌ 场景文件不存在: {scene_path}")
        return
    
    # 创建输出目录
    scene_name = os.path.basename(os.path.dirname(scene_path))
    output_path = f"{output_dir}/{scene_name}"
    os.makedirs(output_path, exist_ok=True)
    print(f"📁 输出目录: {output_path}")
    
    # 配置模拟器
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.gpu_device_id = -1  # 软件渲染
    
    # RGB 传感器
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [512, 512]  # 正方形用于立方体贴图
    rgb_sensor_spec.position = [0.0, 1.5, 0.0]
    rgb_sensor_spec.orientation = [0.0, 0.0, 0.0]
    
    # Depth 传感器
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [512, 512]
    depth_sensor_spec.position = [0.0, 1.5, 0.0]
    depth_sensor_spec.orientation = [0.0, 0.0, 0.0]
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]
    
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    
    print("\n🔧 初始化 Habitat-Sim...")
    sim = habitat_sim.Simulator(cfg)
    print("✅ 模拟器初始化成功")
    
    # 设置初始位置
    agent = sim.get_agent(0)
    state = habitat_sim.AgentState()
    state.position = [0.0, 0.0, 0.0]
    state.rotation = [1.0, 0.0, 0.0, 0.0]
    agent.set_state(state)
    
    # 方法 1: 生成立方体贴图（快速）
    print("\n📦 方法 1: 生成立方体贴图 (6 个面)")
    print("-" * 70)
    cube_faces = render_cube_faces(sim, cube_res)
    
    # 保存立方体贴图
    cube_dir = f"{output_path}/cube_map"
    os.makedirs(cube_dir, exist_ok=True)
    
    for face_name, data in cube_faces.items():
        rgb_path = f"{cube_dir}/{face_name}.png"
        depth_path = f"{cube_dir}/{face_name}_depth.png"
        
        Image.fromarray(data['rgb']).save(rgb_path)
        
        # 保存深度图（归一化）
        depth_norm = data['depth'].squeeze()  # 移除多余维度
        if depth_norm.max() > 0:
            depth_norm = (depth_norm / depth_norm.max() * 255).astype(np.uint8)
        else:
            depth_norm = np.zeros(depth_norm.shape, dtype=np.uint8)
        Image.fromarray(depth_norm).save(depth_path)
        
        print(f"   💾 {face_name:8s}: {rgb_path}")
    
    # 创建立方体贴图拼接图（可选）
    print("\n🖼️  创建立方体贴图拼接视图...")
    try:
        combined_width = cube_res * 4
        combined_height = cube_res * 3
        combined = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        
        def extract_rgb(face_data):
            """从面部数据中提取 RGB 通道（移除 Alpha）"""
            rgb = face_data['rgb']
            if len(rgb.shape) == 3 and rgb.shape[2] >= 3:
                return rgb[:, :, :3]  # 只取 RGB
            return rgb
        
        # 布局:
        #     [top]
        # [left][front][right][back]
        #   [bottom]
        
        combined[0:cube_res, cube_res:2*cube_res] = extract_rgb(cube_faces['top'])
        combined[2*cube_res:3*cube_res, cube_res:2*cube_res] = extract_rgb(cube_faces['bottom'])
        combined[cube_res:2*cube_res, 0:cube_res] = extract_rgb(cube_faces['left'])
        combined[cube_res:2*cube_res, cube_res:2*cube_res] = extract_rgb(cube_faces['front'])
        combined[cube_res:2*cube_res, 2*cube_res:3*cube_res] = extract_rgb(cube_faces['right'])
        combined[cube_res:2*cube_res, 3*cube_res:4*cube_res] = extract_rgb(cube_faces['back'])
        
        combined_path = f"{output_path}/cube_map_combined.png"
        Image.fromarray(combined).save(combined_path)
        print(f"   ✅ 拼接图: {combined_path}")
    except Exception as e:
        print(f"   ⚠️  拼接图生成失败: {e}")
        print(f"   提示: 可以单独查看每个面的图片")
    
    # 方法 2: 生成等距柱状全景图（慢但完整）
    print("\n🌍 方法 2: 生成等距柱状全景图 (360°)")
    print("-" * 70)
    print("⚠️  注意: 逐像素渲染较慢，请耐心等待...")
    
    try:
        rgb_pano, depth_pano = render_equirectangular(sim, pano_res)
        
        # 保存全景图
        pano_rgb_path = f"{output_path}/equirectangular_rgb.png"
        pano_depth_path = f"{output_path}/equirectangular_depth.png"
        
        Image.fromarray(rgb_pano).save(pano_rgb_path)
        
        # 保存深度全景图
        if depth_pano.max() > 0:
            depth_norm = (depth_pano / depth_pano.max() * 255).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth_pano, dtype=np.uint8)
        Image.fromarray(depth_norm).save(pano_depth_path)
        
        print(f"\n✅ 全景图生成成功!")
        print(f"   💾 RGB 全景: {pano_rgb_path}")
        print(f"   💾 深度全景: {pano_depth_path}")
        print(f"   📊 分辨率: {pano_res[1]}x{pano_res[0]}")
        
    except Exception as e:
        print(f"\n⚠️  全景图生成失败: {e}")
        print("   提示: 可以使用立方体贴图代替")
    
    # 生成 HTML 查看器
    print("\n🌐 生成 HTML 全景查看器...")
    html_content = create_html_viewer(scene_name, output_path, cube_faces)
    html_path = f"{output_path}/viewer.html"
    with open(html_path, 'w') as f:
        f.write(html_content)
    print(f"   💾 HTML 查看器: {html_path}")
    
    # 清理
    sim.close()
    
    print("\n" + "=" * 70)
    print("🎉 全景可视化完成！")
    print("=" * 70)
    print(f"\n查看方式:")
    print(f"  1. 打开 HTML 查看器: xdg-open {html_path}")
    print(f"  2. 查看立方体贴图: ls {cube_dir}/")
    if os.path.exists(pano_rgb_path):
        print(f"  3. 查看全景图: xdg-open {pano_rgb_path}")


def create_html_viewer(scene_name, output_path, cube_faces):
    """创建简单的 HTML 全景查看器"""
    
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{scene_name} - 全景查看器</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        
        h1 {{
            text-align: center;
            color: #667eea;
            margin-bottom: 10px;
        }}
        
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section-title {{
            font-size: 1.5em;
            color: #333;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
            padding-left: 15px;
        }}
        
        .cube-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            max-width: 800px;
            margin: 0 auto;
        }}
        
        .cube-face {{
            position: relative;
            aspect-ratio: 1;
            overflow: hidden;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .cube-face img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}
        
        .cube-face .label {{
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(0,0,0,0.6);
            color: white;
            padding: 5px;
            text-align: center;
            font-size: 0.9em;
        }}
        
        .empty {{
            visibility: hidden;
        }}
        
        .panorama-container {{
            text-align: center;
        }}
        
        .panorama-img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        
        .info-box {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }}
        
        .info-item {{
            margin-bottom: 10px;
            color: #555;
        }}
        
        .info-label {{
            font-weight: bold;
            color: #333;
        }}
        
        footer {{
            text-align: center;
            color: #999;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏠 {scene_name}</h1>
        <p class="subtitle">HM3D 场景全景查看器</p>
        
        <div class="section">
            <h2 class="section-title">📦 立方体贴图 (Cube Map)</h2>
            <p style="margin-bottom: 15px; color: #666;">
                立方体贴图由 6 个面组成，展示了从中心点向各个方向的视图。
            </p>
            <div class="cube-grid">
                <div class="cube-face empty"></div>
                <div class="cube-face">
                    <img src="cube_map/top.png" alt="Top">
                    <div class="label">↑ 上方 (Top)</div>
                </div>
                <div class="cube-face empty"></div>
                <div class="cube-face empty"></div>
                
                <div class="cube-face">
                    <img src="cube_map/left.png" alt="Left">
                    <div class="label">← 左侧 (Left)</div>
                </div>
                <div class="cube-face">
                    <img src="cube_map/front.png" alt="Front">
                    <div class="label">→ 前方 (Front)</div>
                </div>
                <div class="cube-face">
                    <img src="cube_map/right.png" alt="Right">
                    <div class="label">→ 右侧 (Right)</div>
                </div>
                <div class="cube-face">
                    <img src="cube_map/back.png" alt="Back">
                    <div class="label">← 后方 (Back)</div>
                </div>
                
                <div class="cube-face empty"></div>
                <div class="cube-face">
                    <img src="cube_map/bottom.png" alt="Bottom">
                    <div class="label">↓ 下方 (Bottom)</div>
                </div>
                <div class="cube-face empty"></div>
                <div class="cube-face empty"></div>
            </div>
            
            <div style="text-align: center; margin-top: 20px;">
                <img src="cube_map_combined.png" alt="Combined Cube Map" 
                     style="max-width: 100%; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                <p style="margin-top: 10px; color: #666; font-size: 0.9em;">
                    立方体贴图拼接视图
                </p>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">🌍 等距柱状全景图 (Equirectangular)</h2>
            <p style="margin-bottom: 15px; color: #666;">
                360° 全景图，可以在支持全景查看的软件中打开（如 Photoshop、Blender 或在线全景查看器）。
            </p>
            <div class="panorama-container">
                <img src="equirectangular_rgb.png" alt="360° Panorama" class="panorama-img">
                <p style="margin-top: 10px; color: #666; font-size: 0.9em;">
                    RGB 全景图 - 可以用鼠标拖动查看不同方向
                </p>
            </div>
        </div>
        
        <div class="info-box">
            <h3 style="margin-bottom: 15px; color: #667eea;">📊 场景信息</h3>
            <div class="info-item">
                <span class="info-label">场景名称:</span> {scene_name}
            </div>
            <div class="info-item">
                <span class="info-label">数据格式:</span> glTF 2.0 (.glb)
            </div>
            <div class="info-item">
                <span class="info-label">渲染时间:</span> {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
            <div class="info-item">
                <span class="info-label">提示:</span> 
                将 equirectangular_rgb.png 上传到在线全景查看器可以获得更好的交互体验
            </div>
        </div>
        
        <footer>
            <p>Generated by VLMnav Panorama Viewer | HM3D Dataset</p>
        </footer>
    </div>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description='HM3D 全景可视化工具')
    parser.add_argument('--scene', type=int, default=None, 
                       help='选择要可视化的场景编号')
    parser.add_argument('--path', type=str, default=None,
                       help='直接指定场景文件路径')
    parser.add_argument('--output', type=str, default='panorama_visualization',
                       help='输出目录')
    parser.add_argument('--pano-res', type=str, default='512x1024',
                       help='全景图分辨率 WxH (默认: 1024x512)')
    parser.add_argument('--cube-res', type=int, default=512,
                       help='立方体贴图分辨率 (默认: 512)')
    parser.add_argument('--list', action='store_true',
                       help='列出所有可用场景')
    
    args = parser.parse_args()
    
    if args.list:
        base_dir = "/home/tao_h/VLMnav/data/scene_datasets/hm3d/minival"
        print("\n📋 可用的 HM3D Minival 场景:")
        print("-" * 60)
        
        if not os.path.exists(base_dir):
            print(f"❌ 目录不存在: {base_dir}")
            return
        
        scenes = []
        for i, scene_folder in enumerate(sorted(os.listdir(base_dir)), 1):
            scene_path = os.path.join(base_dir, scene_folder)
            if os.path.isdir(scene_path):
                for file in os.listdir(scene_path):
                    if file.endswith('.glb'):
                        size_mb = os.path.getsize(os.path.join(scene_path, file)) / (1024 * 1024)
                        scenes.append((i, scene_folder, size_mb))
                        print(f"  [{i:2d}] {scene_folder:30s} ({size_mb:.1f} MB)")
                        break
        
        print("-" * 60)
        print(f"总计: {len(scenes)} 个场景\n")
        return
    
    # 确定场景路径
    if args.path:
        scene_path = args.path
    elif args.scene is not None:
        base_dir = "/home/tao_h/VLMnav/data/scene_datasets/hm3d/minival"
        scenes = []
        for scene_folder in sorted(os.listdir(base_dir)):
            scene_path_full = os.path.join(base_dir, scene_folder)
            if os.path.isdir(scene_path_full):
                for file in os.listdir(scene_path_full):
                    if file.endswith('.glb'):
                        scenes.append(os.path.join(scene_path_full, file))
                        break
        
        if 0 < args.scene <= len(scenes):
            scene_path = scenes[args.scene - 1]
        else:
            print(f"❌ 无效的场景编号: {args.scene}")
            return
    else:
        # 默认使用第一个场景
        base_dir = "/home/tao_h/VLMnav/data/scene_datasets/hm3d/minival"
        for scene_folder in sorted(os.listdir(base_dir)):
            scene_path_full = os.path.join(base_dir, scene_folder)
            if os.path.isdir(scene_path_full):
                for file in os.listdir(scene_path_full):
                    if file.endswith('.glb'):
                        scene_path = os.path.join(scene_path_full, file)
                        print(f"使用默认场景: {scene_folder}")
                        break
                break
        else:
            print("❌ 没有找到可用的场景")
            return
    
    # 解析全景图分辨率
    try:
        w, h = map(int, args.pano_res.split('x'))
        pano_res = (h, w)
    except:
        print(f"❌ 无效的分辨率格式: {args.pano_res}，使用默认值 1024x512")
        pano_res = (512, 1024)
    
    # 创建全景可视化
    create_panorama_viewer(scene_path, args.output, pano_res, args.cube_res)


if __name__ == "__main__":
    main()
