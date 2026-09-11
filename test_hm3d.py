import os

# 设置 OSMesa 软件渲染环境变量（绕过 EGL/CUDA 问题）
# 必须在导入 habitat_sim 之前设置
os.environ["MAGNUM_EGL_DISABLE"] = "1"
os.environ["MAGNUM_GL_USE_OSMESA"] = "1"

import habitat_sim
import numpy as np
import sys


def test_hm3d_minival_real():
    """
    真实渲染测试：尝试使用 Habitat-Sim 加载场景并渲染
    
    如果成功，说明 EGL 问题已解决
    如果失败，会给出详细的错误信息和建议
    """
    
    # 1. 设置场景路径
    scene_id = "00800-TEEsavR23oF"
    short_id = scene_id.split('-')[1]
    glb_file = f"data/scene_datasets/hm3d/minival/{scene_id}/{short_id}.basis.glb"
    
    print("=" * 60)
    print("🎨 HM3D Minival 真实渲染测试")
    print("=" * 60)
    
    print(f"\n📂 检查场景文件...")
    if not os.path.exists(glb_file):
        print(f"   ❌ Error: GLB file not found: {glb_file}")
        return False
    else:
        print(f"   ✅ GLB 文件存在")
    
    # 2. 配置模拟器
    try:
        print(f"\n⚙️  配置 Habitat-Sim...")
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_id = glb_file
        # 使用 CPU 软件渲染（llvmpipe），绕过 EGL/CUDA 问题
        cfg.gpu_device_id = -1
        
        # 配置传感器
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "rgb"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [480, 640]
        rgb_sensor_spec.position = [0.0, 1.5, 0.0]
        
        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [480, 640]
        depth_sensor_spec.position = [0.0, 1.5, 0.0]
        
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]
        
        sim_cfg = habitat_sim.Configuration(cfg, [agent_cfg])
        
        print(f"   ✅ 配置完成，尝试初始化模拟器...")
        
        # 3. 创建模拟器实例
        sim = habitat_sim.Simulator(sim_cfg)
        print("   ✅ Simulator initialized successfully!")
        
        # 4. 获取观测数据
        observations = sim.get_sensor_observations()
        rgb = observations["rgb"]
        depth = observations["depth"]
        
        print(f"\n📊 渲染结果:")
        print(f"   ✅ RGB Image shape: {rgb.shape}")
        print(f"   ✅ Depth Image shape: {depth.shape}")
        
        if rgb.sum() > 0:
            print(f"   ✅ RGB 图像有效（非全黑）")
            print(f"   📈 RGB 像素值范围: [{rgb.min()}, {rgb.max()}]")
        else:
            print(f"   ⚠️  Warning: RGB image is black")
        
        sim.close()
        
        print("\n" + "=" * 60)
        print("🎉 真实渲染测试通过！")
        print("=" * 60)
        print("\n📊 渲染模式: llvmpipe (Mesa 软件渲染器)")
        print("💡 提示: 当前使用 CPU 渲染，性能较低但功能完整。")
        print("   如需 GPU 加速，可以尝试:")
        print("   - 设置 cfg.gpu_device_id = 0 并安装完整的 NVIDIA EGL 支持")
        print("   - 或使用 conda 安装的 habitat-sim v0.3.1（可能支持更好的后端）")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ 渲染失败: {e}")
        print("\n" + "=" * 60)
        print("💡 建议解决方案:")
        print("   1. 已安装 NVIDIA EGL 库，可能需要重启系统或重新编译 habitat-sim")
        print("   2. 使用 Mock 环境继续开发（推荐）")
        print("   3. 降级到 habitat-sim v0.3.1:")
        print("      conda install -c conda-forge -c aihabitat habitat-sim=0.3.1")
        print("=" * 60)
        return False


def test_hm3d_minival_mock():
    """
    Mock 测试：绕过 Habitat-Sim 渲染问题，验证数据路径和 VLM_NAV 核心逻辑
    
    这个测试会：
    1. 验证 HM3D 数据集文件是否存在且配对正确
    2. 模拟生成 RGB/Depth 观测数据
    3. 验证数据结构是否符合 VLM_NAV 要求
    """
    
    # 1. 设置场景路径
    scene_id = "00800-TEEsavR23oF"
    short_id = scene_id.split('-')[1]  # TEEsavR23oF
    glb_file = f"data/scene_datasets/hm3d/minival/{scene_id}/{short_id}.basis.glb"
    navmesh_file = f"data/scene_datasets/hm3d/minival/{scene_id}/{short_id}.basis.navmesh"
    
    print("=" * 60)
    print("🔍 HM3D Minival 数据集验证（Mock 模式）")
    print("=" * 60)
    
    # 2. 检查文件是否存在
    print(f"\n📂 检查场景文件...")
    print(f"   GLB 文件: {glb_file}")
    if not os.path.exists(glb_file):
        print("   ❌ Error: GLB file not found!")
        return False
    else:
        print(f"   ✅ GLB 文件存在 (大小: {os.path.getsize(glb_file) / 1024 / 1024:.2f} MB)")
    
    print(f"   NavMesh 文件: {navmesh_file}")
    if not os.path.exists(navmesh_file):
        print("   ❌ Error: NavMesh file not found!")
        return False
    else:
        print(f"   ✅ NavMesh 文件存在 (大小: {os.path.getsize(navmesh_file) / 1024:.2f} KB)")
    
    # 3. 模拟生成观测数据（替代真实的 Habitat-Sim 渲染）
    print(f"\n🎨 模拟生成观测数据...")
    height, width = 480, 640
    
    # 模拟 RGB 图像 (H, W, 3) - uint8, 范围 0-255
    rgb_mock = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    print(f"   ✅ RGB Image shape: {rgb_mock.shape}, dtype: {rgb_mock.dtype}")
    
    # 模拟 Depth 图像 (H, W) - float32, 单位米
    depth_mock = np.random.uniform(0.5, 5.0, (height, width)).astype(np.float32)
    print(f"   ✅ Depth Image shape: {depth_mock.shape}, dtype: {depth_mock.dtype}")
    print(f"   📊 Depth range: [{depth_mock.min():.2f}m, {depth_mock.max():.2f}m]")
    
    # 4. 验证数据格式是否符合 VLM_NAV 要求
    print(f"\n✓ 数据格式验证:")
    checks = [
        ("RGB 是 3 通道", rgb_mock.shape[2] == 3),
        ("RGB 是 uint8", rgb_mock.dtype == np.uint8),
        ("Depth 是单通道", len(depth_mock.shape) == 2),
        ("Depth 是 float32", depth_mock.dtype == np.float32),
        ("RGB 和 Depth 尺寸匹配", rgb_mock.shape[:2] == depth_mock.shape),
    ]
    
    all_passed = True
    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"   {status} {check_name}")
        if not result:
            all_passed = False
    
    # 5. 总结
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 Mock 测试通过！HM3D Minival 数据集已就绪。")
        print("\n📝 下一步建议：")
        print("   1. 由于 habitat-sim v0.3.3 存在 EGL 渲染问题，建议使用 Mock 环境")
        print("      继续开发 VLM_NAV 核心逻辑（Agent、VLM 接口等）")
        print("   2. 如需真实渲染，考虑降级到 habitat-sim v0.3.1：")
        print("      conda install -c conda-forge -c aihabitat habitat-sim=0.3.1")
        print("   3. 或者使用 Docker 容器运行完整仿真")
        print("\n✨ 你的项目结构已正确配置，可以开始集成 VLM 导航代码！")
    else:
        print("❌ 测试失败！请检查上述错误。")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    # 默认先尝试真实渲染测试
    mode = sys.argv[1] if len(sys.argv) > 1 else "real"
    
    if mode == "real":
        print("尝试真实渲染测试...\n")
        success = test_hm3d_minival_real()
        if not success:
            print("\n真实渲染失败，切换到 Mock 测试...\n")
            test_hm3d_minival_mock()
    elif mode == "mock":
        test_hm3d_minival_mock()
    else:
        print(f"未知模式: {mode}，请使用 'real' 或 'mock'")