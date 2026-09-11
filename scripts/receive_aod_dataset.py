#!/usr/bin/env python3
"""
AOD 数据集接收端脚本（Ubuntu 端）
使用方法：
1. 在 Ubuntu 上运行此脚本启动服务器
2. 在 Windows 上使用 send_aod_data.py 发送数据
3. 或直接使用 rsync/scp 传输
"""

import os
import sys
import json
from pathlib import Path

def create_directory_structure():
    """创建 AOD 数据集目录结构"""
    base_dir = Path("/home/tao_h/VLMnav/datasets/AOD_dataset")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建示例场景目录
    sample_scenes = ["Home_001_1", "Home_001_2"]
    
    for scene in sample_scenes:
        scene_dir = base_dir / scene
        (scene_dir / "AVDB").mkdir(parents=True, exist_ok=True)
        (scene_dir / "jpg_rgb").mkdir(parents=True, exist_ok=True)
        (scene_dir / "high_res_depth").mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Directory structure created at: {base_dir}")
    return base_dir

def verify_dataset_integrity(dataset_path):
    """验证数据集完整性"""
    path = Path(dataset_path)
    
    required_files = [
        "annotations.json",
        "image_structs.mat",
        "present_instance_names.txt",
        "AVDB/AOS_initial_positions.json",
        "AVDB/object_locations.json",
        "AVDB/image_positions.json",
        "AVDB/shortest_path_lengths.json"
    ]
    
    print("\n验证数据集完整性...")
    missing_files = []
    
    for scene_dir in path.iterdir():
        if scene_dir.is_dir() and scene_dir.name.startswith("Home_"):
            print(f"\n检查场景：{scene_dir.name}")
            for req_file in required_files:
                file_path = scene_dir / req_file
                if file_path.exists():
                    print(f"  ✓ {req_file}")
                else:
                    print(f"  ✗ {req_file} (缺失)")
                    missing_files.append(f"{scene_dir.name}/{req_file}")
    
    if missing_files:
        print(f"\n⚠️  发现 {len(missing_files)} 个缺失文件:")
        for f in missing_files[:10]:  # 只显示前 10 个
            print(f"  - {f}")
        return False
    else:
        print("\n✅ 数据集完整性验证通过!")
        return True

def count_images(dataset_path):
    """统计图像数量"""
    path = Path(dataset_path)
    rgb_count = 0
    depth_count = 0
    
    for scene_dir in path.iterdir():
        if scene_dir.is_dir():
            rgb_count += len(list((scene_dir / "jpg_rgb").glob("*.jpg")))
            depth_count += len(list((scene_dir / "high_res_depth").glob("*.png")))
    
    print(f"\n图像统计:")
    print(f"  RGB 图像：{rgb_count} 张")
    print(f"  深度图像：{depth_count} 张")
    print(f"  总计：{rgb_count + depth_count} 张")

def main():
    print("=" * 60)
    print("AOD 数据集接收与验证工具 (Ubuntu 端)")
    print("=" * 60)
    
    dataset_path = Path("/home/tao_h/VLMnav/datasets/AOD_dataset")
    
    # 检查数据集是否已存在
    if dataset_path.exists():
        print(f"\n✓ 检测到数据集目录：{dataset_path}")
        
        # 验证完整性
        if verify_dataset_integrity(dataset_path):
            count_images(dataset_path)
            print("\n✅ 数据集已就绪，可以开始适配工作!")
            return 0
        else:
            print("\n⚠️  数据集不完整，请继续传输或检查源文件")
            return 1
    else:
        print(f"\n✗ 数据集目录不存在：{dataset_path}")
        print("\n请按照以下步骤传输数据:")
        print("\n方法 1: 使用 SCP/SFTP")
        print("  在 Windows 上使用 WinSCP 或 FileZilla")
        print("  连接到 Ubuntu，传输 D:\\aod\\AOD_dataset 到 /home/tao_h/VLMnav/datasets/")
        
        print("\n方法 2: 使用网络共享")
        print("  1. 在 Windows 上共享 D:\\aod 文件夹")
        print("  2. 在 Ubuntu 上挂载:")
        print("     sudo mount -t cifs //Windows_IP/aod /mnt/windows_aod")
        print("  3. 复制数据:")
        print("     cp -r /mnt/windows_aod/AOD_dataset /home/tao_h/VLMnav/datasets/")
        
        print("\n方法 3: 使用压缩传输")
        print("  1. 在 Windows 上压缩:")
        print("     Compress-Archive -Path 'D:\\aod\\AOD_dataset' -DestinationPath 'D:\\aod\\AOD.zip'")
        print("  2. 复制到 Ubuntu")
        print("  3. 在 Ubuntu 上解压:")
        print("     unzip AOD.zip -d /home/tao_h/VLMnav/datasets/")
        
        # 创建目录结构
        create_directory_structure()
        return 1

if __name__ == "__main__":
    sys.exit(main())
