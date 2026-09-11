#!/usr/bin/env python3
"""
AOD 数据集一键传输助手（Ubuntu 端）
自动检测可用的传输方法并指导用户完成传输
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil

class AODTransferAssistant:
    def __init__(self):
        self.source_windows = "D:\\aod\\AOD_dataset"
        self.target_ubuntu = Path("/home/tao_h/VLMnav/datasets/AOD_dataset")
        self.temp_dir = Path("/tmp/aod_transfer")
        
    def check_prerequisites(self):
        """检查前置条件"""
        print("=" * 60)
        print("AOD 数据集传输助手 - 环境检查")
        print("=" * 60)
        
        checks = {
            "目标目录可写": self._check_writable(),
            "磁盘空间充足": self._check_disk_space(),
            "网络连通性": self._check_network(),
        }
        
        all_ok = True
        for check, result in checks.items():
            status = "✓" if result else "✗"
            print(f"{status} {check}")
            if not result:
                all_ok = False
        
        return all_ok
    
    def _check_writable(self):
        try:
            self.target_ubuntu.parent.mkdir(parents=True, exist_ok=True)
            test_file = self.target_ubuntu.parent / ".write_test"
            test_file.touch()
            test_file.unlink()
            return True
        except:
            return False
    
    def _check_disk_space(self):
        try:
            # AOD 数据集估计需要 5GB+
            required_gb = 5
            stat = shutil.disk_usage("/")
            available_gb = stat.free / (1024**3)
            print(f"   可用磁盘空间：{available_gb:.1f} GB (需要：{required_gb} GB)")
            return available_gb >= required_gb
        except:
            return True  # 无法检查时假设足够
    
    def _check_network(self):
        try:
            # 检查是否能访问网络（用于网络共享）
            subprocess.run(["ping", "-c", "1", "8.8.8.8"], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL,
                         timeout=2)
            return True
        except:
            return False
    
    def show_transfer_methods(self):
        """显示传输方法"""
        print("\n" + "=" * 60)
        print("可选择的传输方法")
        print("=" * 60)
        
        methods = [
            {
                "name": "方法 1: SCP/SFTP (推荐)",
                "desc": "使用 WinSCP 或 FileZilla 图形化工具",
                "difficulty": "简单",
                "speed": "中等",
                "steps": [
                    "1. Windows 上下载 WinSCP (https://winscp.net/)",
                    "2. 连接到 Ubuntu: ssh://tao_h@你的 Ubuntu IP",
                    "3. 拖拽 D:\\aod\\AOD_dataset 到 /home/tao_h/VLMnav/datasets/",
                    "4. 等待传输完成（约 15-30 分钟）"
                ]
            },
            {
                "name": "方法 2: WSL2 直接复制",
                "desc": "如果使用 WSL2，可以直接访问 Windows 文件",
                "difficulty": "简单",
                "speed": "快",
                "steps": [
                    "1. 在 Ubuntu (WSL) 中执行:",
                    "   cp -r /mnt/d/aod/AOD_dataset /home/tao_h/VLMnav/datasets/",
                    "2. 或使用压缩加速:",
                    "   cd /mnt/d/aod && zip -r AOD.zip AOD_dataset",
                    "   mv AOD.zip /home/tao_h/VLMnav/datasets/",
                    "   cd /home/tao_h/VLMnav/datasets/ && unzip AOD.zip"
                ]
            },
            {
                "name": "方法 3: 网络共享挂载",
                "desc": "挂载 Windows 共享文件夹到 Ubuntu",
                "difficulty": "中等",
                "speed": "中等",
                "steps": [
                    "1. Windows 上共享 D:\\aod 文件夹",
                    "2. Ubuntu 安装 CIFS: sudo apt install cifs-utils",
                    "3. 挂载：sudo mount -t cifs //Windows_IP/aod /mnt/windows_aod",
                    "4. 复制：cp -r /mnt/windows_aod/AOD_dataset /home/tao_h/VLMnav/datasets/"
                ]
            },
            {
                "name": "方法 4: 外部存储设备",
                "desc": "使用 U 盘或移动硬盘",
                "difficulty": "简单",
                "speed": "快",
                "steps": [
                    "1. Windows 上复制到 U 盘",
                    "2. Ubuntu 上挂载 U 盘",
                    "3. 复制到目标目录"
                ]
            }
        ]
        
        for i, method in enumerate(methods, 1):
            print(f"\n{i}. {method['name']}")
            print(f"   难度：{method['difficulty']} | 速度：{method['speed']}")
            print(f"   {method['desc']}")
            print("   步骤:")
            for step in method['steps']:
                print(f"     {step}")
    
    def create_target_directory(self):
        """创建目标目录结构"""
        print("\n" + "=" * 60)
        print("创建目标目录")
        print("=" * 60)
        
        try:
            self.target_ubuntu.mkdir(parents=True, exist_ok=True)
            print(f"✓ 已创建目录：{self.target_ubuntu}")
            
            # 创建子目录结构
            subdirs = ["AVDB", "jpg_rgb", "high_res_depth"]
            for subdir in subdirs:
                (self.target_ubuntu / "Home_001_1" / subdir).mkdir(parents=True, exist_ok=True)
                (self.target_ubuntu / "Home_001_2" / subdir).mkdir(parents=True, exist_ok=True)
            
            print("✓ 已创建示例场景目录结构")
            return True
            
        except Exception as e:
            print(f"✗ 创建目录失败：{e}")
            return False
    
    def verify_transfer(self):
        """验证传输结果"""
        print("\n" + "=" * 60)
        print("验证数据传输")
        print("=" * 60)
        
        if not self.target_ubuntu.exists():
            print("✗ 数据集目录不存在")
            return False
        
        # 统计文件
        total_files = 0
        total_size = 0
        scenes = []
        
        for item in self.target_ubuntu.rglob("*"):
            if item.is_file():
                total_files += 1
                total_size += item.stat().st_size
        
        for scene_dir in self.target_ubuntu.iterdir():
            if scene_dir.is_dir() and scene_dir.name.startswith("Home_"):
                scenes.append(scene_dir.name)
        
        print(f"✓ 检测到 {len(scenes)} 个场景:")
        for scene in scenes:
            print(f"  - {scene}")
        
        print(f"\n✓ 总计：{total_files} 个文件，{total_size / (1024**3):.2f} GB")
        
        # 检查关键文件
        print("\n检查关键文件...")
        sample_scene = self.target_ubuntu / scenes[0] if scenes else self.target_ubuntu
        
        key_files = [
            "annotations.json",
            "AVDB/AOS_initial_positions.json",
            "AVDB/object_locations.json",
            "AVDB/image_positions.json"
        ]
        
        missing = []
        for key_file in key_files:
            file_path = sample_scene / key_file
            if file_path.exists():
                print(f"  ✓ {key_file}")
            else:
                print(f"  ✗ {key_file} (缺失)")
                missing.append(key_file)
        
        if missing:
            print(f"\n⚠️  发现 {len(missing)} 个缺失文件，请检查传输是否完整")
            return False
        else:
            print("\n✅ 数据传输验证通过!")
            return True
    
    def run(self):
        """主流程"""
        print("\n🚀 " + "=" * 58)
        print("   AOD 数据集传输助手 (Windows → Ubuntu)")
        print("=" * 60 + "\n")
        
        # 检查环境
        if not self.check_prerequisites():
            print("\n⚠️  环境检查未完全通过，但可以继续尝试")
        
        # 显示传输方法
        self.show_transfer_methods()
        
        # 创建目录
        if self.create_target_directory():
            print("\n" + "=" * 60)
            print("下一步操作")
            print("=" * 60)
            print("\n请选择一种传输方法并按照说明操作。")
            print("\n传输完成后，运行以下命令验证:")
            print(f"  cd /home/tao_h/VLMnav")
            print(f"  python scripts/receive_aod_dataset.py")
            print("\n祝你顺利！如有问题请参考 AOD 数据集传输指南.md")
        else:
            print("\n✗ 无法创建目录，请检查权限和磁盘空间")
            return 1
        
        return 0

def main():
    assistant = AODTransferAssistant()
    sys.exit(assistant.run())

if __name__ == "__main__":
    main()
