#!/bin/bash

# AOD 数据集传输脚本 - Windows 到 Ubuntu
# 使用方法：
# 1. 在 Windows 上运行此脚本生成压缩包
# 2. 将压缩包复制到 Ubuntu
# 3. 在 Ubuntu 上解压

echo "=========================================="
echo "AOD 数据集传输助手"
echo "=========================================="
echo ""

# 创建压缩包的 Windows 批处理命令
cat > compress_aod.bat << 'EOF'
@echo off
echo Compressing AOD dataset...
echo This may take several minutes due to large file sizes.
echo.

REM 设置源目录和目标目录
set SOURCE_DIR=D:\aod\AOD_dataset
set OUTPUT_FILE=D:\aod\AOD_dataset_transfer.zip

echo Source: %SOURCE_DIR%
echo Output: %OUTPUT_FILE%
echo.

REM 使用 PowerShell 进行高效压缩
powershell -Command "& { 
    Compress-Archive -Path '%SOURCE_DIR%\*' -DestinationPath '%OUTPUT_FILE%' -Force -CompressionLevel Optimal 
}"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ Compression completed successfully!
    echo Output file: %OUTPUT_FILE%
    echo.
    echo Next steps:
    echo 1. Copy the zip file to your Ubuntu system
    echo 2. Extract it to: /home/tao_h/VLMnav/datasets/AOD_dataset
) else (
    echo.
    echo ✗ Compression failed. Please check if the source directory exists.
    pause
)
EOF

echo "已创建 Windows 压缩脚本：compress_aod.bat"
echo ""
echo "=========================================="
echo "操作步骤："
echo "=========================================="
echo ""
echo "步骤 1: 在 Windows 系统上运行"
echo "  1. 将 compress_aod.bat 复制到 D:\\aod\\ 文件夹"
echo "  2. 双击运行 compress_aod.bat"
echo "  3. 等待压缩完成（约 5-10 分钟）"
echo ""
echo "步骤 2: 传输到 Ubuntu"
echo "  方法 A - SCP/SFTP:"
echo "    在 Ubuntu 终端执行："
echo "    scp /mnt/d/aod/AOD_dataset_transfer.zip tao_h@localhost:/home/tao_h/VLMnav/datasets/"
echo ""
echo "  方法 B - 网络共享:"
echo "    sudo mount -t cifs //Windows_IP/aod /mnt/windows_aod"
echo "    cp /mnt/windows_aod/AOD_dataset_transfer.zip /home/tao_h/VLMnav/datasets/"
echo ""
echo "  方法 C - 外部存储:"
echo "    使用 U 盘或移动硬盘复制文件"
echo ""
echo "步骤 3: 在 Ubuntu 上解压"
echo "  cd /home/tao_h/VLMnav/datasets/"
echo "  unzip AOD_dataset_transfer.zip"
echo "  mv AOD_dataset AOD_dataset_temp"
echo "  mkdir AOD_dataset"
echo "  mv AOD_dataset_temp/* AOD_dataset/"
echo ""
echo "=========================================="
