#!/bin/bash
# 下载 Habitat 测试场景脚本

set -e

echo "🚀 下载 Habitat 测试场景..."
echo "================================"

cd /home/tao_h/VLMnav

# 使用完整的 Python 路径
PYTHON="/home/tao_h/miniconda3/envs/vlm_nav/bin/python"

# 创建目录
mkdir -p data/scene_datasets

# 使用正确的参数名（--data-path 而不是 --data_path）
echo "正在下载 habitat_test_scenes..."
$PYTHON -m habitat_sim.utils.datasets_download \
  --uids habitat_test_scenes \
  --data-path data/scene_datasets

echo ""
echo "✅ 下载完成！"
echo ""
echo "验证下载结果..."
ls -lh data/scene_datasets/habitat-test-scenes/ 2>&1 || echo "检查目录结构..."
find data/scene_datasets -type f -name "*.glb" | head -5