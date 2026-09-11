#!/bin/bash
# 启动 Qwen-VL 评估的脚本
# API key 不再明文写死: 优先用已有环境变量, 其次从 .env 加载 (复制 .env.example 为 .env 并填入)

if [ -z "$QWEN_API_KEY" ] && [ -f "$(dirname "$0")/.env" ]; then
  set -a; source "$(dirname "$0")/.env"; set +a
fi
: "${QWEN_API_KEY:?未设置 QWEN_API_KEY — 请复制 .env.example 为 .env 并填入, 或先 export}"

# 激活 Conda 环境
source /home/tao_h/miniconda3/bin/activate vlm_nav

# 设置 Python 路径
export PYTHONPATH=/home/tao_h/VLMnav/src:$PYTHONPATH

# 运行评估
echo "🚀 Starting evaluation with Qwen-VL..."
python scripts/main.py --config ObjectNav --num_episodes 5

echo "✅ Evaluation completed!"
