#!/bin/bash
# VLMnav 小规模评估脚本
# 运行 5 个 episodes 进行验证

set -e  # 遇到错误立即退出

echo "============================================================"
echo "🚀 VLMnav 小规模评估 (5 episodes)"
echo "============================================================"
echo ""

# 激活环境
echo "📦 激活 conda 环境..."
source /home/tao_h/miniconda3/bin/activate vlm_nav

# 检查 Ollama 服务
echo "🔍 检查 Ollama 服务..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama 服务正常运行"
else
    echo "❌ Ollama 服务未运行，请先启动: ollama serve"
    exit 1
fi

# 检查模型
echo "🔍 检查 LLaVA 7B 模型..."
if curl -s http://localhost:11434/api/tags | grep -q "llava:7b"; then
    echo "✅ LLaVA 7B 模型已加载"
else
    echo "⚠️  LLaVA 7B 模型未找到，正在下载..."
    ollama pull llava:7b
fi

echo ""
echo "============================================================"
echo "🎯 开始评估..."
echo "============================================================"
echo ""

# 设置环境变量（确保配置文件中的占位符被正确解析）
export OLLAMA_MODEL="llava:7b"

# 运行评估
cd /home/tao_h/VLMnav
python scripts/main.py --config ObjectNav --num_episodes 5

echo ""
echo "============================================================"
echo "✅ 评估完成！"
echo "============================================================"
echo ""
echo "📊 查看结果："
echo "   - 日志目录: ls -lh logs/ObjectNav_*/"
echo "   - GIF 动画: find logs/ -name 'animation.gif' | head -5"
echo ""
