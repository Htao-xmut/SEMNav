#!/bin/bash
# VLMnav 评估进度监控脚本

echo "============================================================"
echo "📊 VLMnav 评估进度监控"
echo "============================================================"
echo ""

# 查找最新的日志目录
LATEST_LOG=$(ls -td /home/tao_h/VLMnav/logs/ObjectNav_* 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ 未找到任何运行日志"
    exit 1
fi

echo "📁 最新运行: $(basename $LATEST_LOG)"
echo ""

# 检查 episode 完成情况
EPISODE_DIR="$LATEST_LOG/0_of_1"
if [ -d "$EPISODE_DIR" ]; then
    TOTAL_EPISODES=$(ls -d $EPISODE_DIR/*/ 2>/dev/null | wc -l)
    COMPLETED_EPISODES=$(ls $EPISODE_DIR/*/animation.gif 2>/dev/null | wc -l)
    
    echo "📈 Episode 进度:"
    echo "   已完成: $COMPLETED_EPISODES"
    echo "   总计: $TOTAL_EPISODES (正在运行)"
    echo ""
    
    # 列出已完成的 episodes
    echo "✅ 已完成的 Episodes:"
    for episode_dir in $EPISODE_DIR/*/; do
        if [ -f "$episode_dir/animation.gif" ]; then
            episode_name=$(basename $episode_dir)
            steps=$(ls -d $episode_dir/step* 2>/dev/null | wc -l)
            errors=$(ls -d $episode_dir/*_ERROR 2>/dev/null | wc -l)
            echo "   - $episode_name: $steps steps ($errors errors)"
        fi
    done
else
    echo "⏳ 实验尚未开始或目录结构不同"
fi

echo ""
echo "💡 提示:"
echo "   - 实时查看: watch -n 5 bash monitor_progress.sh"
echo "   - 查看 GIF: xdg-open $EPISODE_DIR/*/animation.gif"
echo ""
