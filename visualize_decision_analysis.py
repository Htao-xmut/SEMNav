#!/usr/bin/env python3
"""
生成决策分析可视化图表
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

def extract_step_data(log_dir: Path):
    """提取步骤数据"""
    # 尝试多种路径模式
    episode_dirs = []
    for pattern in ["*/0_of_1/*", "0_of_1/*"]:
        found = list(log_dir.glob(pattern))
        if found:
            episode_dirs.extend([d for d in found if d.is_dir()])
    
    if not episode_dirs:
        raise ValueError(f"未找到 episode 目录在 {log_dir}")
    
    episode_dir = episode_dirs[0]
    
    steps = []
    for i in range(10):
        step_dir = episode_dir / f"step{i}"
        if not step_dir.exists():
            continue
        
        details_file = step_dir / "details.txt"
        with open(details_file, 'r') as f:
            content = f.read()
        
        # 提取箭头数
        arrow_match = re.search(r"There are (\d+) red arrow", content)
        num_arrows = int(arrow_match.group(1)) if arrow_match else 0
        
        # 提取执行的动作
        action_match = re.search(r"ACTION_NUMBER\n(\d+)", content)
        action = int(action_match.group(1)) if action_match else -1
        
        # 检查响应格式
        response_section = content.split("RESPONSE\n")[1].split("\nSTOPPING")[0].strip() if "RESPONSE" in content else ""
        
        if response_section.startswith("{"):
            format_type = "JSON"
        elif "'" in response_section and "action" in response_section:
            format_type = "Python Dict"
        else:
            format_type = "Natural Language"
        
        steps.append({
            'step': i,
            'num_arrows': num_arrows,
            'action': action,
            'format': format_type,
            'is_valid': 0 <= action < num_arrows if num_arrows > 0 else False
        })
    
    return steps

def create_visualization(steps, output_file='decision_analysis.png'):
    """创建可视化图表"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('VLM Navigation Decision Analysis', fontsize=16, fontweight='bold')
    
    # 数据准备
    step_nums = [s['step'] for s in steps]
    num_arrows = [s['num_arrows'] for s in steps]
    actions = [s['action'] for s in steps]
    validity = [1 if s['is_valid'] else 0 for s in steps]
    
    # 颜色映射（按格式类型）
    color_map = {'JSON': 'green', 'Python Dict': 'orange', 'Natural Language': 'red'}
    colors = [color_map[s['format']] for s in steps]
    
    # 图1: 箭头数量 vs 选择的动作
    ax1 = axes[0, 0]
    ax1.bar(step_nums, num_arrows, alpha=0.5, label='Available Actions', color='skyblue')
    ax1.scatter(step_nums, actions, color=colors, s=100, zorder=5, label='Selected Action', edgecolors='black', linewidth=1.5)
    ax1.set_xlabel('Step Number', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Available vs Selected Actions', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticks(step_nums)
    
    # 图2: 动作有效性
    ax2 = axes[0, 1]
    valid_steps = [i for i, v in enumerate(validity) if v == 1]
    invalid_steps = [i for i, v in enumerate(validity) if v == 0]
    
    if valid_steps:
        ax2.scatter([steps[i]['step'] for i in valid_steps], [actions[i] for i in valid_steps], 
                   color='green', s=150, marker='✓', label='Valid', edgecolors='black', linewidth=2)
    if invalid_steps:
        ax2.scatter([steps[i]['step'] for i in invalid_steps], [actions[i] for i in invalid_steps], 
                   color='red', s=150, marker='✗', label='Invalid', edgecolors='black', linewidth=2)
    
    ax2.set_xlabel('Step Number', fontsize=12)
    ax2.set_ylabel('Action Number', fontsize=12)
    ax2.set_title('Action Validity Check', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(step_nums)
    
    # 图3: 响应格式分布
    ax3 = axes[1, 0]
    format_counts = {}
    for s in steps:
        fmt = s['format']
        format_counts[fmt] = format_counts.get(fmt, 0) + 1
    
    wedges, texts, autotexts = ax3.pie(format_counts.values(), labels=format_counts.keys(), 
                                        autopct='%1.1f%%', colors=['green', 'orange', 'red'])
    ax3.set_title('Response Format Distribution', fontsize=13, fontweight='bold')
    
    # 图4: 时间线趋势
    ax4 = axes[1, 1]
    ax4.plot(step_nums, num_arrows, 'o-', label='Available Actions', linewidth=2, markersize=8)
    ax4.plot(step_nums, actions, 's--', label='Selected Action', linewidth=2, markersize=8, color='red')
    ax4.fill_between(step_nums, 0, [a-1 if a > 0 else 0 for a in num_arrows], alpha=0.2, color='green', label='Valid Range')
    ax4.set_xlabel('Step Number', fontsize=12)
    ax4.set_ylabel('Action Number', fontsize=12)
    ax4.set_title('Decision Timeline', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(step_nums)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 可视化图表已保存至: {output_file}")
    plt.close()

def main():
    """主函数"""
    log_base = Path('/home/tao_h/VLMnav/logs')
    latest_log = sorted(log_base.glob('ObjectNav_default_*'))[-1]
    
    print(f"📂 分析日志: {latest_log.name}")
    
    steps = extract_step_data(latest_log)
    print(f"📊 提取了 {len(steps)} 个步骤的数据")
    
    create_visualization(steps, output_file=latest_log / 'decision_analysis.png')
    
    # 打印统计摘要
    print("\n📈 统计摘要:")
    print(f"  - 总步数: {len(steps)}")
    print(f"  - 有效动作: {sum(1 for s in steps if s['is_valid'])}")
    print(f"  - 无效动作: {sum(1 for s in steps if not s['is_valid'])}")
    print(f"  - 有效率: {sum(1 for s in steps if s['is_valid']) / len(steps) * 100:.1f}%")
    
    format_stats = {}
    for s in steps:
        fmt = s['format']
        format_stats[fmt] = format_stats.get(fmt, 0) + 1
    
    print(f"\n📝 格式分布:")
    for fmt, count in format_stats.items():
        print(f"  - {fmt}: {count} ({count/len(steps)*100:.1f}%)")

if __name__ == "__main__":
    main()
