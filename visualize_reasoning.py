#!/usr/bin/env python3
"""
可视化 Reasoning 与动作选择的对应关系
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import re
import textwrap

def extract_reasoning_data(log_dir: Path):
    """提取 reasoning 数据"""
    episode_dirs = []
    for pattern in ["*/0_of_1/*", "0_of_1/*"]:
        found = list(log_dir.glob(pattern))
        if found:
            episode_dirs.extend([d for d in found if d.is_dir()])
    
    if not episode_dirs:
        raise ValueError(f"未找到 episode 目录")
    
    episode_dir = episode_dirs[0]
    steps_data = []
    
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
        
        # 提取动作
        action_match = re.search(r"ACTION_NUMBER\n(\d+)", content)
        action = int(action_match.group(1)) if action_match else -1
        
        # 提取 reasoning
        reasoning = ""
        if "REASONING\n" in content:
            reasoning_section = content.split("REASONING\n")[1].split("\nSTOPPING")[0].strip()
            reasoning = reasoning_section[:150]  # 截断到150字符
        
        # 有效性检查
        is_valid = 0 <= action < num_arrows if num_arrows > 0 else False
        
        steps_data.append({
            'step': i,
            'num_arrows': num_arrows,
            'action': action,
            'reasoning': reasoning,
            'is_valid': is_valid
        })
    
    return steps_data

def create_reasoning_visualization(steps_data, output_file='reasoning_analysis.png'):
    """创建 Reasoning 分析可视化"""
    fig, axes = plt.subplots(4, 1, figsize=(16, 20))
    fig.suptitle('VLM Navigation Decision Analysis with Reasoning', fontsize=18, fontweight='bold')
    
    step_nums = [s['step'] for s in steps_data]
    actions = [s['action'] for s in steps_data]
    validity = [1 if s['is_valid'] else 0 for s in steps_data]
    
    # 图1: 动作选择时间线
    ax1 = axes[0]
    colors = ['green' if v else 'red' for v in validity]
    ax1.bar(step_nums, actions, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, label='Turn Around (Action 0)')
    ax1.set_xlabel('Step Number', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Selected Action', fontsize=14, fontweight='bold')
    ax1.set_title('Action Selection Timeline (Green=Valid, Red=Invalid)', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticks(step_nums)
    
    # 添加数值标签
    for i, (step, action, valid) in enumerate(zip(step_nums, actions, validity)):
        status = "✓" if valid else "✗"
        ax1.text(step, action + 0.2, f"{action} {status}", ha='center', va='bottom', 
                fontsize=11, fontweight='bold', color='darkgreen' if valid else 'darkred')
    
    # 图2-4: 每个步骤的 Reasoning（分三组显示）
    reasoning_groups = [steps_data[0:3], steps_data[3:6], steps_data[6:10]]
    group_titles = ['Steps 0-2: Initial Exploration', 'Steps 3-5: Search Phase', 'Steps 6-9: Loop Pattern']
    
    for idx, (group, title) in enumerate(zip(reasoning_groups, group_titles)):
        ax = axes[idx + 1]
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', loc='left', pad=20)
        
        y_position = 0.95
        for step_info in group:
            step_num = step_info['step']
            action = step_info['action']
            reasoning = step_info['reasoning']
            is_valid = step_info['is_valid']
            
            # 步骤标题
            status_icon = "✅" if is_valid else "❌"
            title_text = f"Step {step_num}: Action {action} {status_icon}"
            ax.text(0.05, y_position, title_text, fontsize=12, fontweight='bold', 
                   transform=ax.transAxes, color='darkgreen' if is_valid else 'darkred')
            y_position -= 0.08
            
            # Reasoning 文本（自动换行）
            wrapped_reasoning = textwrap.fill(reasoning, width=120)
            ax.text(0.05, y_position, wrapped_reasoning, fontsize=10, 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow' if is_valid else 'lightcoral', alpha=0.3))
            
            # 计算文本高度并调整位置
            num_lines = len(wrapped_reasoning.split('\n'))
            y_position -= (num_lines * 0.04 + 0.08)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Reasoning 可视化已保存至: {output_file}")
    plt.close()

def main():
    """主函数"""
    log_base = Path('/home/tao_h/VLMnav/logs')
    latest_log = sorted(log_base.glob('ObjectNav_default_*'))[-1]
    
    print(f"📂 分析日志: {latest_log.name}")
    
    steps_data = extract_reasoning_data(latest_log)
    print(f"📊 提取了 {len(steps_data)} 个步骤的 Reasoning 数据")
    
    create_reasoning_visualization(steps_data, output_file=latest_log / 'reasoning_analysis.png')
    
    # 打印统计摘要
    print("\n📈 Reasoning 质量统计:")
    print(f"  - 总步数: {len(steps_data)}")
    print(f"  - 有效动作: {sum(1 for s in steps_data if s['is_valid'])}")
    print(f"  - 无效动作: {sum(1 for s in steps_data if not s['is_valid'])}")
    print(f"  - Reasoning 提供率: {sum(1 for s in steps_data if s['reasoning']) / len(steps_data) * 100:.1f}%")
    
    # 检测循环模式
    actions = [s['action'] for s in steps_data]
    if len(actions) >= 5:
        last_5 = actions[-5:]
        if len(set(last_5)) == 1:
            print(f"\n⚠️  检测到循环模式: 最后 5 步都选择了 Action {last_5[0]}")

if __name__ == "__main__":
    main()
