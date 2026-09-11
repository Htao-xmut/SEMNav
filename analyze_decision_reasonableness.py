#!/usr/bin/env python3
"""
分析 VLM 导航决策的合理性
对比模型选择的动作与基于环境观测的推荐动作
"""

import os
import sys
from pathlib import Path
import pandas as pd
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

def load_episode_data(log_dir):
    """加载 Episode 的数据"""
    episode_dir = Path(log_dir) / "0_of_1" / "0_00800-TEEsavR23oF"
    
    if not episode_dir.exists():
        print(f"❌ Episode 目录不存在: {episode_dir}")
        return None
    
    # 加载结果数据
    pkl_file = episode_dir / "df_results.pkl"
    if pkl_file.exists():
        with open(pkl_file, 'rb') as f:
            df = pickle.load(f)
    else:
        print("⚠️ 未找到 df_results.pkl")
        df = None
    
    # 收集每个步骤的信息
    steps_info = []
    for step_dir in sorted(episode_dir.glob("step*")):
        if not step_dir.is_dir():
            continue
        
        # 跳过错误目录
        if "ERROR" in step_dir.name:
            continue
        
        try:
            step_num = int(step_dir.name.replace("step", ""))
        except ValueError:
            continue
        
        details_file = step_dir / "details.txt"
        
        info = {
            'step': step_num,
            'dir': step_dir,
            'color_image': step_dir / "color_sensor.png",
            'chosen_image': step_dir / "color_sensor_chosen.png",
            'voxel_map': step_dir / "voxel_map.png",
        }
        
        # 解析 details.txt
        if details_file.exists():
            with open(details_file, 'r') as f:
                content = f.read()
                
            # 提取 ACTION_NUMBER
            if "ACTION_NUMBER" in content:
                action_line = content.split("ACTION_NUMBER\n")[1].split("\n")[0]
                try:
                    info['model_action'] = int(action_line)
                except:
                    info['model_action'] = -1
            
            # 提取 PROMPT 中的箭头数量
            if "There are" in content and "red arrows" in content:
                prompt_section = content.split("PROMPT\n")[1].split("\nRESPONSE")[0]
                import re
                match = re.search(r"There are (\d+) red arrows", prompt_section)
                if match:
                    info['num_arrows'] = int(match.group(1))
            
            # 提取 RESPONSE
            if "RESPONSE" in content:
                response_section = content.split("RESPONSE\n")[1].split("\nSTOPPING")[0]
                info['response'] = response_section.strip()
        
        steps_info.append(info)
    
    return df, steps_info

def analyze_action_reasonableness(step_info):
    """
    分析动作选择的合理性
    返回: (是否合理, 推荐动作, 原因)
    """
    num_arrows = step_info.get('num_arrows', 0)
    model_action = step_info.get('model_action', -1)
    
    # 检查动作编号是否越界
    valid_range = list(range(num_arrows)) if num_arrows > 0 else [0]
    
    if model_action not in valid_range:
        return False, valid_range[0] if valid_range else 0, f"动作编号 {model_action} 超出范围 {valid_range}"
    
    # 如果只有 0 个箭头，应该选择 0（turn around）
    if num_arrows == 0:
        if model_action == 0:
            return True, 0, "无可用路径，正确选择转身"
        else:
            return False, 0, f"无可用路径但选择了 {model_action}"
    
    # 如果有多个箭头，分析策略
    # Action 0 通常是 turn around，适合在死胡同或需要重新探索时
    if model_action == 0:
        if num_arrows <= 2:
            return True, 0, "可用选项少，选择转身重新探索是合理的"
        else:
            return None, None, "需要查看图像判断转身是否合理"
    
    # 其他动作：假设是向前/转向，通常合理
    return None, None, "需要查看图像判断具体动作合理性"

def create_analysis_report(steps_info):
    """创建详细的分析报告"""
    
    print("=" * 100)
    print("📊 VLM 导航决策合理性分析报告")
    print("=" * 100)
    print()
    
    total_steps = len(steps_info)
    valid_actions = 0
    invalid_actions = 0
    reasonable_decisions = 0
    questionable_decisions = 0
    
    for step_info in steps_info:
        step_num = step_info['step']
        num_arrows = step_info.get('num_arrows', 0)
        model_action = step_info.get('model_action', -1)
        
        is_valid = model_action in range(num_arrows) if num_arrows > 0 else model_action == 0
        reasonableness, recommended, reason = analyze_action_reasonableness(step_info)
        
        print(f"{'='*100}")
        print(f"📍 Step {step_num}")
        print(f"{'='*100}")
        print(f"  🔢 可用动作数: {num_arrows} (编号范围: 0-{max(0, num_arrows-1)})")
        print(f"  🤖 模型选择: Action {model_action}")
        print(f"  ✅ 动作有效: {'是' if is_valid else '❌ 否'}")
        
        if reasonableness is not None:
            status = "✅ 合理" if reasonableness else "❌ 不合理"
            print(f"  🎯 决策合理性: {status}")
            print(f"  💡 推荐动作: Action {recommended}")
            print(f"  📝 原因: {reason}")
            
            if reasonableness:
                reasonable_decisions += 1
            else:
                questionable_decisions += 1
        else:
            print(f"  ⚠️  决策合理性: 需要查看图像判断")
            questionable_decisions += 1
        
        if is_valid:
            valid_actions += 1
        else:
            invalid_actions += 1
        
        print(f"  💬 模型响应: {step_info.get('response', 'N/A')[:100]}")
        print()
    
    # 统计总结
    print(f"\n{'='*100}")
    print("📈 整体统计")
    print(f"{'='*100}")
    print(f"  总步数: {total_steps}")
    print(f"  有效动作: {valid_actions}/{total_steps} ({valid_actions/total_steps*100:.1f}%)")
    print(f"  无效动作: {invalid_actions}/{total_steps} ({invalid_actions/total_steps*100:.1f}%)")
    print(f"  合理决策: {reasonable_decisions}")
    print(f"  待评估决策: {questionable_decisions}")
    print()
    
    if invalid_actions > 0:
        print(f"  ⚠️  **关键问题**: {invalid_actions} 个步骤选择了无效的动作编号！")
        print(f"     可能原因:")
        print(f"     1. 模型将箭头上的视觉标签数字当成了动作编号")
        print(f"     2. Prompt 中没有明确说明编号范围是 [0, N-1]")
        print(f"     3. 箭头渲染逻辑可能有问题")
        print()

def visualize_step_comparison(steps_info, step_nums=[0, 1, 2]):
    """可视化特定步骤的对比"""
    
    for step_num in step_nums:
        if step_num >= len(steps_info):
            continue
        
        step_info = steps_info[step_num]
        
        fig = plt.figure(figsize=(20, 8))
        gs = GridSpec(1, 3, width_ratios=[1, 1, 1])
        
        # 原始观测
        if step_info['color_image'].exists():
            ax1 = plt.subplot(gs[0])
            img = Image.open(step_info['color_image'])
            ax1.imshow(img)
            ax1.set_title(f"Step {step_num}: 原始观测\n(可用动作: {step_info.get('num_arrows', 'N/A')})", fontsize=12, fontweight='bold')
            ax1.axis('off')
        
        # 标注后的观测
        if step_info['chosen_image'].exists():
            ax2 = plt.subplot(gs[1])
            img = Image.open(step_info['chosen_image'])
            ax2.imshow(img)
            model_action = step_info.get('model_action', -1)
            ax2.set_title(f"Step {step_num}: 标注图像\n模型选择: Action {model_action}", fontsize=12, fontweight='bold')
            ax2.axis('off')
        
        # 俯视图
        if step_info['voxel_map'].exists():
            ax3 = plt.subplot(gs[2])
            img = Image.open(step_info['voxel_map'])
            ax3.imshow(img)
            ax3.set_title(f"Step {step_num}: 俯视图\n(未发送给模型)", fontsize=12, fontweight='bold')
            ax3.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'/tmp/step_{step_num}_analysis.png', dpi=150, bbox_inches='tight')
        print(f"✅ Step {step_num} 可视化已保存: /tmp/step_{step_num}_analysis.png")
        plt.close()

if __name__ == "__main__":
    # 查找最新日志
    log_base = Path('/home/tao_h/VLMnav/logs')
    latest_log = sorted(log_base.glob('ObjectNav_*'))[-1]
    
    print(f"📂 分析日志目录: {latest_log}\n")
    
    # 加载数据
    result = load_episode_data(latest_log)
    if result is None:
        sys.exit(1)
    
    df, steps_info = result
    
    # 创建文本报告
    create_analysis_report(steps_info)
    
    # 可视化前几个步骤
    print("\n🎨 生成可视化对比图...")
    visualize_step_comparison(steps_info, step_nums=[0, 1, 2, 3])
    
    print("\n✅ 分析完成！")
