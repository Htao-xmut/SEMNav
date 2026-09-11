#!/usr/bin/env python3
"""
分析 Qwen-VL 运行的 Episode 结果
"""

import pandas as pd
import pickle
import sys
from pathlib import Path

def analyze_episode(pkl_path):
    """分析单个 Episode 的结果"""
    
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)
    
    print('=' * 80)
    print(f'📊 Episode 结果分析: {pkl_path}')
    print('=' * 80)
    
    print(f'\n✅ 总步数: {len(df)}')
    
    # 检查每一步的决策
    print(f'\n📝 步骤详情:')
    for idx, row in df.iterrows():
        action = row.get('action_number', 'N/A')
        success = row.get('success', 'N/A')
        status = "✅" if success == 1 else "❌"
        print(f'  Step {idx}: {status} Action={action}, Success={success}')
    
    # 最终状态
    final_row = df.iloc[-1]
    print(f'\n🎯 最终状态 (Step {len(df)-1}):')
    print(f'  • distance_to_goal: {final_row.get("distance_to_goal", "N/A"):.2f}m')
    print(f'  • goal_reached: {final_row.get("goal_reached", False)}')
    print(f'  • finish_status: {final_row.get("finish_status", "N/A")}')
    print(f'  • spl: {final_row.get("spl", 0):.4f}')
    
    # 统计成功率
    success_count = (df['success'] == 1).sum()
    total_steps = len(df)
    step_success_rate = success_count / total_steps * 100 if total_steps > 0 else 0
    
    print(f'\n📈 统计信息:')
    print(f'  • 成功步骤数: {success_count}/{total_steps} ({step_success_rate:.1f}%)')
    print(f'  • 失败步骤数: {total_steps - success_count}')
    
    # 判断 Episode 是否成功
    episode_success = final_row.get('goal_reached', False)
    print(f'\n🏆 Episode 结果: {"✅ 成功" if episode_success else "❌ 失败"}')
    
    if not episode_success:
        print(f'  ⚠️  失败原因: 距离目标还有 {final_row.get("distance_to_goal", 0):.2f}m (阈值: 0.3m)')
    
    return episode_success

if __name__ == "__main__":
    # 查找最新的 Episode 结果
    log_dir = Path('/home/tao_h/VLMnav/logs')
    latest_run = sorted(log_dir.glob('ObjectNav_*'))[-1]
    pkl_files = list(latest_run.glob('*/0_of_1/*/df_results.pkl'))
    
    if pkl_files:
        for pkl in pkl_files[:3]:  # 分析前3个
            analyze_episode(pkl)
            print('\n')
    else:
        print("❌ 未找到结果文件")
        sys.exit(1)
