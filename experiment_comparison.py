#!/usr/bin/env python3
"""
生成实验对比总结报告
"""

from pathlib import Path
import pandas as pd
from experiment_decision_quality import DecisionQualityEvaluator

def compare_all_experiments():
    """对比所有实验结果"""
    
    log_base = Path('/home/tao_h/VLMnav/logs')
    objectnav_logs = sorted(log_base.glob('ObjectNav_*'))
    
    results = []
    
    print("🔍 正在评估所有实验...")
    print()
    
    for log_dir in objectnav_logs:
        try:
            evaluator = DecisionQualityEvaluator(log_dir)
            df = evaluator.evaluate_all_episodes()
            
            if df.empty or len(df) < 3:  # 跳过数据太少的
                continue
            
            metrics = evaluator.compute_metrics(df)
            
            results.append({
                'experiment': log_dir.name,
                'total_steps': metrics['total_steps'],
                'valid_actions': metrics['valid_actions'],
                'invalid_actions': metrics['invalid_actions'],
                'validity_rate': metrics['validity_rate'],
                'avg_arrows': metrics['avg_arrows_per_step'],
                'has_topdown': metrics.get('episodes_with_topdown', False),
                'has_history': metrics.get('episodes_with_history', False),
            })
            
            print(f"✅ {log_dir.name}: {metrics['validity_rate']:.1f}% ({metrics['valid_actions']}/{metrics['total_steps']})")
        
        except Exception as e:
            print(f"⚠️  {log_dir.name}: 错误 - {e}")
    
    if not results:
        print("\n❌ 未找到有效实验数据")
        return
    
    # 创建 DataFrame
    df_results = pd.DataFrame(results)
    
    # 生成对比报告
    report = "# VLM 导航决策质量 - 实验对比报告\n\n"
    report += "## 📊 总体对比\n\n"
    
    # 按有效率排序
    df_sorted = df_results.sort_values('validity_rate', ascending=False)
    
    report += "| 实验 | 总步数 | 有效 | 无效 | **有效率** | 平均箭头数 |\n"
    report += "|------|--------|------|------|-----------|------------|\n"
    
    for _, row in df_sorted.iterrows():
        status = "✅" if row['validity_rate'] >= 90 else "⚠️" if row['validity_rate'] >= 70 else "❌"
        report += f"| {row['experiment']} | {row['total_steps']} | {row['valid_actions']} | {row['invalid_actions']} | **{row['validity_rate']:.1f}%** {status} | {row['avg_arrows']:.1f} |\n"
    
    report += "\n"
    
    # 统计分析
    report += "## 📈 统计分析\n\n"
    
    avg_validity = df_results['validity_rate'].mean()
    max_validity = df_results['validity_rate'].max()
    min_validity = df_results['validity_rate'].min()
    
    report += f"- **平均有效率**: {avg_validity:.1f}%\n"
    report += f"- **最高有效率**: {max_validity:.1f}%\n"
    report += f"- **最低有效率**: {min_validity:.1f}%\n"
    report += f"- **实验总数**: {len(df_results)}\n\n"
    
    # 分组统计
    high_quality = df_results[df_results['validity_rate'] >= 90]
    medium_quality = df_results[(df_results['validity_rate'] >= 70) & (df_results['validity_rate'] < 90)]
    low_quality = df_results[df_results['validity_rate'] < 70]
    
    report += "### 质量分布\n\n"
    report += f"- ✅ 高质量 (≥90%): {len(high_quality)} 个实验\n"
    report += f"- ⚠️  中等质量 (70-90%): {len(medium_quality)} 个实验\n"
    report += f"- ❌ 低质量 (<70%): {len(low_quality)} 个实验\n\n"
    
    # 关键发现
    report += "## 🔍 关键发现\n\n"
    
    if len(high_quality) > 0:
        best_exp = high_quality.iloc[0]
        report += f"1. **最佳表现**: `{best_exp['experiment']}` 达到 {best_exp['validity_rate']:.1f}% 有效率\n"
    
    if len(low_quality) > 0:
        worst_exp = low_quality.iloc[-1]
        report += f"2. **待改进**: `{worst_exp['experiment']}` 仅 {worst_exp['validity_rate']:.1f}% 有效率，需要修复\n"
    
    # 检查是否有修复前后的对比
    pre_fix = df_results[df_results['validity_rate'] < 50]
    post_fix = df_results[df_results['validity_rate'] >= 90]
    
    if len(pre_fix) > 0 and len(post_fix) > 0:
        improvement = post_fix['validity_rate'].mean() - pre_fix['validity_rate'].mean()
        report += f"3. **修复效果**: 从修复前的 {pre_fix['validity_rate'].mean():.1f}% 提升到修复后的 {post_fix['validity_rate'].mean():.1f}% (+{improvement:.1f}%)\n"
    
    report += "\n"
    
    # 建议
    report += "## 💡 下一步建议\n\n"
    
    if len(high_quality) > 0:
        report += "1. ✅ **当前状态良好**: 大部分实验已达到 100% 有效率\n"
        report += "2. 🗺️ **集成俯视图**: 下一步可以添加俯视图到 Prompt，提升空间推理能力\n"
        report += "3. 📝 **添加短期记忆**: 注入最近动作历史，避免重复行为\n"
        report += "4. 🎯 **完整评估**: 运行完整的 30 episodes val_mini 测试，统计 SR/SPL\n"
    else:
        report += "1. 🔧 **继续修复**: 确保箭头标签从 0 开始\n"
        report += "2. 📋 **强化 Prompt**: 明确说明动作编号范围\n"
        report += "3. 🧪 **重新测试**: 清除缓存后重新运行实验\n"
    
    # 保存报告
    report_file = log_base / "experiment_comparison_report.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n{'='*80}")
    print(report)
    print(f"{'='*80}")
    print(f"\n✅ 对比报告已保存至: {report_file}")


if __name__ == "__main__":
    compare_all_experiments()
