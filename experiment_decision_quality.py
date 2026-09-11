#!/usr/bin/env python3
"""
VLM 导航决策质量评估实验
对比不同配置下的动作选择合理性
"""

import os
import sys
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

class DecisionQualityEvaluator:
    """决策质量评估器"""
    
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.results = []
        
    def load_episode_data(self, episode_dir):
        """加载单个 Episode 的数据"""
        steps_data = []
        
        for step_dir in sorted(episode_dir.glob("step*")):
            if not step_dir.is_dir() or "ERROR" in step_dir.name:
                continue
            
            try:
                step_num = int(step_dir.name.replace("step", ""))
            except ValueError:
                continue
            
            details_file = step_dir / "details.txt"
            if not details_file.exists():
                continue
            
            with open(details_file, 'r') as f:
                content = f.read()
            
            # 解析关键信息
            info = {
                'step': step_num,
                'episode': episode_dir.name,
            }
            
            # 提取 ACTION_NUMBER
            if "ACTION_NUMBER" in content:
                action_line = content.split("ACTION_NUMBER\n")[1].split("\n")[0]
                try:
                    info['model_action'] = int(action_line)
                except:
                    info['model_action'] = -1
            
            # 提取箭头数量
            if "There are" in content and "arrow" in content:
                import re
                match = re.search(r"There are (\d+) red arrow", content)
                if match:
                    info['num_arrows'] = int(match.group(1))
            
            # 提取 RESPONSE
            if "RESPONSE" in content:
                response_section = content.split("RESPONSE\n")[1].split("\nSTOPPING")[0]
                info['response'] = response_section.strip()
            
            # 提取 PROMPT 类型（是否包含俯视图说明）
            prompt_section = content.split("PROMPT\n")[1].split("\nRESPONSE")[0] if "PROMPT" in content else ""
            info['has_topdown_instruction'] = "TOP-DOWN MAP" in prompt_section or "俯视图" in prompt_section
            info['has_history'] = "RECENT ACTION HISTORY" in prompt_section or "history" in prompt_section.lower()
            
            # 计算有效性
            num_arrows = info.get('num_arrows', 0)
            model_action = info.get('model_action', -1)
            
            if num_arrows > 0:
                info['is_valid'] = 0 <= model_action < num_arrows
            else:
                info['is_valid'] = (model_action == 0)
            
            steps_data.append(info)
        
        return steps_data
    
    def evaluate_all_episodes(self):
        """评估所有 episodes"""
        all_steps = []
        
        # 尝试多种目录结构
        episode_dirs = []
        
        # 结构 1: log_dir/0_of_1/episode_id
        for subdir in self.log_dir.glob("*/0_of_1/*"):
            if subdir.is_dir():
                episode_dirs.append(subdir)
        
        # 结构 2: log_dir/0_of_1/episode_id (直接在第一层)
        if not episode_dirs:
            for subdir in self.log_dir.glob("0_of_1/*"):
                if subdir.is_dir():
                    episode_dirs.append(subdir)
        
        # 结构 3: 直接在 log_dir 下
        if not episode_dirs:
            for subdir in self.log_dir.glob("*"):
                if subdir.is_dir() and (subdir / "step0").exists():
                    episode_dirs.append(subdir)
        
        print(f"找到 {len(episode_dirs)} 个 episodes")
        
        for episode_dir in sorted(episode_dirs):
            steps_data = self.load_episode_data(episode_dir)
            all_steps.extend(steps_data)
        
        return pd.DataFrame(all_steps)
    
    def compute_metrics(self, df):
        """计算评估指标"""
        metrics = {}
        
        # 基础统计
        total_steps = len(df)
        valid_actions = df['is_valid'].sum()
        invalid_actions = total_steps - valid_actions
        
        metrics['total_steps'] = total_steps
        metrics['valid_actions'] = int(valid_actions)
        metrics['invalid_actions'] = int(invalid_actions)
        metrics['validity_rate'] = float(valid_actions / total_steps * 100) if total_steps > 0 else 0
        
        # 按步骤分布分析
        metrics['avg_arrows_per_step'] = float(df['num_arrows'].mean()) if 'num_arrows' in df.columns else 0
        
        # 常见错误模式
        if invalid_actions > 0:
            invalid_df = df[~df['is_valid']]
            metrics['out_of_range_errors'] = int((invalid_df['model_action'] >= invalid_df['num_arrows']).sum())
            metrics['negative_action_errors'] = int((invalid_df['model_action'] < 0).sum())
        
        # Prompt 特性分析
        if 'has_topdown_instruction' in df.columns:
            metrics['episodes_with_topdown'] = int(df['has_topdown_instruction'].any())
            metrics['episodes_with_history'] = int(df['has_history'].any())
        
        return metrics
    
    def generate_report(self, df, output_file='decision_quality_report.md'):
        """生成详细报告"""
        metrics = self.compute_metrics(df)
        
        report = f"""# VLM 导航决策质量评估报告

## 📊 总体统计

| 指标 | 数值 |
|------|------|
| 总步数 | {metrics['total_steps']} |
| 有效动作 | {metrics['valid_actions']} |
| 无效动作 | {metrics['invalid_actions']} |
| **有效率** | **{metrics['validity_rate']:.1f}%** |
| 平均每步箭头数 | {metrics['avg_arrows_per_step']:.1f} |

"""
        
        # 错误分析
        if metrics['invalid_actions'] > 0:
            report += f"""## ❌ 错误分析

| 错误类型 | 次数 |
|---------|------|
| 超出范围 | {metrics.get('out_of_range_errors', 0)} |
| 负数动作 | {metrics.get('negative_action_errors', 0)} |

"""
        
        # 按 Episode 分析
        report += "## 📈 分 Episode 统计\n\n"
        report += "| Episode | 步数 | 有效 | 无效 | 有效率 |\n"
        report += "|---------|------|------|------|--------|\n"
        
        for episode in df['episode'].unique():
            ep_df = df[df['episode'] == episode]
            valid = ep_df['is_valid'].sum()
            total = len(ep_df)
            rate = valid / total * 100 if total > 0 else 0
            report += f"| {episode} | {total} | {valid} | {total - valid} | {rate:.1f}% |\n"
        
        report += "\n"
        
        # Prompt 特性影响
        if 'has_topdown_instruction' in df.columns:
            report += "## 🔍 Prompt 特性分析\n\n"
            
            # 有俯视图说明的 episodes
            topdown_eps = df[df['has_topdown_instruction']]['episode'].unique()
            no_topdown_eps = df[~df['has_topdown_instruction']]['episode'].unique()
            
            if len(topdown_eps) > 0:
                topdown_df = df[df['episode'].isin(topdown_eps)]
                topdown_rate = topdown_df['is_valid'].mean() * 100
                report += f"- **有俯视图说明**: {len(topdown_eps)} episodes, 有效率 {topdown_rate:.1f}%\n"
            
            if len(no_topdown_eps) > 0:
                no_topdown_df = df[df['episode'].isin(no_topdown_eps)]
                no_topdown_rate = no_topdown_df['is_valid'].mean() * 100
                report += f"- **无俯视图说明**: {len(no_topdown_eps)} episodes, 有效率 {no_topdown_rate:.1f}%\n"
            
            report += "\n"
        
        # 建议
        report += """## 💡 改进建议

"""
        if metrics['validity_rate'] < 90:
            report += "- ⚠️ 有效率低于 90%，建议检查箭头标签与 Prompt 的一致性\n"
        if metrics.get('out_of_range_errors', 0) > 0:
            report += "- 🔧 存在越界错误，确认箭头标签是否从 0 开始\n"
        if not metrics.get('episodes_with_topdown', False):
            report += "- 🗺️ 建议集成俯视图到 Prompt，提升空间推理能力\n"
        if not metrics.get('episodes_with_history', False):
            report += "- 📝 建议添加短期记忆（最近动作历史），避免重复行为\n"
        
        return report


def main():
    """主函数"""
    # 查找最新日志
    log_base = Path('/home/tao_h/VLMnav/logs')
    
    if not log_base.exists():
        print("❌ 日志目录不存在")
        return
    
    # 获取所有 ObjectNav 日志
    objectnav_logs = sorted(log_base.glob('ObjectNav_*'))
    
    if not objectnav_logs:
        print("❌ 未找到 ObjectNav 日志")
        return
    
    print(f"📂 找到 {len(objectnav_logs)} 个日志目录")
    print()
    
    # 评估最新的 3 个日志
    for log_dir in objectnav_logs[-3:]:
        print(f"{'='*80}")
        print(f"📊 评估: {log_dir.name}")
        print(f"{'='*80}")
        
        evaluator = DecisionQualityEvaluator(log_dir)
        df = evaluator.evaluate_all_episodes()
        
        if df.empty:
            print("⚠️  未找到有效数据")
            print()
            continue
        
        # 生成报告
        report = evaluator.generate_report(df)
        
        # 保存报告
        report_file = log_dir / "decision_quality_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(report)
        print(f"\n✅ 报告已保存至: {report_file}")
        print()


if __name__ == "__main__":
    main()
