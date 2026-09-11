#!/usr/bin/env python3
"""
深度决策合理性分析
分析模型在 details.txt 中的推理过程与最终动作选择的匹配度
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional

class DeepDecisionAnalyzer:
    """深度决策分析器"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        
    def parse_step_details(self, step_dir: Path) -> Optional[Dict]:
        """解析单个步骤的详细信息"""
        details_file = step_dir / "details.txt"
        if not details_file.exists():
            return None
        
        with open(details_file, 'r') as f:
            content = f.read()
        
        info = {
            'step': int(step_dir.name.replace("step", "")),
            'raw_content': content,
        }
        
        # 提取 PROMPT
        if "PROMPT\n" in content:
            prompt_section = content.split("PROMPT\n")[1].split("\nRESPONSE")[0]
            info['prompt'] = prompt_section
            
            # 提取箭头数量
            arrow_match = re.search(r"There are (\d+) red arrow", prompt_section)
            if arrow_match:
                info['num_arrows'] = int(arrow_match.group(1))
            
            # 提取有效范围
            range_match = re.search(r"between 0 and (\d+)", prompt_section)
            if range_match:
                info['max_action'] = int(range_match.group(1))
        
        # 提取 RESPONSE（模型的完整回复）
        if "RESPONSE\n" in content:
            response_section = content.split("RESPONSE\n")[1].split("\nSTOPPING")[0].strip()
            info['response'] = response_section
            
            # 尝试解析 JSON
            try:
                # 清理可能的 markdown 代码块
                cleaned = response_section.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                
                response_json = json.loads(cleaned)
                info['parsed_response'] = response_json
                
                # 提取关键字段
                if 'action' in response_json:
                    info['model_action'] = int(response_json['action'])
                if 'reasoning' in response_json or 'observation' in response_json:
                    info['reasoning'] = response_json.get('reasoning', '') or response_json.get('observation', '')
            except (json.JSONDecodeError, ValueError) as e:
                info['parse_error'] = str(e)
                info['model_action'] = -1
        
        # 提取 ACTION_NUMBER（实际执行的动作）
        if "ACTION_NUMBER\n" in content:
            action_line = content.split("ACTION_NUMBER\n")[1].split("\n")[0]
            try:
                info['executed_action'] = int(action_line)
            except:
                info['executed_action'] = -1
        
        # 计算有效性
        num_arrows = info.get('num_arrows', 0)
        model_action = info.get('model_action', -1)
        executed_action = info.get('executed_action', -1)
        
        if num_arrows > 0:
            info['is_valid'] = 0 <= model_action < num_arrows
            info['action_in_range'] = f"[0, {num_arrows-1}]"
        else:
            info['is_valid'] = False
            info['action_in_range'] = "N/A"
        
        # 判断推理与动作是否一致
        if 'reasoning' in info and 'model_action' in info:
            reasoning = info['reasoning'].lower()
            action = info['model_action']
            
            # 简单的语义一致性检查
            consistency_hints = []
            if action == 0 and ('turn around' in reasoning or 'back' in reasoning or 'reverse' in reasoning):
                consistency_hints.append("✅ 转身动作与推理一致")
            elif action > 0 and ('forward' in reasoning or 'ahead' in reasoning or 'straight' in reasoning):
                consistency_hints.append("✅ 前进动作与推理一致")
            elif 'left' in reasoning and action > 0:
                consistency_hints.append("⚠️  提到左转但选择了动作 {}".format(action))
            elif 'right' in reasoning and action > 0:
                consistency_hints.append("⚠️  提到右转但选择了动作 {}".format(action))
            
            info['consistency'] = consistency_hints
        
        return info
    
    def analyze_episode(self, episode_dir: Path) -> List[Dict]:
        """分析整个 episode"""
        steps_analysis = []
        
        for step_dir in sorted(episode_dir.glob("step*")):
            if not step_dir.is_dir() or "ERROR" in step_dir.name:
                continue
            
            try:
                step_info = self.parse_step_details(step_dir)
                if step_info:
                    steps_analysis.append(step_info)
            except Exception as e:
                print(f"⚠️  解析 {step_dir.name} 失败: {e}")
        
        return steps_analysis
    
    def generate_detailed_report(self, steps_analysis: List[Dict]) -> str:
        """生成详细的分析报告"""
        report = "# VLM 导航决策深度分析报告\n\n"
        report += "## 📊 总体统计\n\n"
        
        total_steps = len(steps_analysis)
        valid_steps = sum(1 for s in steps_analysis if s.get('is_valid', False))
        invalid_steps = total_steps - valid_steps
        
        report += f"| 指标 | 数值 |\n"
        report += f"|------|------|\n"
        report += f"| 总步数 | {total_steps} |\n"
        report += f"| 有效动作 | {valid_steps} |\n"
        report += f"| 无效动作 | {invalid_steps} |\n"
        report += f"| **有效率** | **{valid_steps/total_steps*100:.1f}%** |\n\n"
        
        # 逐步骤深度分析
        report += "## 🔍 逐步骤决策分析\n\n"
        
        for step_info in steps_analysis:
            step_num = step_info['step']
            report += f"### Step {step_num}\n\n"
            
            # 基本信息
            report += f"**可用动作**: {step_info.get('num_arrows', 'N/A')} 个箭头 {step_info.get('action_in_range', '')}\n\n"
            
            # 模型推理
            if 'reasoning' in step_info:
                reasoning_preview = step_info['reasoning'][:200]
                report += f"**模型推理**:\n> {reasoning_preview}...\n\n"
            
            # 动作选择
            model_action = step_info.get('model_action', 'N/A')
            executed_action = step_info.get('executed_action', 'N/A')
            is_valid = step_info.get('is_valid', False)
            
            status_icon = "✅" if is_valid else "❌"
            report += f"**动作选择**: {status_icon} 模型选择 `{model_action}`, 执行 `{executed_action}`\n\n"
            
            # 一致性分析
            if 'consistency' in step_info and step_info['consistency']:
                report += "**一致性检查**:\n"
                for hint in step_info['consistency']:
                    report += f"- {hint}\n"
                report += "\n"
            
            # 错误分析
            if not is_valid:
                max_action = step_info.get('max_action', step_info.get('num_arrows', 0) - 1)
                report += f"**❌ 错误原因**: 动作 `{model_action}` 超出有效范围 [0, {max_action}]\n\n"
            
            report += "---\n\n"
        
        # 关键发现总结
        report += "## 💡 关键发现与建议\n\n"
        
        # 常见推理模式
        reasoning_patterns = {}
        for step_info in steps_analysis:
            if 'reasoning' in step_info:
                reasoning = step_info['reasoning'].lower()
                if 'bed' in reasoning or 'bedroom' in reasoning:
                    reasoning_patterns['寻找床/卧室'] = reasoning_patterns.get('寻找床/卧室', 0) + 1
                if 'door' in reasoning or 'doorway' in reasoning:
                    reasoning_patterns['识别门/通道'] = reasoning_patterns.get('识别门/通道', 0) + 1
                if 'kitchen' in reasoning or 'living room' in reasoning:
                    reasoning_patterns['识别房间类型'] = reasoning_patterns.get('识别房间类型', 0) + 1
                if 'obstacle' in reasoning or 'block' in reasoning or 'wall' in reasoning:
                    reasoning_patterns['避障考虑'] = reasoning_patterns.get('避障考虑', 0) + 1
        
        if reasoning_patterns:
            report += "### 推理模式统计\n\n"
            for pattern, count in sorted(reasoning_patterns.items(), key=lambda x: x[1], reverse=True):
                report += f"- **{pattern}**: {count} 次\n"
            report += "\n"
        
        # 改进建议
        report += "### 改进建议\n\n"
        if invalid_steps > 0:
            report += f"1. ⚠️  **仍有 {invalid_steps} 个无效动作**，需确认箭头标签是否完全从 0 开始\n"
        
        if valid_steps == total_steps:
            report += "1. ✅ **所有动作均有效**，决策质量优秀\n"
            report += "2. 🗺️ **建议集成俯视图**，提升空间定位能力\n"
            report += "3. 📝 **添加短期记忆**，注入最近动作历史\n"
        
        # 检查是否有 NO PATH FOUND
        no_path_count = sum(1 for s in steps_analysis if "NO PATH FOUND" in s.get('raw_content', ''))
        if no_path_count > 0:
            report += f"4. 🔧 **路径规划问题**: {no_path_count} 步出现 'NO PATH FOUND'，需修复仿真环境\n"
        
        return report


def main():
    """主函数"""
    # 查找最新日志
    log_base = Path('/home/tao_h/VLMnav/logs')
    objectnav_logs = sorted(log_base.glob('ObjectNav_*'))
    
    if not objectnav_logs:
        print("❌ 未找到 ObjectNav 日志")
        return
    
    latest_log = objectnav_logs[-1]
    print(f"📂 分析最新日志: {latest_log.name}")
    
    # 查找 episode 目录（多种可能的结构）
    episode_dirs = []
    
    # 尝试多种路径模式
    for pattern in ["*/0_of_1/*", "0_of_1/*"]:
        found = list(latest_log.glob(pattern))
        if found:
            episode_dirs.extend([d for d in found if d.is_dir()])
    
    if not episode_dirs:
        print("❌ 未找到 episode 目录")
        print(f"   可用目录: {list(latest_log.iterdir())}")
        return
    
    episode_dir = episode_dirs[0]
    print(f"🎯 Episode: {episode_dir.name}")
    
    # 执行深度分析
    analyzer = DeepDecisionAnalyzer(latest_log)
    steps_analysis = analyzer.analyze_episode(episode_dir)
    
    if not steps_analysis:
        print("⚠️  未找到有效数据")
        return
    
    print(f"📊 分析 {len(steps_analysis)} 个步骤\n")
    
    # 生成报告
    report = analyzer.generate_detailed_report(steps_analysis)
    
    # 打印报告
    print(report)
    
    # 保存报告
    report_file = latest_log / "deep_decision_analysis.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n✅ 深度分析报告已保存至: {report_file}")


if __name__ == "__main__":
    main()
