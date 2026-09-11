#!/usr/bin/env python3
"""
VLMnav 错误分析脚本
分析高错误率的根本原因
"""

import os
import sys
from pathlib import Path

def analyze_errors(log_dir):
    """分析指定日志目录中的错误模式"""
    
    print("=" * 80)
    print("🔍 VLMnav 错误分析报告")
    print("=" * 80)
    print(f"\n📁 分析目录: {log_dir}\n")
    
    episode_dirs = sorted([d for d in Path(log_dir).glob('0_of_1/*/') if d.is_dir()])
    
    if not episode_dirs:
        print("❌ 未找到任何 episode 目录")
        return
    
    total_episodes = len(episode_dirs)
    total_steps = 0
    error_steps = 0
    error_types = {}
    
    print(f"📊 找到 {total_episodes} 个 episodes\n")
    
    for ep_dir in episode_dirs:
        ep_name = ep_dir.name
        steps = sorted([s for s in ep_dir.glob('step*') if s.is_dir()])
        
        ep_total = len(steps)
        ep_errors = len([s for s in steps if 'ERROR' in s.name])
        
        total_steps += ep_total
        error_steps += ep_errors
        
        # 分析错误步骤
        for step_dir in steps:
            if 'ERROR' not in step_dir.name:
                continue
            
            step_num = step_dir.name.replace('step', '').replace('_ERROR', '')
            
            # 检查有哪些文件
            files = list(step_dir.iterdir())
            file_names = [f.name for f in files]
            
            # 判断错误类型
            if 'details.txt' in file_names:
                details_size = (step_dir / 'details.txt').stat().st_size
                if details_size == 0:
                    error_type = "LOGGING_FAILED (details.txt is empty)"
                else:
                    error_type = "UNKNOWN (details.txt exists but marked as ERROR)"
            else:
                error_type = "CRASH_BEFORE_LOGGING (no details.txt)"
            
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # 打印每个 episode 的统计
        success_rate = ((ep_total - ep_errors) / ep_total * 100) if ep_total > 0 else 0
        print(f"Episode {ep_name}:")
        print(f"  Steps: {ep_total}, Errors: {ep_errors}, Success Rate: {success_rate:.1f}%")
        
        # 列出错误步骤
        if ep_errors > 0:
            error_nums = [s.name.replace('step', '').replace('_ERROR', '') 
                         for s in steps if 'ERROR' in s.name]
            print(f"  Error Steps: {', '.join(error_nums)}")
        print()
    
    # 总体统计
    print("=" * 80)
    print("📈 总体统计")
    print("=" * 80)
    print(f"总步骤数: {total_steps}")
    print(f"错误步骤数: {error_steps}")
    print(f"总体成功率: {(total_steps - error_steps) / total_steps * 100:.1f}%")
    print()
    
    print("🔴 错误类型分布:")
    for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
        percentage = count / error_steps * 100 if error_steps > 0 else 0
        print(f"  {error_type}: {count} ({percentage:.1f}%)")
    print()
    
    # 建议
    print("=" * 80)
    print("💡 诊断建议")
    print("=" * 80)
    
    if "CRASH_BEFORE_LOGGING" in error_types or "LOGGING_FAILED" in error_types:
        print("\n⚠️  发现代码级错误（在记录日志前崩溃）")
        print("   可能原因:")
        print("   1. VLM API 调用超时或返回空响应")
        print("   2. JSON 解析失败导致后续代码崩溃")
        print("   3. 动作转换函数收到无效的 action_number")
        print("\n   建议修复:")
        print("   - 检查 Ollama API 是否稳定")
        print("   - 增加 VLM 调用的重试机制")
        print("   - 添加更详细的异常捕获和日志")
    
    overall_success = (total_steps - error_steps) / total_steps * 100
    if overall_success < 50:
        print(f"\n❌ 成功率过低 ({overall_success:.1f}%)")
        print("   这会影响实验结果的可信度")
        print("   建议先修复错误再运行完整评估")
    elif overall_success < 70:
        print(f"\n⚠️  成功率偏低 ({overall_success:.1f}%)")
        print("   可以接受，但建议优化")
    else:
        print(f"\n✅ 成功率可接受 ({overall_success:.1f}%)")
    
    print()


if __name__ == "__main__":
    # 查找最新的日志目录
    log_base = "/home/tao_h/VLMnav/logs"
    log_dirs = sorted([d for d in Path(log_base).glob('ObjectNav_*') if d.is_dir()])
    
    if not log_dirs:
        print("❌ 未找到任何 ObjectNav 日志目录")
        sys.exit(1)
    
    latest_log = log_dirs[-1]
    analyze_errors(str(latest_log))
