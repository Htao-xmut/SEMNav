#!/usr/bin/env python3
"""失败归因 — batch CSV → 表 IX 字母编码计数 (ApexNAV Fig.7 式)

用法:
  python -u failure_attribution.py logs/batch_+C+G+S+W_s0_n32.csv [--write]

分类判据 (口径与论文 §V-F 表 IX 一致; 全部可从 batch_run 计量列推导):
  A 成功        : goal_reached=True
  B 误停通过    : finish_status='fp' 且停票门有参与 (stop_vetoed>0 或
                  votes_cleared>0 — 拦过别的请求, 这单漏拦)
  C 连票绕门    : fp 且门全程静默 (stop_requests>0 但 stop_vetoed=0 且
                  votes_cleared=0) — bm17 型; 1.2 修复后应为 0
  D 漏检超时    : max_steps (子分: not_found_exhausted 为体面收场)
  E 死锁/打转   : max_steps 且终距 ≥ 初始测地距离 (净零进展启发式)
  F 深度建图漏占: 需逐步日志人工核验, 脚本不计 (保守 0)

--write: 把 failure_code 列写回 CSV (原地加列)。
"""
import sys, argparse
import pandas as pd


def classify(row):
    """单 episode → (编码, 说明)"""
    if row.get('goal_reached'):
        return 'A', 'success'
    fs = row.get('finish_status', '')
    if fs == 'fp':
        gate_active = row.get('stop_vetoed', 0) > 0 or row.get('votes_cleared', 0) > 0
        if row.get('stop_requests', 0) > 0 and not gate_active:
            return 'C', 'fp with silent gate (bypass pattern)'
        return 'B', 'fp passed gate'
    if fs == 'not_found_exhausted':
        return 'D', 'not found (dignified coverage stop)'
    if fs == 'max_steps':
        if row.get('initial_geodesic', 0) > 0 \
                and row.get('final_distance', 0) >= row.get('initial_geodesic', 0):
            return 'E', 'timeout, net-zero progress (wander/lock heuristic)'
        return 'D', 'timeout (detection recall)'
    return '?', f'abnormal finish_status={fs}'


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('csv', help='batch_run 输出的 CSV 路径')
    p.add_argument('--write', action='store_true', help='failure_code 列写回 CSV')
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    codes, notes = zip(*[classify(r) for _, r in df.iterrows()])
    df['failure_code'] = codes
    df['failure_note'] = notes

    n = len(df)
    print(f'===== 表 IX 失败模式分类: {args.csv} (n={n}) =====')
    label = {
        'A': '成功', 'B': '误停通过 (fp 停票漏拦)', 'C': '连票绕门',
        'D': '漏检超时 (含体面收场)', 'E': '死锁/打转 (启发式)',
        'F': '深度建图漏占 (需人工核验)', '?': '异常终局',
    }
    order = ['A', 'B', 'C', 'D', 'E', 'F', '?']
    for c in order:
        k = (df['failure_code'] == c).sum()
        if c == 'F':
            print(f'| {c} | {label[c]:<24} | {k:>3} (脚本不计) |')
            continue
        print(f'| {c} | {label[c]:<24} | {k:>3} ({k / n:.0%}) |')

    # 细分参考信号
    sub = df[df['failure_code'] == 'D']['finish_status'].value_counts().to_dict()
    if sub:
        print(f'D 细分: {sub}')
    if (df['failure_code'] == 'C').any():
        print('\n⚠️ C 类非零 — 停票门存在静默路径, 需查:')
        print(df[df['failure_code'] == 'C'][
            ['episode_ndx', 'category', 'final_distance', 'stop_requests']].to_string())

    if args.write:
        df.to_csv(args.csv, index=False)
        print(f'\n[ATTR] failure_code 列已写回 {args.csv}')


if __name__ == '__main__':
    main()
