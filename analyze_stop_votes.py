"""停止投票 + YOLO 检测 + 距离逐步对齐: 为什么 1.96m 没停"""
import pandas as pd
import re, json

df = pd.read_pickle('logs/ObjectNav_avdb_depth_ep51_coca_50step/0_of_1/0_Home_001_1/df_results.pkl')
print('cols with stop:', [c for c in df.columns if 'stop' in c.lower() or 'done' in c.lower() or 'call' in c.lower()])
print()
for i in range(len(df)):
    row = df.iloc[i]
    called = row.get('called_stopping', None)
    done = row.get('done', None)
    dist = row['distance_to_goal']
    act = row.get('action_number', None)
    succ = row.get('success', None)
    print(f'step {i:2d}: dist={dist:5.2f} act={act} called_stopping={called} done={done} success={succ}')
