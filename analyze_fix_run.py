import pandas as pd
df = pd.read_pickle('logs/ObjectNav_avdb_depth_ep51_coca_fix/0_of_1/0_Home_001_1/df_results.pkl')
d = df['distance_to_goal']
print('steps:', len(df), '| min: %.2fm @ step %d' % (d.min(), d.idxmin()),
      '| final: %.2f' % d.iloc[-1], '| finish:', df['finish_status'].iloc[-1])
print()
for i in range(len(df)):
    row = df.iloc[i]
    mark = ' ← STOP' if row.get('called_stopping') else ''
    print(f'step {i:2d}: dist={row["distance_to_goal"]:5.2f} act={row.get("action_number")}{mark}')
