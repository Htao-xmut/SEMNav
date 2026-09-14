"""hard_depth_filter 单测 — 表 III "推荐制 vs 硬剔除" 对照行

不起模拟器: object.__new__(AVDBEnv) + 合成 edge_options。
验证: 锥外平移候选剔除 / 旋转类保留 / 重编号对齐 / wrapper 执行列表
同步 (bm7/bm12 执行通道错位教训) / 锥内无可走项放行。
"""
import sys, os, types, unittest
sys.path.insert(0, '/home/tao_h/VLMnav/src')
from avdb_env import AVDBEnv


def make_env(obs=None):
    env = object.__new__(AVDBEnv)
    # 生产时序: _inject_photo 先设全量列表, _step_env 才拿到 obs → 桩同构
    env.simWrapper = types.SimpleNamespace(
        last_edge_options=list(obs['edge_options']) if obs else None)
    return env


def make_obs():
    opts = [
        {'chain_type': 'rotate_cw', 'chain_count': 6,
         'direction': 'turn_around', 'composite': None},          # [0] 180° 转
        {'chain_type': 'rotate_ccw', 'chain_count': 1,
         'direction': 'rotate_left', 'composite': None},          # [1] 左转
        {'chain_type': 'forward', 'chain_count': 1,
         'direction': 'forward', 'composite': None},              # [2] 前进 0°
        {'chain_type': 'right', 'chain_count': 1,
         'direction': 'right', 'composite': None},                # [3] 右移 90°
        {'composite': [('rotate_cw', 1), ('forward', 1)],
         'direction': 'rotate_right_then_forward',
         'chain_type': None, 'chain_count': 0},                   # [4] 右转+走 30°
    ]
    lines = ['## AVAILABLE DIRECTIONS']
    lines += [f'  [{i}] option {i}' for i in range(5)]
    return {
        'depth_rec_deg': 0.0,
        'edge_options': opts,
        'option_geo': [{'r': 0.0, 'theta': float(i * 30), 'kind': 'x'}
                       for i in range(5)],
        'avail_actions': '\n'.join(lines),
    }


class TestHardDepthFilter(unittest.TestCase):

    def test_cone_filter_and_renumber(self):
        """rec=0° ±45°: 右移 90° 被剔, 其余留; avail_actions 重编号对齐"""
        obs = make_obs()
        env = make_env(obs)
        env._hard_filter_depth_rec(obs)
        kept = obs['edge_options']
        self.assertEqual(len(kept), 4)
        self.assertNotIn('right', [o.get('chain_type') for o in kept])
        # 前进仍可走, 编号从 3 → 2 (右移槽位删除后前移)
        self.assertEqual(kept[2]['chain_type'], 'forward')
        lines = obs['avail_actions'].split('\n')
        self.assertEqual(len(lines), 5)  # 表头 + 4 行
        self.assertIn('[2] option 2', lines[3])
        self.assertNotIn('[4]', obs['avail_actions'])
        # option_geo 同步剪枝
        self.assertEqual(len(obs['option_geo']), 4)
        # wrapper 执行列表同步 (agent 所见 = 世界所执行)
        self.assertEqual(env.simWrapper.last_edge_options, kept)

    def test_rotations_always_kept(self):
        """旋转类不受锥约束 (180° turn_around 与左转都保留)"""
        env = make_env()
        obs = make_obs()
        env._hard_filter_depth_rec(obs)
        types_kept = [o.get('chain_type') for o in obs['edge_options']]
        self.assertIn('rotate_cw', types_kept)   # turn_around
        self.assertIn('rotate_ccw', types_kept)

    def test_no_walkable_in_cone_no_change(self):
        """rec=180°: 全部平移候选出锥 → 放行全部 (不困死)"""
        env = make_env()
        obs = make_obs()
        obs['depth_rec_deg'] = 180.0
        n_before = len(obs['edge_options'])
        env._hard_filter_depth_rec(obs)
        self.assertEqual(len(obs['edge_options']), n_before)
        self.assertIsNone(env.simWrapper.last_edge_options)  # 未触碰

    def test_no_rec_no_change(self):
        """depth_rec_deg 缺失 (grid_hub:F 剥离后) → 不剪"""
        env = make_env()
        obs = make_obs()
        obs.pop('depth_rec_deg')
        env._hard_filter_depth_rec(obs)
        self.assertEqual(len(obs['edge_options']), 5)

    def test_backward_dropped_rec_zero(self):
        """rec=0°: 后退 180° (平移类) 被剔 — 平移类才受锥约束"""
        env = make_env()
        obs = make_obs()
        obs['edge_options'].append(
            {'chain_type': 'backward', 'chain_count': 1,
             'direction': 'backward', 'composite': None})
        obs['avail_actions'] += '\n  [5] BACKWARD'
        obs['option_geo'].append({'r': 1.0, 'theta': 180.0, 'kind': 'backward'})
        env._hard_filter_depth_rec(obs)
        self.assertNotIn('backward',
                         [o.get('chain_type') for o in obs['edge_options']])


if __name__ == '__main__':
    unittest.main(verbosity=2)
