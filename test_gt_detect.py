"""GT 检测注入单测 — ideal_detect (SEM-Nav-Ideal, 表 IV 上限行)

不起模拟器: object.__new__ 绕过 __init__, 真导航图 + 桩深度相机。
验证五类行为:
  1. gt_goal_pos 未注入 → None (caller 回退 YOLO 的信号)
  2. 沿相机前向 2m → 检出, 正投影几何精确 (u=960/v=540/middle_center)
  3. 背后 (z<0.4) / 超远 (z>8) / 出画 → miss
  4. 深度遮挡 (窗口内更近物体) → miss
  5. 返回 info 与 yolo.check_target 同构 (下游零改动的前提)
"""
import sys, os, json, unittest
import numpy as np

sys.path.insert(0, '/home/tao_h/avdb_habitat_converter/scripts')
sys.path.insert(0, '/home/tao_h/VLMnav/src')
from avdb_sim_wrapper import AVDBSimWrapper
from avdb_depth import AVDBDepthCamera

GRAPH = '/home/tao_h/avdb_habitat_converter/data/processed/navigation_graph.json'


class FakeDepthCam:
    """桩深度相机: 几何用真静态方法 cam_R, 深度图/地面参数可控"""
    cam_R = staticmethod(AVDBDepthCamera.cam_R)

    def __init__(self, depth_img=None, floor_h=0.86):
        self.depth_img = depth_img
        self.floor_h = floor_h

    def floor_params(self, node):
        return (3.0, self.floor_h)

    def load_depth(self, node):
        return self.depth_img


def make_wrapper(depth_img=None):
    w = object.__new__(AVDBSimWrapper)
    with open(GRAPH) as f:
        w.nav_graph = json.load(f)
    w.gt_goal_pos = None
    w.depth_cam = FakeDepthCam(depth_img)
    return w


def first_node(nav_graph):
    name = next(iter(nav_graph['nodes']))
    return name, nav_graph['nodes'][name]


class TestIdealDetect(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.w = make_wrapper()
        cls.node, cls.nd = first_node(cls.w.nav_graph)
        # 相机几何: cam_center = world_pos + [0,h,0]; 前向 = direction 归一化
        cls.h = 0.86
        cls.cam_center = np.array(cls.nd['world_pos'], float) \
            + np.array([0.0, cls.h, 0.0])
        d = np.array(cls.nd['direction'], float)
        cls.d_hat = d / np.linalg.norm(d)
        cls.R = AVDBDepthCamera.cam_R(cls.nd['direction'])

    def test_cam_convention_forward_maps_to_z(self):
        """前置校验: cam_R 行=相机三轴 → R @ 前向 = [0,0,1] (测试自身的几何前提)"""
        self.assertTrue(np.allclose(self.R @ self.d_hat, [0, 0, 1], atol=1e-9))

    def test_no_gt_injected_returns_none(self):
        """未注入 gt_goal_pos → None, caller 应回退 YOLO"""
        self.w.gt_goal_pos = None
        self.assertIsNone(self.w.ideal_detect(self.node, 'tv'))

    def test_goal_along_forward_2m_detected(self):
        """目标在相机正前 2m → 检出; u=960 v=540 → middle_center, bbox 居中"""
        self.w.gt_goal_pos = (self.cam_center + self.d_hat * 2.0).tolist()
        self.w.depth_cam.depth_img = np.zeros((1080, 1920), np.float32)  # 无遮挡
        found, info = self.w.ideal_detect(self.node, 'tv')
        self.assertTrue(found)
        self.assertEqual(info['confidence'], 1.0)
        self.assertEqual(info['position'], 'middle_center')
        det = info['all_detections'][0]
        # 正投影: 半边长 = 0.14/2 * 975 = 68.25px → 归一化宽 136.5/1920
        half_n_w, half_n_h = 68.25 / 1920.0, 68.25 / 1080.0
        self.assertAlmostEqual(det['center_x'], 0.5, places=6)
        self.assertAlmostEqual(det['center_y'], 0.5, places=6)
        self.assertAlmostEqual(det['bbox_norm'][0], 0.5 - half_n_w, places=6)
        self.assertAlmostEqual(det['bbox_norm'][2], 0.5 + half_n_w, places=6)
        self.assertAlmostEqual(det['area_ratio'], 2 * half_n_w * 2 * half_n_h,
                               places=6)

    def test_goal_behind_camera_miss(self):
        """z < 0.4 (相机背后) → miss"""
        self.w.gt_goal_pos = (self.cam_center - self.d_hat * 2.0).tolist()
        found, info = self.w.ideal_detect(self.node, 'tv')
        self.assertFalse(found)
        self.assertFalse(info['target_found'])
        self.assertEqual(info['all_detections'], [])

    def test_goal_too_far_miss(self):
        """z > 8m → miss (超出深度可信范围)"""
        self.w.gt_goal_pos = (self.cam_center + self.d_hat * 9.0).tolist()
        found, _ = self.w.ideal_detect(self.node, 'tv')
        self.assertFalse(found)

    def test_goal_out_of_fov_miss(self):
        """横向大偏移 → 出画面 (u 越界) → miss"""
        right = self.R[0]  # 相机 x 轴 (右) 在世界的方向
        self.w.gt_goal_pos = (self.cam_center + self.d_hat * 3.0 + right * 8.0).tolist()
        found, _ = self.w.ideal_detect(self.node, 'tv')
        self.assertFalse(found)

    def test_goal_occluded_miss(self):
        """投影窗口内有更近物体 (深度 1.0m < z-0.4=1.6m) → miss"""
        self.w.gt_goal_pos = (self.cam_center + self.d_hat * 2.0).tolist()
        self.w.depth_cam.depth_img = np.full((1080, 1920), 1.0, np.float32)
        found, _ = self.w.ideal_detect(self.node, 'tv')
        self.assertFalse(found)

    def test_info_shape_isomorphic_to_yolo(self):
        """检出/miss 的 info dict 键集与 yolo.check_target 同构 (下游零改动)"""
        self.w.gt_goal_pos = (self.cam_center + self.d_hat * 2.0).tolist()
        self.w.depth_cam.depth_img = np.zeros((1080, 1920), np.float32)
        found, info = self.w.ideal_detect(self.node, 'tv')
        self.assertEqual(
            set(info.keys()),
            {'target_found', 'target_name', 'confidence', 'position',
             'area_ratio', 'all_detections'})
        self.assertEqual(
            set(info['all_detections'][0].keys()),
            {'class_id', 'class_name', 'confidence', 'bbox_norm',
             'area_ratio', 'center_x', 'center_y'})

    def test_unknown_node_miss(self):
        """节点不在图里 → miss (不崩)"""
        self.w.gt_goal_pos = (self.cam_center + self.d_hat * 2.0).tolist()
        found, info = self.w.ideal_detect('no_such_node', 'tv')
        self.assertFalse(found)
        self.assertEqual(info['all_detections'], [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
