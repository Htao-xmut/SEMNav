"""检测器变体单测 — direct_vlm / maxconf_fusion (表 IV 行)

不起模拟器: 桩 VLM client (call_chat 返回预设 JSON) + 桩 simWrapper。
验证: grounding 解析 (0-1000/0-1 双口径, 无框视同 miss, conf 兜底钳位) /
direct_vlm 全量替换 + 目击登记 / maxconf 只在 YOLO 检出时问 VLM 且取 max。
"""
import sys, types, unittest
import numpy as np
sys.path.insert(0, '/home/tao_h/VLMnav/src')
from avdb_env import AVDBEnv

IMG = np.zeros((480, 640, 3), np.uint8)


class FakeVLM:
    def __init__(self, resp):
        self.resp = resp
        self.calls = 0

    def call_chat(self, *a, **kw):
        self.calls += 1
        return self.resp


def make_env(resp='', ff=None):
    env = object.__new__(AVDBEnv)
    env._ff = ff or {}
    vlm = FakeVLM(resp)
    env.agent = types.SimpleNamespace(actionVLM=vlm)
    env.simWrapper = types.SimpleNamespace(
        current_node='n1', target_name='tv',
        memory={'step_count': 3, 'rooms': {}},
        target_sighting_nodes=[],
        _record_detection=lambda *a, **k: None,
        yolo=types.SimpleNamespace(
            format_for_prompt=lambda info, t, yolo_ever_found=False:
            f"FMT conf={info.get('confidence', 0):.2f} found={info.get('target_found')}"),
    )
    return env, vlm


class TestGroundingDetect(unittest.TestCase):

    def test_hit_bbox_0_1000(self):
        env, vlm = make_env('{"present": 1, "confidence": 0.80, '
                            '"bbox": [100, 200, 500, 800]}')
        found, info = env._vlm_grounding_detect(IMG, 'tv')
        self.assertTrue(found)
        self.assertEqual(info['confidence'], 0.80)
        self.assertEqual(info['bbox_norm'] if 'bbox_norm' in info
                         else info['all_detections'][0]['bbox_norm'],
                         [0.1, 0.2, 0.5, 0.8])
        det = info['all_detections'][0]
        self.assertAlmostEqual(det['area_ratio'], 0.4 * 0.6, places=6)
        self.assertEqual(info['position'], 'middle_left')  # cx=.3 cy=.5

    def test_hit_bbox_already_normalized(self):
        """bbox 0-1 口径 (max ≤1.05) 不再除 1000"""
        env, _ = make_env('{"present": 1, "confidence": 0.9, '
                          '"bbox": [0.1, 0.2, 0.5, 0.8]}')
        found, info = env._vlm_grounding_detect(IMG, 'tv')
        self.assertTrue(found)
        self.assertEqual(info['all_detections'][0]['bbox_norm'],
                         [0.1, 0.2, 0.5, 0.8])

    def test_present_zero_miss(self):
        env, _ = make_env('{"present": 0, "confidence": 0, "bbox": null}')
        found, info = env._vlm_grounding_detect(IMG, 'tv')
        self.assertFalse(found)
        self.assertFalse(info['target_found'])

    def test_present_without_bbox_miss(self):
        """present=1 但无 bbox → 视同 miss (无框无法伺服/过米制门)"""
        env, _ = make_env('{"present": 1, "confidence": 0.9, "bbox": null}')
        found, info = env._vlm_grounding_detect(IMG, 'tv')
        self.assertFalse(found)

    def test_conf_default_and_clamp(self):
        env, _ = make_env('{"present": 1, "confidence": 0.10, '
                          '"bbox": [100, 100, 400, 400]}')
        _, info = env._vlm_grounding_detect(IMG, 'tv')
        self.assertEqual(info['confidence'], 0.30)  # 钳到下限
        env2, _ = make_env('{"present": 1, "bbox": [100, 100, 400, 400]}')
        _, info2 = env2._vlm_grounding_detect(IMG, 'tv')
        self.assertEqual(info2['confidence'], 0.60)  # 缺省

    def test_garbage_response_miss(self):
        env, _ = make_env('I think maybe there is a TV somewhere...')
        found, info = env._vlm_grounding_detect(IMG, 'tv')
        self.assertFalse(found)
        self.assertEqual(info['all_detections'], [])

    def test_inverted_bbox_miss(self):
        env, _ = make_env('{"present": 1, "confidence": 0.9, '
                          '"bbox": [500, 500, 100, 100]}')
        found, _ = env._vlm_grounding_detect(IMG, 'tv')
        self.assertFalse(found)


class TestApplyVariants(unittest.TestCase):

    def test_direct_vlm_replaces_and_registers(self):
        env, vlm = make_env('{"present": 1, "confidence": 0.75, '
                            '"bbox": [200, 200, 600, 700]}',
                            ff={'direct_vlm': True})
        sightings = []
        env.simWrapper.target_sighting_nodes = sightings
        obs = {'color_sensor': IMG,
               'yolo_detection': {'target_found': False, 'all_detections': []}}
        env._apply_detector_variants(obs)
        self.assertEqual(vlm.calls, 1)
        self.assertTrue(obs['yolo_detection']['target_found'])
        self.assertEqual(obs['yolo_detection']['confidence'], 0.75)
        self.assertIn('FMT conf=0.75 found=True', obs['yolo_text'])
        self.assertEqual(len(sightings), 1)          # VLM 目击登记
        self.assertIn('DirectVLM', sightings[0][2])

    def test_maxconf_fuses_when_yolo_found(self):
        env, vlm = make_env('{"present": 1, "confidence": 0.85, '
                            '"bbox": [200, 200, 600, 700]}',
                            ff={'maxconf_fusion': True})
        yolo_info = {'target_found': True, 'target_name': 'tv',
                     'confidence': 0.40, 'position': 'middle_center',
                     'area_ratio': 0.05,
                     'all_detections': [{'class_id': 58, 'class_name': 'tv',
                                         'confidence': 0.40,
                                         'bbox_norm': [0.3, 0.3, 0.5, 0.5],
                                         'area_ratio': 0.04,
                                         'center_x': 0.4, 'center_y': 0.4}]}
        obs = {'color_sensor': IMG, 'yolo_detection': yolo_info}
        env._apply_detector_variants(obs)
        self.assertEqual(vlm.calls, 1)
        self.assertEqual(obs['yolo_detection']['confidence'], 0.85)
        self.assertEqual(obs['yolo_detection']['all_detections'][0]
                         ['confidence'], 0.85)
        # bbox 仍取 YOLO 框 (只融合置信, 不换框)
        self.assertEqual(obs['yolo_detection']['all_detections'][0]
                         ['bbox_norm'], [0.3, 0.3, 0.5, 0.5])

    def test_maxconf_no_vlm_call_when_yolo_miss(self):
        """YOLO 无检出 → 不问 VLM (召回侧不变)"""
        env, vlm = make_env('', ff={'maxconf_fusion': True})
        obs = {'color_sensor': IMG,
               'yolo_detection': {'target_found': False, 'all_detections': []}}
        env._apply_detector_variants(obs)
        self.assertEqual(vlm.calls, 0)
        self.assertFalse(obs['yolo_detection']['target_found'])

    def test_maxconf_lower_vlm_keeps_yolo(self):
        """VLM 置信更低 → 保 YOLO (max), 只重建文本"""
        env, vlm = make_env('{"present": 1, "confidence": 0.35, '
                            '"bbox": [200, 200, 600, 700]}',
                            ff={'maxconf_fusion': True})
        yolo_info = {'target_found': True, 'confidence': 0.55,
                     'all_detections': [{'class_name': 'tv', 'confidence': 0.55,
                                         'bbox_norm': [0.1, 0.1, 0.2, 0.2]}]}
        obs = {'color_sensor': IMG, 'yolo_detection': yolo_info}
        env._apply_detector_variants(obs)
        self.assertEqual(obs['yolo_detection']['confidence'], 0.55)

    def test_no_flag_no_dispatch(self):
        env, vlm = make_env('', ff={})
        obs = {'color_sensor': IMG,
               'yolo_detection': {'target_found': False, 'all_detections': []}}
        env._apply_detector_variants(obs)
        self.assertEqual(vlm.calls, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
