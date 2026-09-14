"""AVDB Env — VLM navigates, graph validates. Extends ObjectNavEnv."""
import sys, os, logging, json, gzip, re, time, numpy as np, pandas as pd
import cv2
sys.path.insert(0, '/home/tao_h/avdb_habitat_converter/scripts')
from avdb_sim_wrapper import AVDBSimWrapper, PolarAction
from avdb_agent import AVDBAgent  # make AVDBAgent available for globals() lookup
import feature_flags
from vlm_fail_policy import VLMFailPolicy
from env import ObjectNavEnv
from utils import create_gif, log_exception
import habitat_sim

class AVDBEnv(ObjectNavEnv):
    NAV_GRAPH_PATH = "/home/tao_h/avdb_habitat_converter/data/processed/navigation_graph.json"

    # === 1.3 停票门收紧: 到达判据全部对齐主判据 1.0m ===
    #   HARD_STOP_DEPTH      深探 ≤1.0m 才可停 (米制硬判据)
    #   SOFT_APPROACH_DEPTH  深探 (1.0, 2.2]m 不放行 STOP, 拒停转接近锁
    #                        继续逼近 (软接近分级; 近了走面积/硬停捷径)
    #   VISUAL_SUCCESS_THRESHOLD  VLM 判 close 且 eval <1.0m 才直达成功;
    #                        [1.0, 2.5) 落入守卫过门, 软接近伺服循环把
    #                        停距逼进 1.0m — 论文"每个停止请求过门"声明
    #                        自此对所有停止路径成立
    HARD_STOP_DEPTH = 1.0           # meters
    SOFT_APPROACH_DEPTH = 2.2       # meters
    MISMATCH_DEPTH = 0.5            # meters (矛盾检测独立边界, 见 _stop_evidence)
    SMALL_BOX_QUARANTINE_AREA = 0.005   # #50: 小框深度隔离阈值 (见 _stop_evidence)
    VISUAL_SUCCESS_THRESHOLD = 1.0  # meters

    def _initialize_agent(self, cfg: dict):
        """Override: use AVDBAgent which injects YOLO into stopping prompt."""
        PolarAction.default = PolarAction(cfg['agent_cfg']['default_action'], 0, 'default')
        cfg['agent_cfg']['sensor_cfg'] = cfg['sim_cfg']['sensor_cfg']
        self.agent: AVDBAgent = AVDBAgent(cfg['agent_cfg'])
        # 特性开关 (消融统一配置面): bm_run/batch runner 经
        # cfg['feature_flags'] 或 CLI --no-xxx 注入, 见 feature_flags.py
        self._ff = feature_flags.resolve(cfg)
        self.agent._ff = self._ff   # #52a: flag 下沉 agent 层 (角度折算门)
        self._apply_flag_overrides()
        logging.info(f'[FEATURE-FLAGS] tier={feature_flags.describe(self._ff)} '
                     f'flags={self._ff}')

    def _apply_flag_overrides(self):
        """表 IV 对照开关的判据覆盖 (实例级, 不动类常量)

        single_thresh: 收紧前单阈值判据复现 — 深探 ≤2.2m 即 close (无
        1.0 硬/2.2 软分级), VLM 视觉确认上限 2.5m。eval 侧阈值由
        batch_run/bm_run 按 1.5 配套设 (success_threshold 是 cfg 值)。
        """
        if self._ff.get('single_thresh'):
            self.HARD_STOP_DEPTH = 2.2
            self.SOFT_APPROACH_DEPTH = 2.2   # 软接近区退化消失
            self.VISUAL_SUCCESS_THRESHOLD = 2.5
            logging.info('[FEATURE-FLAGS] single_thresh: pre-1.3 flat criteria '
                         '(hard 2.2m / visual-success 2.5m)')

    def _initialize_experiment(self):
        self.all_episodes = []
        # batch runner 计量容器: 每 episode 一行 (表 VI/VIII 数据源)
        self.episode_stats_list = []
        split = self.cfg.get('split', 'val')
        target = self.cfg.get('target_category', None)
        dd = '/home/tao_h/VLMnav/data/datasets/objectnav_avdb_home001_1'
        cd = os.path.join(dd, split, "content")
        for fn in sorted(os.listdir(cd)):
            if fn.endswith('.json.gz'):
                with gzip.open(os.path.join(cd, fn), 'rt') as f:
                    data = json.load(f)
                    eps = list(data.values()) if isinstance(data, dict) else data.get('episodes', [])
                    if target: eps = [e for e in eps if e.get('object_category') == target]
                    self.all_episodes.extend(eps)
        for ep in self.all_episodes:
            if 'object' not in ep and 'goals' in ep: ep['object'] = ep['goals'][0]['object_category']
            if 'goal' not in ep and 'goals' in ep: ep['goal'] = {'name': ep['goals'][0]['object_category'], 'position': ep['goals'][0]['position']}
            if 'shortest_path' not in ep and 'info' in ep: ep['shortest_path'] = ep['info'].get('geodesic_distance', 1.0)

        # Support start_episode_idx: select episode from the FILTERED list (0-based index into matched eps)
        start_idx = self.cfg.get('start_episode_idx', 0)
        if start_idx > 0 and start_idx < len(self.all_episodes):
            picked = self.all_episodes[start_idx]
            self.all_episodes = [picked]
            gd = picked.get('info', {}).get('geodesic_distance', 0)
            logging.info(f'AVDB: using episode[{start_idx}], geod={gd:.1f}m')

        # batch runner: 连续切片 [batch_start, batch_start+batch_count)
        # (bm_run 单跑走上面的 start_episode_idx; 批量跑不设 start_episode_idx)
        b0 = self.cfg.get('batch_start', None)
        if b0 is not None and b0 > 0:
            bc = self.cfg.get('batch_count', 0) or 0
            self.all_episodes = self.all_episodes[b0:b0 + bc] if bc > 0 \
                else self.all_episodes[b0:]
            logging.info(f'AVDB: batch slice start={b0} count={bc} '
                         f'→ {len(self.all_episodes)} eps')

        self.num_episodes = len(self.all_episodes)
        logging.info(f'AVDB: {self.num_episodes} eps for {target}')

    def _walk_graph(self, start_node, edge_type, steps):
        """Walk `steps` edges of `edge_type` from start_node. Return final node name."""
        node = start_node
        for _ in range(steps):
            nxt = None
            for t, e in self.simWrapper.nav_graph['graph'].get(node, {}).items():
                if e['edge_type'] == edge_type:
                    nxt = t; break
            if nxt: node = nxt
            else: break
        return node

    def _load_photo(self, node_name):
        """Load PIL Image from a graph node's photo file."""
        from PIL import Image
        for d in ['/mnt/d/aod/AOD_dataset/Home_001_1/jpg_rgb',
                  '/home/tao_h/avdb_habitat_converter/data/processed/rgb']:
            p = os.path.join(d, node_name)
            if os.path.exists(p):
                return Image.open(p).convert('RGB')
        return None

    def _warmup_preempt_check(self, best, target):
        """任务三: warmup 强证据判定 (conf≥0.5 且 bbox 面积≥0.5% → 抢占)

        返回 (preempted, message): preempted=True 时 caller 跳过 VLM 环境
        分析直接返回; message 保持 caller 正则兼容 (\(\d+°\) 与
        confidence \*\*x%**) 供 [APPROACH-LOCK] warmup 桥接解析。
        """
        conf = best['yolo_info'].get('confidence', 0)
        direction, angle = best['direction'], best['angle']
        bbox_area = 0.0
        for det_ in (best['yolo_info'] or {}).get('all_detections', []):
            if det_.get('class_name') == target and det_.get('bbox_norm'):
                x1_, y1_, x2_, y2_ = det_['bbox_norm']
                bbox_area = max(bbox_area, max(0.0, (x2_ - x1_) * (y2_ - y1_)))
        if (getattr(self, '_ff', None) or {}).get('warmup_preempt', True) \
                and conf >= 0.5 and bbox_area >= 0.005:
            self._warmup_preempted = True
            logging.info(f'[WARMUP-PREEMPT] strong evidence conf={conf:.2f} '
                         f'area={bbox_area:.3f} at {angle}° → skip VLM '
                         f'analysis, approach lock + AUTO-STEER (zero VLM)')
            msg = (
                f"\n## 🎯 WARMUP PREEMPT (metric reflex, zero VLM)\n"
                f"YOLO detected '{target}' at **{direction} ({angle}°)** with confidence **{conf:.0%}**.\n"
                f"★ Metric layer has taken over: approach lock + AUTO-STEER walk to it.\n"
                f"★ VLM suspended until evidence decays (conf <0.3 or lost >10 steps).\n"
            )
            return True, msg
        return False, ""

    def _vlm_call(self, prompt, images, site):
        """A②/A③: 关键 VLM 调用统一通道 — 失败分类 / 重试 / 判局策略。

        只接六图扫描类关键调用 (warmup/rescan — 输出写入记忆或影响动作
        选择; 身份/接近/crop 等门问询的既有降级路径不动)。失败返回 ''
        (上层走既有降级), 策略触发时置 _vlm_episode_invalid →
        _run_episode 下一轮中止 (p26 型 18 崩在 step0/早期止损)。
        """
        pol = getattr(self, '_vlm_policy', None)
        if pol is None:
            pol = self._vlm_policy = VLMFailPolicy()
        attempt = 0
        while True:
            try:
                resp = self.agent.actionVLM.call_chat(
                    0, images, prompt, plain_text=True)
                pol.record_ok(site)
                return resp
            except Exception as e:
                kind = pol.classify(e)
                if kind == 'infra' and attempt < pol.CONFIG['retry_infra']:
                    attempt += 1
                    time.sleep(pol.CONFIG['retry_sleep_s'] * attempt)
                    logging.warning(f'[VLM-RETRY] {site} {kind} 失败 '
                                    f'({e}), 重试 {attempt}/{pol.CONFIG["retry_infra"]}')
                    continue
                if kind == 'parse' and attempt < pol.CONFIG['retry_parse']:
                    attempt += 1
                    logging.warning(f'[VLM-RETRY] {site} {kind} 失败 '
                                    f'({e}), 重试 {attempt}/{pol.CONFIG["retry_parse"]}')
                    continue
                pol.record_fail(site, kind, e)
                st = getattr(self, '_ep_stats', None)
                if st is not None:
                    st['vlm_critical_fails'] = st.get('vlm_critical_fails', 0) + 1
                abort, reason = pol.should_abort()
                if abort:
                    self._vlm_episode_invalid = True
                    self._vlm_invalid_reason = reason
                    if st is not None:
                        st['vlm_invalid'] = 1
                        st['vlm_invalid_reason'] = reason
                    logging.critical(f'[VLM-FAIL-POLICY] {reason} '
                                     f'→ 本局判无效, 下一轮提前中止')
                else:
                    logging.error(f'[VLM-FAIL] {site} {kind} 失败 '
                                  f'({e}) — 降级继续, 计 {pol.total_fails} 次')
                return ''

    def _record_exec_trace(self, obs):
        """A④: wrapper 最终执行回写 details.txt — 决策→覆盖→硬规则→执行
        全链一行审计。step27 '决定转180实际前进' 类错位从此直读可判:
        OVERRIDE 行(决策侧) + EXEC 行(日志) + FINAL-EXEC 行(真执行) 三对齐。
        """
        try:
            tr = getattr(self.simWrapper, 'last_exec_trace', None)
            if not tr:
                return
            ep_dir = (f'logs/{self.outer_run_name}/{self.inner_run_name}/'
                      f'{self.curr_run_name}/step{self.step}')
            dt_path = f'{ep_dir}/details.txt'
            if os.path.exists(dt_path):
                with open(dt_path, 'a') as f:
                    f.write(f"\nFINAL-EXEC: vlm=[{tr.get('vlm_idx')}] "
                            f"forced_by={tr.get('forced_by')} "
                            f"intent={tr.get('intent')} "
                            f"final=[{tr.get('final_edge')} x{tr.get('chain')}] "
                            f"pending_discarded={tr.get('pending_discarded')} "
                            f"sub_walk={tr.get('sub_walk')}\n")
        except Exception as e:
            logging.debug(f'[EXEC-TRACE] writeback failed: {e}')

    def _warmup_scan(self, step_tag='warmup_scan'):
        """360° environment scan with YOLO detection. Returns (summary_str, target_direction_priority).

        If YOLO detects the target, the scan records which direction and confidence,
        then generates a priority instruction to go that way first.
        Also saves a scan GIF with YOLO bounding boxes.

        step_tag: 帧目录标签 — 初始扫描 'warmup_scan' (只写一次, 永不被
        覆盖), 途中 rescan 传 'step{N}_rescan' (用户实测 bug: 旧实现每次
        rescan 重写 warmup_scan_XX 同名目录, 初始扫描帧被覆盖丢失, 事后
        取证只能看到最后一次 rescan 的画面)。
        """
        node = self.simWrapper.current_node
        if not node or not self.simWrapper.nav_graph:
            return "", None

        # === #47: 扫描基准量 — 目击锁方位 = 起始朝向 + 扫描角 + 画面内偏移 ===
        # 旧公式在锁时刻取 _yaw_world()+scan_angle 两处错: ① 锁前的途中
        # null 步 (rescan 取 obs) 曾把 current_node 偷换成同位置任意朝向
        # 变体 (bmv10 step38b 错 181.6°, wrapper 侧已改纯读); ② 相机朝向
        # ≠ 目标方位 — 漏了目标框在画面里的偏移 (step42-43: middle_right
        # = +41°, 相机 −56°+41° = −15° ≈ GT −13.8°)。扫描纯读不动
        # current_node → 入口朝向即基准; 偏移在 YOLO 命中处记 (见 best 块)。
        try:
            self._scan_base_yaw_deg = float(np.degrees(self._yaw_world()))
        except Exception:
            self._scan_base_yaw_deg = None
        self._scan_hit_offset_deg = None

        from PIL import Image, ImageDraw
        target = self.current_episode.get('object_category', 'target')

        # Collect 6 photos at 60° intervals (2 rotate edges each)
        scan_data = []
        walk_node = node
        scan_nodes = []          # 扫描覆盖的 6 个节点 → 喂给 2D 探索地图
        gif_frames = []
        directions = [
            ('0°', 0), ('60°', 60), ('120°', 120),
            ('180°', 180), ('240°', 240), ('300°', 300),
        ]
        # Each 60° = 2 rotate_cw edges (~30° each)
        ROTATE_STEPS = 2

        for direction, angle in directions:
            scan_nodes.append(walk_node)
            # 加载照片 — 路径顺序必须与导航流程 (_inject_photo/_load_photo)
            # 一致: /mnt/d 原版优先。此前这里用 /home/tao_h 副本优先, 两个
            # 目录同名文件内容不同 (md5 不一致), 导致扫描 0° 照片与当前
            # step 照片不一样 (warmup/rescan 均受影响)。
            img = self._load_photo(walk_node)
            photo_path = None
            for d in ['/mnt/d/aod/AOD_dataset/Home_001_1/jpg_rgb',
                      '/home/tao_h/avdb_habitat_converter/data/processed/rgb']:
                p = os.path.join(d, walk_node)
                if os.path.exists(p):
                    photo_path = p
                    break
            if img is None:
                walk_node = self._walk_graph(walk_node, 'rotate_cw', ROTATE_STEPS)
                continue

            yolo_info = None
            yolo_found_target = False
            has_yolo = hasattr(self.simWrapper, 'yolo')
            if not has_yolo:
                logging.error(f'[WARMUP] simWrapper has NO yolo attribute!')
            elif not target:
                logging.error(f'[WARMUP] target is empty!')
            elif not photo_path:
                logging.error(f'[WARMUP] photo_path is None for {direction}!')
            else:
                try:
                    # 检测器变体分派 (扫描与逐步同口径):
                    # gt_detect 优先 (上限行) → direct_vlm (VLM grounding) → YOLO
                    _gt = self.simWrapper.ideal_detect(walk_node, target)
                    if _gt is not None:
                        found, info = _gt
                    elif (getattr(self, '_ff', None) or {}).get('direct_vlm'):
                        found, info = self._vlm_grounding_detect(
                            np.array(img), target)
                    else:
                        found, info = self.simWrapper.yolo.check_target(photo_path, target)
                    # 黑名单拦截层3: 已排除节点 (conf<0.9) 的重复误报
                    # 不进扫描结果 — 否则周期 rescan 会在同一个坑反复锁死
                    if found and info.get('confidence', 0) < 0.9 \
                            and hasattr(self.simWrapper, '_node_in_rejected') \
                            and self.simWrapper._node_in_rejected(walk_node)[0]:
                        found = False
                        yolo_info = dict(info)
                        yolo_info['target_found'] = False
                        logging.info(f'[BLACKLIST] scan detection at {walk_node} '
                                     f'suppressed (rescan would re-lock false lead)')
                    yolo_found_target = found
                    if yolo_info is None:
                        yolo_info = info
                except Exception as e:
                    logging.warning(f'[WARMUP] YOLO error at {direction}: {e}')

            # Draw YOLO bounding boxes on the photo
            annotated = img.copy()
            draw = ImageDraw.Draw(annotated)
            all_dets = yolo_info.get('all_detections', []) if yolo_info else []
            logging.info(f'[WARMUP] {direction}: node={walk_node} photo={photo_path} YOLO={len(all_dets)}objs target={yolo_found_target}')

            for det in all_dets:
                x1, y1, x2, y2 = [int(v * s) for v, s in
                    zip(det['bbox_norm'], [img.width, img.height, img.width, img.height])]
                cls_name = det['class_name']
                conf = det['confidence']
                is_target = (cls_name == target)

                color = (0, 255, 0) if is_target else (255, 60, 60)
                # Thicker outline
                draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
                # Semi-transparent white background for label
                label = f"★TARGET {cls_name} {conf:.0%}" if is_target else f"{cls_name} {conf:.0%}"
                tw = len(label) * 7 + 4
                draw.rectangle([x1, y1-22, x1+tw, y1], fill=(40, 40, 40))
                draw.rectangle([x1, y1-22, x1+tw, y1], outline=color, width=2)
                draw.text((x1+2, y1-20), label, fill=color)

            # Direction label on image
            draw.rectangle([(0, 0), (img.width, 30)], fill=(0, 0, 0))
            status = '★★★ TARGET FOUND! ★★★' if yolo_found_target else 'scanning...'
            draw.text((5, 4), f"{direction} ({angle}deg) | {status}", fill=(255, 255, 255))
            draw.text((5, 16), f"Target: {target}", fill=(200, 200, 200))

            gif_frames.append(annotated)
            scan_data.append({
                'direction': direction, 'angle': angle,
                'yolo_found': yolo_found_target, 'yolo_info': yolo_info,
                'img': img,   # 原图 (无标注) — 修3A 单图证据门验证用
                'node': walk_node,   # #37: 该朝向照片的站位节点 (目击登记用)
            })

            # Next 60° = 2 rotate_cw edges
            walk_node = self._walk_graph(walk_node, 'rotate_cw', ROTATE_STEPS)

        # === 扫描记忆 → 深度探索图 (360° 熟悉环境, 不只是找目标) ===
        # 6 个朝向的地面参数/可行走射线全部写入 ExplorationMap2D:
        # 遮挡阴影、前沿、未访问方向从此拥有开局全景视野。
        try:
            self.simWrapper.integrate_scan_into_map(scan_nodes)
        except Exception as e:
            logging.debug(f'[SCAN-MAP] warmup integrate failed: {e}')
        # === live 深度也写入 (信息不浪费): 6 个扫描视角逐个传送渲染 ===
        # 照片射线有 1.8m 盲区; live 传感器无盲区, 覆盖画在其上。
        # warmup 与途中 rescan 共用本函数 → 两处都生效。
        try:
            sw = self.simWrapper
            n_live = 0
            for wn in scan_nodes:
                got = sw.render_live_at(wn)
                if got is None:
                    continue
                quat, dimg = got
                wpn = sw.nav_graph['nodes'][wn]['world_pos']
                n_live += sw.exploration_map.update_live(
                    wpn, quat, dimg, fov_deg=sw.fov,
                    pitch=sw.sensor_pitch, sensor_h=sw.sensor_height)
            if n_live:
                logging.info(f'[SCAN-MAP] live depth: {n_live} rays from '
                             f'{len(scan_nodes)} scan views')
        except Exception as e:
            logging.debug(f'[SCAN-MAP] live integrate failed: {e}')

        # Save scan frames as PNG + step directories for GIF inclusion
        # (目录带 step_tag — 初始 warmup 与 rescan 互不覆盖, 见 docstring)
        ep_dir = f'logs/{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'
        if gif_frames:
            scan_dir = f'{ep_dir}/{step_tag}'
            os.makedirs(scan_dir, exist_ok=True)
            for fi, frame in enumerate(gif_frames):
                frame.save(os.path.join(scan_dir, f'scan_{fi:02d}.png'))
                # Also save as step-like dirs for main GIF inclusion
                step_dir = f'{ep_dir}/{step_tag}_{fi:02d}'
                os.makedirs(step_dir, exist_ok=True)
                frame.save(os.path.join(step_dir, 'color_sensor.png'))
            # GIF
            gif_path = f'{scan_dir}/{step_tag}.gif'
            gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                              duration=1000, loop=0, optimize=False)
            logging.info(f'[WARMUP] Scan: {scan_dir}/ + step dirs for GIF')

        # Check if YOLO found the target
        yolo_hits = [d for d in scan_data if d['yolo_found']]
        target_priority = ""
        if yolo_hits:
            best = max(yolo_hits, key=lambda d: d['yolo_info'].get('confidence', 0))
            conf = best['yolo_info'].get('confidence', 0)
            direction = best['direction']
            angle = best['angle']
            # === #47b (bmv10 实弹): 目击框画面内偏移 — 照片相机朝向 ≠ 目标方位 ===
            # bmv10 step42-43: 目击 middle_right, GT 相对相机 +41° — 旧锁
            # 公式漏掉这一段 (锁 304°, 真 344.9°≡−15.1°)。偏移由目标框 cx
            # 反解 (照片 HFOV≈89°, 1920×1080 数据集渲染 f≈975px — 相机内参
            # 属传感器规格, 零样本合规)。真实图回放两病例修正后误差 ≤1.5°。
            self._scan_hit_offset_deg = self._yolo_frame_offset_deg(
                best.get('yolo_info'), target)
            # === #53b (d01r2 实弹): 扫描目击框面积同记 — 扫描锁的 miss
            # 预算分档依据 (远距 area<0.10 → 预算 6; d01r2 锁 −124° 转身
            # 需 4+ 步, 默认 _approach_last_area=1.0 吃 3 步预算 → 释锁
            # → #37 拉回重锁无限循环) ===
            _yi = best.get('yolo_info') or {}
            _area = _yi.get('area_ratio')
            if _area is None:
                for _d in (_yi.get('all_detections') or []):
                    if _d.get('class_name') == target and _d.get('bbox_norm'):
                        _x1, _y1, _x2, _y2 = [float(v)
                                              for v in _d['bbox_norm']]
                        _area = abs(_x2 - _x1) * abs(_y2 - _y1)
                        break
            self._scan_hit_area = float(_area) if _area else None
            # === #37 (bmv6 实弹): 扫描命中登记为目击 ===
            # bmv6: 3 次 APPROACH-LOCK 全来自六图扫描 YOLO 命中 (conf
            # 0.51/0.56/0.56), 但扫描命中只产 target_priority 文本 — 从未
            # 进 target_sighting_nodes (wrapper 只登记活体帧) → 接近失检
            # 释锁后证据全部蒸发, "确曾看见"变成从没看见。这里按 wrapper
            # 同构格式登记 (含 ③c 配对帧), #31 目击史守卫从此也认扫描目击。
            if (getattr(self, '_ff', None) or {}).get('scan_sighting', True) \
                    and best.get('node'):
                sw_s = self.simWrapper
                _desc = f"YOLO scan hit {angle}deg, conf={conf:.2f}"
                sw_s.target_sighting_nodes = \
                    (getattr(sw_s, 'target_sighting_nodes', None) or []) + [
                        (self.step, best['node'], _desc)]
                if len(sw_s.target_sighting_nodes) > 5:
                    sw_s.target_sighting_nodes = sw_s.target_sighting_nodes[-5:]
                if not hasattr(self, '_sighting_frames'):
                    self._sighting_frames = {}
                self._sighting_frames[best['node']] = best.get('img')
                if hasattr(sw_s, '_record_detection'):
                    sw_s._record_detection(best['node'], _desc)
                logging.info(f'[SCAN-SIGHTING] {best["node"]} (scan {angle}°, '
                             f'conf {conf:.2f}) registered — evidence survives '
                             f'approach loss (#37)')
            # === 任务三: warmup 反应式抢占 — 强信号期 Fast 压制 Slow ===
            # 强证据 → 跳过下方 VLM 环境分析调用 (零 VLM 开销的开局先验);
            # 方位经 caller 桥接进接近锁后, 方案3 接近模式本身不问 VLM —
            # AUTO-STEER 离散 P 控制 (|e|>15° 转 ±30°, 否则前进) 直接逼近;
            # 滞回双阈值由接近锁现成提供 (re-arm 0.5 / 延续 0.3 / 连续
            # 丢失 >10 步交还语义层)
            preempted, preempt_msg = self._warmup_preempt_check(best, target)
            if preempted:
                summary = (
                    f"\n## 🌐 INITIAL ENVIRONMENT SCAN (360°)\n"
                    f"6 photos at 60° intervals (0°/60°/120°/180°/240°/300°). YOLO scanned each.\n"
                    f"**Metric preempt engaged** — strong YOLO evidence, no VLM analysis needed.\n"
                )
                return summary + preempt_msg, preempt_msg
            # Compute exact rotation steps needed to face the target direction
            # Each rotate_cw edge is ~30°. 60° = 2 edges, 90° = 3 edges, etc.
            rot_steps = angle // 30  # number of 30° rotate edges needed
            if rot_steps <= 6:
                rot_actions = f'turn RIGHT {rot_steps}x ({angle}° total)'
            else:
                rot_steps = (360 - angle) // 30
                rot_actions = f'turn LEFT {rot_steps}x ({360-angle}° total)'

            target_priority = (
                f"\n## 🎯 YOLO FOUND TARGET DURING SCAN!\n"
                f"YOLO detected '{target}' at **{direction} ({angle}°)** with confidence **{conf:.0%}**.\n"
                f"★ YOUR FIRST ACTION: {rot_actions}, then FORWARD toward it!\n"
                f"★ DO NOT explore elsewhere — GO DIRECTLY to this direction!\n"
                f"★ This is a CONFIRMED sighting by YOLO. Highest priority!\n"
            )
            logging.info(f'[WARMUP] YOLO found target at {direction} ({angle}deg), conf={conf:.2f}')
        else:
            logging.info('[WARMUP] YOLO did not find target during scan')

        # VLM environment analysis with the 4 photos
        try:
            # P6 (语义问询层): 覆盖检查进 prompt — 用户思路 "让 VLM 判断
            # 这个区域是否已扫尽、有没有可疑位置"; 覆盖摘要给跨扫描记忆
            _p6 = (getattr(self, '_ff', None) or {}).get('scan_inquiry', True)
            cov_note = self._scan_coverage_note() if _p6 else ''
            prompt = (
                f"You are at the starting position. Target: {target}.\n"
                f"Below are 6 photos at 60° intervals (0°/60°/120°/180°/240°/300°).\n\n"
                f"Analyze:\n"
                f"1. What ROOMS in each direction?\n"
                f"2. Where is '{target}' most likely located? At which angle?\n"
                f"3. Output exactly: FIRST_DIRECTION: <angle>°\n"
                f"   (pick one of: 0°, 60°, 120°, 180°, 240°, 300°)\n"
                f"4. Coverage check: look carefully at ALL 6 photos for a "
                f"partially visible or hidden '{target}' (shelves, counter "
                f"edges, behind clutter).\n"
                f"   - If some photo has a suspicious spot worth a closer "
                f"look, output exactly: SCAN_SUSPICIOUS: <angle>°\n"
                f"   - If the full 360° view is checked and nothing is "
                f"suspicious, output exactly: SCAN_CLEAR\n"
                f"5. Exploration lead: which direction most likely LEADS "
                f"TOWARD rooms/surfaces where a {target} is typically kept "
                f"(kitchen counters, pantry, shelves)? Look for doorways, "
                f"hallways, counter runs in each photo. Output exactly: "
                f"LIKELY-BEARING: <angle>° (one of 0°/60°/120°/180°/"
                f"240°/300°). This is a search preference, not a sighting.\n"
                f"{cov_note}"
                f"Be concise. The FIRST_DIRECTION, SCAN_ and LIKELY-BEARING lines are required."
            )
            imgs = [np.array(f) for f in gif_frames]
            # #26c: plain_text=True — 六图扫描协议要规定行 (FIRST_DIRECTION/
            # SCAN_SUSPICIOUS), JSON-only 系统指令会压制其输出 (p16 实弹
            # 三局 SCAN_ 行零解析的根因)。
            # A②/A③: 改走 _vlm_call 统一通道 (分类/重试/判局; plain_text
            # 在通道内固定 True) — p26 型 18 崩从此 step0 止损
            response = self._vlm_call(
                prompt, imgs,
                'warmup' if step_tag == 'warmup_scan' else 'rescan')

            # Parse FIRST_DIRECTION from VLM response
            import re
            # === 修3A (ep1 实弹): 方位提取 (规定行 → action 字段兜底) +
            #     证据门 (可见性声明单图证伪 / 推测声明不注入) ===
            first_dir_angle = self._extract_first_direction(response)
            first_dir_angle, response = self._verify_first_direction(
                response, first_dir_angle, scan_data, target)
            # fallback (action 字段) 过门放行后, response 里可能仍无规定行
            # (VLM 格式漂移时) — 补写一行, 下游 _initialize_episode /
            # _re_scan 两处正则才能解析注入 (+20 bonus / 自动转向)
            if first_dir_angle is not None and not re.search(
                    r'FIRST_DIRECTION:\s*\d+°', response):
                response += f'\nFIRST_DIRECTION: {first_dir_angle}°'
            # === P6: SCAN_SUSPICIOUS / SCAN_CLEAR 解析 + 机位与可疑点
            #     登记 (每次扫描都算一个机位 — 判尽门 ≥2 机位的数据源) ===
            sus_angle, scan_note = self._parse_scan_inquiry(response, target)

            summary = (
                f"\n## 🌐 INITIAL ENVIRONMENT SCAN (360°)\n"
                f"6 photos at 60° intervals (0°/60°/120°/180°/240°/300°). YOLO scanned each.\n"
                f"Each photo's direction = that many degrees CLOCKWISE from your current view.\n\n"
                f"**VLM Analysis:**\n{response}\n"
            )
            if target_priority:
                summary += target_priority
            elif first_dir_angle is not None:
                n_rotates = first_dir_angle // 30
                if first_dir_angle <= 180:
                    turn_hint = f"pick ROTATE RIGHT x{n_rotates} ({n_rotates*30}° total)"
                else:
                    left_steps = (360 - first_dir_angle) // 30
                    turn_hint = f"pick ROTATE LEFT x{left_steps} ({left_steps*30}° total)"
                summary += (
                    f"\n## 🎯 CRITICAL: Warmup recommends **{first_dir_angle}°** as FIRST direction!\n"
                    f"   The 360° scan identified the target is most likely at this angle.\n"
                    f"   → In STEP 4 scoring, give actions toward {first_dir_angle}° a **+20 BONUS**!\n"
                    f"   → To face {first_dir_angle}°: {turn_hint}.\n"
                    f"   → After turning, pick FORWARD to approach and CONFIRM the target.\n"
                    f"   → This is your #1 PRIORITY. Do NOT explore elsewhere first!\n"
                )
            # === P2 (修6): MUST-DO 条件化 — 方位被证据门拒 (first_dir_angle
            #     is None: 可见性声明被证伪 / 推测声明不注入) 且 YOLO 也没
            #     发现时, "MUST-DO NOW" 是拿猜测当令箭 (ep6: 门拒 60° 后
            #     MUST-DO 残留 → VLM 反复往厨房推测方向走)。改推测注记。 ===
            if (getattr(self, '_ff', None) or {}).get('mustdo_gate', True) \
                    and not target_priority and first_dir_angle is None:
                summary += (
                    f"\n**NOTE: No verified sighting direction from this scan "
                    f"(the angle above is an unverified guess). Explore by your "
                    f"own judgment — do NOT treat it as a MUST.**\n"
                )
                logging.info('[WARMUP] first-direction INCONCLUSIVE → '
                             'MUST-DO withheld (speculative note only)')
            else:
                summary += (
                    f"\n**MUST-DO: Go toward the warmup-identified direction NOW.**\n"
                )
            # === P6: 问询结果注入 — YOLO 已发现目标 (target_priority) 时不
            #     注入: 硬证据直行令与"先查可疑/去新区"互相矛盾, 硬证据赢 ===
            if scan_note and not target_priority:
                summary += scan_note
            logging.info(f'[WARMUP] Analysis complete ({len(response)} chars)')
            return summary, target_priority

        except Exception as e:
            logging.warning(f'[WARMUP] VLM analysis failed: {e}')
            if target_priority:
                return target_priority, target_priority
            return "", None

    def _extract_first_direction(self, response):
        """修3A 配套: 方位提取 — 规定行优先, action 字段兜底

        ep6 复测实弹: VLM 未输出规定的 FIRST_DIRECTION 行, 而是把角度写进
        JSON 的 "action": 60 (格式漂移) — 方位信息锁死在 reasoning 里,
        正则匹配不到。兜底: action 值恰为六合法方位之一时提取 (同样过
        _verify_first_direction 证据门, 不直接采信)。
        """
        m = re.search(r'FIRST_DIRECTION:\s*(\d+)°', response)
        if m:
            return int(m.group(1))
        ma = re.search(r'"action"\s*:\s*(\d+)', response)
        if ma and int(ma.group(1)) in (0, 60, 120, 180, 240, 300):
            v = int(ma.group(1))
            logging.info(f'[WARMUP] FIRST_DIRECTION line missing → fallback '
                         f'action-field angle {v}° (format drift, ep6)')
            return v
        return None

    def _scan_coverage_note(self):
        """P6 覆盖摘要 (进 scan prompt): 本 zone 已扫机位数 + 未查可疑方位

        表示口径守红线: zone 栅格标签 + 扫描方位角, 无节点 ID/文件名。
        VLM 拿它回答"这个区域是否已扫尽"时拥有跨扫描记忆 (第几次从不同
        机位扫这里、之前报过哪些未查可疑方位), 而不是每次都从零判断。
        """
        try:
            sw = self.simWrapper
            cell = sw._area_key(sw.current_node)
            n_pos = len((getattr(self, '_scan_positions_by_cell', None) or {})
                        .get(cell, ()))
            sus = (getattr(self, '_suspicious_spots', None) or {}).get(cell, [])
            note = (f"\nCoverage memory: you are in zone {cell[0]}_{cell[1]}; "
                    f"this zone has been 360°-scanned from {n_pos} distinct "
                    f"position(s) so far.")
            if sus:
                angs = ', '.join(f'{a}°' for _, a in sus[-2:])
                note += (f" Previously flagged suspicious directions in this "
                         f"zone (still unresolved): {angs}.")
            return note + '\n'
        except Exception:
            return ''

    def _parse_scan_inquiry(self, response, target):
        """P6 (修9): 解析 SCAN_SUSPICIOUS:<angle>° / SCAN_CLEAR + 状态登记

        用户思路: "让 VLM 判断这个区域是否已扫尽、有没有可疑位置 — 都
        没有 → 去找出口到另一个区域; 有 → 去看看没看到的地方。避免
        灯下黑, 也可以换个机位再扫一次"。落到状态:
          每次扫描 (含 warmup/周期/释锁/P6 触发) → 当前节点记为该 zone
            的扫描机位 (_scan_positions_by_cell, 判尽门 ≥2 机位的数据源);
          SCAN_SUSPICIOUS → 该 zone 挂一个待查可疑方位 (≤2/区域, 旧的
            先出; 下次扫描的覆盖摘要回显, 查过才消);
          SCAN_CLEAR → 该 zone 可疑点清空 (回写: 查过无藏匿)。
        返回 (sus_angle|None, note); note 由 caller 决定是否注入 summary。
        """
        if not (getattr(self, '_ff', None) or {}).get('scan_inquiry', True):
            return None, ''
        cell = None
        try:
            sw = self.simWrapper
            cell = sw._area_key(sw.current_node)
            if not hasattr(self, '_scan_positions_by_cell'):
                self._scan_positions_by_cell = {}
            self._scan_positions_by_cell.setdefault(cell, set()).add(
                sw.current_node)
        except Exception:
            cell = None
        m_s = re.search(r'SCAN_SUSPICIOUS:\s*(\d+)', response or '')
        m_clear = re.search(r'SCAN_CLEAR', response or '')
        # === #32 (bmv2 mahatma 实弹, 用户指令"困惑就去更可能放目标物品
        #     的方位"): LIKELY-BEARING → 世界方位。与 FIRST_DIRECTION 的
        #     ep6 教训不冲突 — 推测方位不作转向命令/不加 bonus, 只作
        #     前沿跳跃的目标偏好 (落点仍是图验证过的未到访格)。 ===
        m_lb = re.search(r'LIKELY-BEARING:\s*(\d{1,3})', response or '')
        if m_lb:
            try:
                self._semantic_bearing_deg = float(np.degrees(
                    self._yaw_world())) + float(m_lb.group(1)) % 360
                self._semantic_bearing_step = getattr(self, 'step', 0)
                # #38: 同时进投票聚合 (单值仍保留供消融回退)
                self._register_semantic_vote(
                    self._semantic_bearing_deg,
                    f'LIKELY-BEARING {int(m_lb.group(1))}°')
                logging.info(f'[SCAN-INQUIRY] LIKELY-BEARING '
                             f'{int(m_lb.group(1))}° → world '
                             f'{self._semantic_bearing_deg:.0f}° '
                             f'(探索去向偏好, 非目击)')
            except Exception:
                pass
        # === #26c 兜底: VLM 回成 JSON 时规定行不在 (plain_text 根因修复
        #     之外再防格式漂移): ① JSON 字段 scan_suspicious/scan_clear
        #     ② 自由文本 CLEAR 关键词。SUSPICIOUS 只认显式角度 — 误报
        #     会把机器人拉去看空气, 宁漏勿误 (漏由机位门多扫兜底) ===
        try:
            m_j = re.search(r'\{.*"scan_[a-z]+"\s*:.*\}', response or '', re.S)
            if m_j:
                j = json.loads(m_j.group(0))
                v = j.get('scan_suspicious')
                if m_s is None and v is not None \
                        and re.fullmatch(r'\d{1,3}', str(v).strip()):
                    m_s = re.match(r'(\d{1,3})', str(v).strip())
                if j.get('scan_clear'):
                    m_clear = m_clear or True
        except Exception:
            pass
        if m_s is None and not m_clear:
            m_clear = re.search(
                r'nothing\s+(?:is\s+)?suspicious|no\s+suspicious|'
                r'nothing\s+worth|all\s+clear', response or '', re.I)
        note = ''
        if m_s:
            ang = int(m_s.group(1)) % 360
            if cell is not None:
                spots = getattr(self, '_suspicious_spots', None) or {}
                # #33 (bmv3 mahatma 实弹, 用户"还是原地打转"): 查过销账 —
                #   该方位已转过看过且无 YOLO 检出 → 不再排队 (VLM 每次
                #   六图都在沙发/电视同一位置再找"可疑点", 转去看→再扫→
                #   再可疑 死循环, zone 永远 CLEAR 不了 → 离区通道被堵)。
                #   可疑点查尽 (队列空) = 事实 CLEAR → 登记离区链路接通。
                insp = (getattr(self, '_inspected_spots', None) or {}).get(
                    cell, set())
                if ang // 60 in insp:
                    logging.info(f'[SCAN-INQUIRY] zone {cell}: SUSPICIOUS '
                                 f'at {ang}° 已查过 (转过看过无检出) → '
                                 f'销账不再排队')
                    m_s = None          # 不回传角度/不注入可疑 note
                    # #34 (bmv4 实弹): 同桶旧排队条目同步出队 — 只挡新条目
                    #   不清旧条目 → 队列永非空 → 事实 CLEAR 永不触发
                    #   (bmv4: 5 次销账 0 次 CLEAR, 0 次离区)
                    _q = [e for e in (spots.get(cell) or [])
                          if e[1] // 60 != ang // 60]
                    if hasattr(self, '_suspicious_spots'):
                        self._suspicious_spots[cell] = _q
                    if not _q:
                        # 队列空 + 该判词唯一可疑点已查 → 事实 CLEAR
                        m_clear = True
                else:
                    # ≤2/区域: 保留最旧一条 + 新一条 (旧可疑点优先查)
                    lst = spots.get(cell, [])[-1:] + [(getattr(self, 'step', 0), ang)]
                    if not hasattr(self, '_suspicious_spots'):
                        self._suspicious_spots = {}
                    self._suspicious_spots[cell] = lst
                    logging.info(f'[SCAN-INQUIRY] zone {cell}: SUSPICIOUS at '
                                 f'{ang}° (spot queued, {len(lst)}/2)')
            if m_s:
                note = (f"\n## 🔍 SCAN INQUIRY: suspicious spot for '{target}' "
                        f"at {ang}° in this zone — turn and inspect that "
                        f"direction BEFORE leaving the zone!\n")
        # #33: 上面的销账分支置 m_s=None + m_clear=True → 此处接手登记
        # (elif 不会回头执行; 守卫保持"真 SUSPICIOUS 优先于 CLEAR"原语义)
        if m_clear and not m_s:
            if cell is not None:
                if not hasattr(self, '_suspicious_spots'):
                    self._suspicious_spots = {}
                self._suspicious_spots[cell] = []
                # 扫描机制 v2 (用户指令): 判 CLEAR = 无价值区 → 登记, 机制
                # 强制第一时间离开 (不再只靠 prompt 注记 — 实测 VLM 无视
                # "move on" 继续原地探索, mahatma ab 局 step20 扫后仍磨 30 步)
                if not hasattr(self, '_zone_clear_cells'):
                    self._zone_clear_cells = set()
                # #31 (coca-ab 实弹, 2026-09-13): 有目击史的 zone 不登记
                #   CLEAR。六图扫描判词是"没看出来", 打不过自身 YOLO/VLM
                #   目击记录 ("确曾看见") — coca 出生 6cm: warmup 台面瓶
                #   YOLO 61% + 接近期一路目击, step11 六图重扫 VLM 没认出
                #   远处小瓶 → 误 CLEAR → v2 三次强制离区拖到 1.98m fp。
                #   目击史 = 机器人自身数据, 零样本合规。真空区 (mahatma
                #   6 次 CLEAR) 无目击记录, 守卫不误伤离区机制。
                _sight_cells = set()
                if (getattr(self, '_ff', None) or {}).get(
                        'zone_clear_sight_guard', True):
                    for _st, _node, _d in (
                            getattr(getattr(self, 'simWrapper', None),
                                    'target_sighting_nodes', None) or []):
                        try:
                            _sight_cells.add(
                                self.simWrapper._area_key(_node))
                        except Exception:
                            pass
                if cell in _sight_cells:
                    logging.info(
                        f'[ZONE-CLEAR] zone {cell}: 有目标目击史 → '
                        f'不登记 CLEAR (YOLO 目击 > VLM 判词, 保留搜索)')
                else:
                    self._zone_clear_cells.add(cell)
                    logging.info(f'[SCAN-INQUIRY] zone {cell}: CLEAR '
                                 f'(queued suspicious spots resolved) → '
                                 f'无价值区, 将强制离区')
            note = ("\n## ✅ SCAN INQUIRY: this zone looks fully scanned "
                    "with nothing suspicious — move on to a NEW zone "
                    "(unvisited direction / exit) next!\n")
            if getattr(self, '_semantic_bearing_deg', None) is not None:
                note += (f"   → Prefer the direction toward "
                         f"~{int(self._semantic_bearing_deg) % 360}° "
                         f"(world) — scan judged it most likely to LEAD to "
                         f"{target}-typical areas. Keep moving that way!\n")
        return (int(m_s.group(1)) % 360) if m_s else None, note

    def _in_cleared_zone(self, obs):
        """扫描机制 v2: 当前是否站在已判 CLEAR 的 zone

        证据撤销: 该 zone 出现 YOLO 活检 (新证据 > 旧判词) → CLEAR 作废,
        正常追击。零样本合规 — 判据是自身检测与扫描问询, 无真值。
        """
        cleared = getattr(self, '_zone_clear_cells', None) or set()
        if not cleared:
            return False
        try:
            cell = self.simWrapper._area_key(self.simWrapper.current_node)
        except Exception:
            return False
        if cell in cleared and (obs.get('yolo_detection') or {}).get('target_found'):
            cleared.discard(cell)
            logging.info(f'[ZONE-CLEAR] zone {cell}: YOLO 活检 → 撤销 CLEAR '
                         f'(新证据优先, 恢复正常搜索)')
            return False
        return cell in cleared

    def _zone_neg_strike(self):
        """#51b (bmd01 实弹): 目击被身份检查否决 → 所在 zone 记否决一击

        d01 coca 死循环: 真瓶 4 次被 IDENTITY no 放走 (远距小框看不清),
        目击区因 #31 守卫永不 CLEAR, 记忆提示把机器人拽回再看再否 —
        用户: "不停地去探索同一个区域, 是要反复确认啥??"。否决 ≥2 击
        → 该 zone 进 _zone_negative_zones, 前沿/探索选点时降到底层
        (#51c 回访门: 其他 zone 都确认完才可再进)。数据源 = 自身身份
        检查判决 + 里程计分区, 零样本合规。
        """
        try:
            cell = self.simWrapper._area_key(self.simWrapper.current_node)
        except Exception:
            return
        self._zone_negative_strike = {
            **getattr(self, '_zone_negative_strike', {}), cell:
            getattr(self, '_zone_negative_strike', {}).get(cell, 0) + 1}
        n = self._zone_negative_strike[cell]
        if n >= 2:
            zs = getattr(self, '_zone_negative_zones', None)
            if zs is None:
                zs = set()
                self._zone_negative_zones = zs
            zs.add(cell)
        logging.info(f'[ZONE-NEG] zone {cell}: identity-reject strike #{n}'
                     + (' → 否决区 (回访门: 其他 zone 优先)' if n >= 2 else ''))

    def _inject_zone_caution(self, obs):
        """#51b (bmd01 实弹): 每步 prompt 注入"已否决勿回"警示

        VLM 推理一直追 "prior memory indicates sightings in zone_X",
        把机器人拽回已否决的区域 (记忆提示只带"看见过"不带"查过了不
        是")。补一行已否决 zone 清单, 记忆提示不再无限拉回。数据源 =
        自身身份判决 (_zone_negative_zones), 零样本合规; zone_revisit_
        gate 消融关 → 不注入。
        """
        _negz = getattr(self, '_zone_negative_zones', None) or set()
        if _negz and (getattr(self, '_ff', None) or {}).get(
                'zone_revisit_gate', True):
            _zs = ', '.join(f'zone_{z[0]}_{z[1]}'
                            for z in sorted(_negz)[:4])
            obs['memory_context'] = (obs.get('memory_context') or '') + (
                '\n⚠ ALREADY CHECKED & REJECTED zones (identity-check said '
                'not the target): ' + _zs + ' — do NOT go back to re-search '
                'them unless every other zone is exhausted; explore NEW '
                'zones instead.\n')

    def _verify_first_direction(self, response, angle, scan_data, target):
        """修3A (ep1 实弹): FIRST_DIRECTION 单图证据门 — 声明类型三态

        六图联合分析两种声明, 可信度天差地别:
        ① 可见性声明 "目标 is visible in the X° image" (ep1): 声称亲眼
          看到 → 可被单图证伪。ep1 实弹: VLM 正确描述了场景 ("TV 旁桌上
          的可乐瓶" — 实际在 120° 照片) 却归属到 240° (该照片全是百叶
          窗)。单图重问 — 没有多图就没有归属混淆: present=0 → 作废
          (response 的 FIRST_DIRECTION 行同步改写, 下游两处正则均解析
          不到 → +20 bonus 与自动动作不注入); present=1 → 注入。
        ② 推测声明 "likely located in kitchen at X°" (ep6 实弹): 无验证
          手段 — 且 ep6 真方位 3.8° 正前, kitchen 推测 60° 差 56°,
          注入猜测 = 放大错误 → 一律不注入 (分析文本保留在 memory 供
          VLM 每步参考, 信息不丢只是不放大成自动动作)。
        声明类型判据: 目标词干 100 字符窗口内有 visible/seen, 且前 60
        字符窗口内无 likely/might/typically 等推测限定词 → ①; 否则 ②。
        成本: ①+1 次 VLM 调用 (仅 YOLO 未发现时); ②零成本。
        返回 (angle, response)。
        """
        if angle is None or not scan_data:
            return angle, response
        if not (getattr(self, '_ff', None) or {}).get('first_dir_gate', True):
            return angle, response   # 消融: 旧行为 (直接采信六图分析)
        # --- 声明类型判定 ---
        # 注意 (?:...) 分组: 裸交替 'a|b|c'+窗口 会因连接优先于交替, 让
        # 窗口约束只挂在最后一个词干上 (前几个词干全文裸匹配)
        stems = [w for w in re.split(r'[ _]+', target) if len(w) >= 4][:4]
        pat = '(?:' + '|'.join(re.escape(s) for s in stems) + ')' if stems \
            else '(' + re.escape(target) + ')'
        seg_m = re.search(pat + r'[^.\n]{0,100}?(?:visible|seen)', response,
                          re.I)
        hedge_m = re.search(pat + r'[^.\n]{0,60}?'
                            r'\b(?:likely|probab\w*|typical\w*|might|may|'
                            r'could|perhaps|usually|often)\b', response,
                            re.I)
        if seg_m is None or hedge_m is not None:
            logging.info(f'[FIRST-DIR-GATE] {angle}° is a LIKELIHOOD claim '
                         f'(no falsifiable visibility statement) → guess is '
                         f'not actionable, skip auto-steer (ep6 lesson)')
            # === #38 (bmv6 实弹, 用户拍板"分类处理"): 推测声明不再一杀了之 ===
            # bmv6: VLM 两次指厨房 (warmup 240° + step4 240°) 都死在这里,
            # 而没过门的单次 LIKELY-BEARING 180° 反而拿了 11 边承诺权。
            # 推测声明降级进语义先验通道 (投票聚合; 无 +20 bonus / 无自动
            # 转向 / 不算 sighting, ep6 红线不破 — 只是"探索往哪边走"的票)。
            if (getattr(self, '_ff', None) or {}).get('semantic_prior', True):
                try:
                    self._register_semantic_vote(
                        float(np.degrees(self._yaw_world())) + float(angle),
                        f'FIRST_DIRECTION likelihood {angle}°')
                except Exception:
                    pass
            response = re.sub(
                r'FIRST_DIRECTION:\s*\d+°',
                f'FIRST_DIRECTION: REJECTED-{angle}deg '
                f'(likelihood claim → downgraded to exploration prior, '
                f'not a sighting)', response)
            return None, response
        # --- 可见性声明 → 单图证伪 ---
        frame = next((d for d in scan_data if d.get('angle') == angle), None)
        if frame is None or frame.get('img') is None:
            return angle, response   # 无对应帧/无原图 → 无法验证, 保守放行
        if frame.get('yolo_found'):
            return angle, response   # YOLO 已在该帧检出 → 走 target_priority
        try:
            found_v, _info = self._vlm_grounding_detect(
                np.array(frame['img']), target)
        except Exception as e:
            logging.warning(f'[FIRST-DIR-GATE] verify failed: {e} → keep hint')
            return angle, response
        if found_v:
            logging.info(f'[FIRST-DIR-GATE] {angle}° single-image verify: '
                         f'target PRESENT → keep direction hint')
            return angle, response
        logging.info(f'[FIRST-DIR-GATE] {angle}° claim REJECTED — single-image '
                     f'verify: NO target in that photo (cross-photo '
                     f'misattribution, ep1) → drop hint, no +20 bonus')
        response = re.sub(
            r'FIRST_DIRECTION:\s*\d+°',
            f'FIRST_DIRECTION: REJECTED-{angle}deg '
            f'(verified photo has no target)', response)
        return None, response

    def _register_semantic_vote(self, world_deg, source):
        """#38: 语义方位投票 — 跨扫描聚合, 单次噪声不再直接拿承诺权

        bmv6 实弹: 五次重扫五个 LIKELY-BEARING (331/306/37/−32/98/−59°),
        旅程追着最新一次猜 (331°→128°→−15° 互相打架)。票只留最近 30 步
        内的 ≤3 张 (扫描节奏 5-10 步, 旧方向过期)。
        """
        if not (getattr(self, '_ff', None) or {}).get('semantic_prior', True):
            return
        _s = int(getattr(self, 'step', 0))
        votes = list(getattr(self, '_semantic_votes', None) or [])
        votes.append((_s, float(world_deg) % 360.0))
        votes = [v for v in votes if _s - v[0] <= 30][-3:]
        self._semantic_votes = votes
        logging.info(f'[SEMANTIC-VOTE] {source} → world '
                     f'{float(world_deg) % 360:.0f}° (票: '
                     + ', '.join(f'{v[1]:.0f}°' for v in votes) + ')')

    def _semantic_bearing_agg(self):
        """#38: 聚合方位 = 最近 ≤3 票圆均值 + 翻向守卫 → (deg, step) | None

        翻向守卫: 与上次已承诺旅程方向相反 (>120°) 时, 需最近两票一致
        (±60°) 指向新方向才允许翻 — "走过了没找到"的合法换向自然满足
        (连续扫描都会指新方向), 单扫噪声翻不过去。
        消融 (semantic_prior off): 退回 #32 单值行为 (最新一次 LIKELY-
        BEARING 原值)。
        """
        if not (getattr(self, '_ff', None) or {}).get('semantic_prior', True):
            d = getattr(self, '_semantic_bearing_deg', None)
            if d is None:
                return None
            return (float(d) % 360.0,
                    int(getattr(self, '_semantic_bearing_step', 0)))
        votes = getattr(self, '_semantic_votes', None) or []
        if not votes:
            return None
        xs = float(np.mean([np.cos(np.radians(v[1])) for v in votes]))
        ys = float(np.mean([np.sin(np.radians(v[1])) for v in votes]))
        agg = float(np.degrees(np.arctan2(ys, xs))) % 360.0
        lc = getattr(self, '_last_committed_bearing', None)
        if lc is not None \
                and int(getattr(self, 'step', 0)) - int(getattr(
                    self, '_last_committed_step', -99)) <= 40 \
                and abs((agg - float(lc) + 180.0) % 360.0 - 180.0) > 120.0:
            if len(votes) < 2 or abs((votes[-1][1] - votes[-2][1] + 180.0)
                                     % 360.0 - 180.0) > 60.0:
                logging.info(
                    f'[SEMANTIC-VOTE] 聚合 {agg:.0f}° 与上次承诺 '
                    f'{float(lc) % 360:.0f}° 相反且票不一致 → 不翻向 '
                    f'(防 bmv6 331°→128° 乒乓)')
                return None
        return (agg, votes[-1][0])

    def _vlm_call_total(self):
        """双 VLM (action + stopping) 累计成功调用数 — batch runner 效率计量"""
        total = 0
        for v in (getattr(self.agent, 'actionVLM', None),
                  getattr(self.agent, 'stoppingVLM', None)):
            total += int(getattr(v, 'call_count', 0))
        return total

    def _initialize_episode(self, episode_ndx: int):
        self.step = 0; self.init_pos = None; self.df = pd.DataFrame({})
        # batch runner 计量: 差分基线须在 warmup 扫描 (会调 VLM) 之前取
        self._ep_stats = {
            'episode_ndx': episode_ndx,
            'stop_vetoed': 0,     # 动作级: 真 stop 被米制门拦下
            'votes_cleared': 0,   # 投票级: done=1 票被拒清零 (未成 stop)
            'vlm_calls_start': self._vlm_call_total(),
            # 修复2 L2 假阳性监控 (v1.2 第四轮点4): CSV 直读, 不靠人眼看日志
            'fastpath_triggered': 0,    # 快通道停票请求数
            'fastpath_final_close': 0,  # 其中米制门终判 close 的次数
            # A③ (vlm_fail_policy): 关键 VLM 分析失败判局计量 (p26 18 崩事故)
            'vlm_critical_fails': 0,
            'vlm_invalid': 0,
            'vlm_invalid_reason': '',
        }
        # A③ 判局策略状态机 (每局新建; 阈值见 vlm_fail_policy.CONFIG)
        self._vlm_policy = VLMFailPolicy()
        self._vlm_episode_invalid = False
        self._vlm_invalid_reason = ''
        self.agent_distance_traveled = 0; self.prev_agent_position = None
        self._arbitration_close = False  # 停止仲裁结果 (每 episode 复位)
        # D2/③/④ 状态复位 (跨回合不残留)
        self._exhausted_stop = False
        # 修复1: 节点级 wrong-strike 记录 (跨回合不残留)
        self._wrong_strikes = {}
        self._exhausted_streak = 0
        self._crop_calls = 0
        self._crop_cells = set()
        self._last_crop_step = -99
        # 修2B (zero_detect_crop): 零检出连击计数 + 兜底触发节流
        self._no_detect_streak = 0
        self._last_zerodetect_crop = -99
        # P4 (release_rescan): 释锁重扫预算 + pending 复位 (跨回合不残留)
        self._release_rescans = 0
        self._pending_release_rescan = False
        # #26 (viewpoint_rescan): 换机位补扫预算 + 双 pending 复位
        self._viewpoint_rescans = 0
        self._pending_viewpoint_shift = False
        self._pending_viewpoint_rescan = False
        # P6 (scan_inquiry): 扫描机位/可疑点登记复位 (跨回合不残留)
        self._scan_positions_by_cell = {}
        self._suspicious_spots = {}
        self._inspected_spots = {}   # #33: 已转过看过的可疑方位 (60° 桶)
        # #51b (bmd01/d03 实弹, 用户指令"走过的区域要记账"): 区域级记忆 —
        #   扫过 = _scan_positions_by_cell (已有); 否决 = 目击被身份检查
        #   否决过的 zone (strike 计数, ≥2 → 否决区)。回访门 (#51c) 据此
        #   分层: 全新 zone > 扫过未否决 > 否决区 (仅全部其他选项耗尽才回)。
        self._zone_negative_strike = {}
        self._zone_negative_zones = set()
        self._forced_return_dest = None   # #34: 旅程目的地 (重规划/等价边)
        self._reacquire_done = set()      # #37: 已补救过的扫描目击节点 (每节点一次)
        self._semantic_votes = []         # #38: 语义方位票 (跨扫描聚合)
        self._last_committed_bearing = None  # #38: 翻向守卫基准
        self._last_committed_step = -99
        # 阶段1 (frontier_hop / low_conf_reposition): 转圈根治 + 灯下黑
        # 换机位 — 复位触发节流与预算
        self._last_frontier_hop_step = -99
        self._all_nodes = []
        self._reposition_calls = 0
        # 问题1-B (parallax_gate): 逐帧目标框面积历史 (视差测距数据源)
        self._bbox_area_hist = []
        # ③(c) 到场配对确认: 目击帧留存 + 到场检查武装
        self._arrival_check_step = -99
        self._sighting_frames = {}
        self._sight_cap_step = -1
        self._arrival_calls = 0
        self._force_stop_rejected = False
        self.simWrapper = AVDBSimWrapper(self.sim_cfg, nav_graph_path=self.NAV_GRAPH_PATH)
        ep = self.all_episodes[episode_ndx]; self.current_episode = ep
        sp = ep['start_position']; sr = ep['start_rotation']
        self.init_pos = sp
        self.curr_run_name = f"{episode_ndx}_{ep['scene_id'].split('/')[-1].replace('.glb','')}"
        self.simWrapper.set_state(pos=sp, quat=sr)
        self.simWrapper.current_node = self.simWrapper._find_closest_node(sp)
        self.simWrapper.target_name = ep['object_category']
        # GT 检测注入 (表 IV 上限行 SEM-Nav-Ideal): 理想检测器替换 YOLO,
        # 量化"感知完美时导航栈上限"。评估侧距离照常只终局用 (不违规)。
        if (getattr(self, '_ff', None) or {}).get('gt_detect', False):
            self.simWrapper.gt_goal_pos = ep['goals'][0]['position']
            logging.info(f'[FEATURE-FLAGS] gt_detect ON: ideal detector '
                         f'replaces YOLO (goal {ep["goals"][0]["position"]})')
        # direct_vlm: wrapper 侧 YOLO 的目击/状态机登记须让位给 VLM 检测源
        # (否则 YOLO 误报目击残留 → 强制回访把机器人拖回假线索, 污染对照行)
        self.simWrapper.suppress_yolo_sightings = bool(
            (getattr(self, '_ff', None) or {}).get('direct_vlm', False))
        # 修复3 (neutral_rooms): 中性区名进决策链, 旧房间标签仅离线日志
        self.simWrapper.neutral_rooms = bool(
            (getattr(self, '_ff', None) or {}).get('neutral_rooms', True))
        # 修2A (hr_180_cooldown): wrapper 硬规则 1a/1d 掉头冷却注入
        self.simWrapper.hr_180_cooldown = bool(
            (getattr(self, '_ff', None) or {}).get('hr_180_cooldown', True))
        self.simWrapper.hr_180_cooldown_steps = int(
            (getattr(self, '_ff', None) or {}).get('hr_180_cooldown_steps', 12))
        # P3 (修7a): wrapper 硬规则 1b/1c 伺服旁路注入 (miss 由
        # _override_and_run 每步同步 — 见 _in_active_servo)
        self.simWrapper.hr_servo_bypass = bool(
            (getattr(self, '_ff', None) or {}).get('hr_servo_bypass', True))
        # #41: wrapper 多步平移选项生成开关 (走廊通畅才提供链式选项)
        self.simWrapper.journey_stride = bool(
            (getattr(self, '_ff', None) or {}).get('journey_stride', True))
        # #47: wrapper 途中 null 步纯读化 (重定位+A1 只在开局做) —
        # bmv10 两次 APPROACH-LOCK 世界方位 180° 级全错的根因之一
        self.simWrapper.lock_bearing_fix = bool(
            (getattr(self, '_ff', None) or {}).get('lock_bearing_fix', True))
        self.path_calculator.requested_start = sp
        self.path_calculator.requested_ends = [ep['goals'][0]['position']]
        self.curr_shortest_path = self.simWrapper.get_path(self.path_calculator)
        self._ep_stats['initial_geodesic'] = float(self.curr_shortest_path)
        logging.info(f'\n=================== RUN: {self.curr_run_name} ===================')
        logging.info(f'Target: {ep["object_category"]}, Graph dist: {self.curr_shortest_path:.2f}m')

        obs = self.simWrapper.step(PolarAction.null)

        # === Warmup: 360° scan + YOLO + GIF before step 0 ===
        warmup_summary, target_priority = self._warmup_scan()
        if warmup_summary:
            existing_mem = obs.get('memory_context', '')
            obs['memory_context'] = warmup_summary + '\n' + existing_mem
            logging.info('[WARMUP] Injected environment scan into step 0 prompt')

        # === Auto-action: if warmup found a clear direction, pre-compute the action ===
        self._warmup_auto_action = None
        import re

        # Check YOLO priority first, then VLM analysis
        angle_to_use = None
        yolo_warmup_conf = None
        if target_priority:
            angle_match = re.search(r'\((\d+)°\)', target_priority)
            conf_match = re.search(r'confidence \*\*([\d.]+)%', target_priority)
            if angle_match:
                angle_to_use = int(angle_match.group(1))
                yolo_warmup_conf = float(conf_match.group(1)) / 100.0 if conf_match else None

        if angle_to_use is None and warmup_summary:
            fd_match = re.search(r'FIRST_DIRECTION:\s*(\d+)°', warmup_summary)
            if fd_match:
                angle_to_use = int(fd_match.group(1))
                logging.info(f'[WARMUP] FIRST_DIRECTION: {angle_to_use}°')

        # === #38 (bmv6 实弹, 用户拍板"目击后第一优先"): warmup 语义先验即承诺 ===
        # bmv6 病灶: warmup LIKELY-BEARING 240°→world 57° (厨房) 解析完
        # 没有消费方, step1-3 跟 depth 几何往沙发走; step4 重扫翻成 331°
        # 才第一次成行 — 可靠先验被耗死, 单次噪声拿了承诺权。现在:
        # 无 YOLO 目击 (angle_to_use None) + 无未查可疑 → 开局即按聚合
        # 方位成行 (投票含 FIRST_DIRECTION 推测降级票), depth rec 降为
        # 无语义时的兜底。
        _ff_w = getattr(self, '_ff', None) or {}
        if (angle_to_use is None and warmup_summary
                and _ff_w.get('semantic_prior', True)
                and _ff_w.get('semantic_go', True)
                and getattr(self, '_approach_bearing', None) is None
                and not getattr(self, '_forced_return_path', None)):
            agg_w = self._semantic_bearing_agg()
            # #39: 可疑点不再一票否决承诺 — 顺路可疑 (±60°) 让给查看,
            # 不顺路照走 (可疑留 zone 记忆), 与 P6 执行端同一规则
            _sus_m = re.search(r'SCAN_SUSPICIOUS:\s*(\d+)', warmup_summary)
            _sus_blocks = False
            if _sus_m is not None and agg_w is not None \
                    and _ff_w.get('semantic_over_suspicious', True):
                try:
                    _swd = (float(np.degrees(self._yaw_world()))
                            + float(_sus_m.group(1))) % 360.0
                    _sus_blocks = abs((_swd - agg_w[0] + 180.0) % 360.0
                                      - 180.0) <= 60.0
                except Exception:
                    _sus_blocks = False
            elif _sus_m is not None and agg_w is None:
                _sus_blocks = True   # 无语义方向 → 可疑照旧优先 (旧行为)
            if agg_w is not None and not _sus_blocks \
                    and self._plan_bearing_path(
                        np.radians(agg_w[0]), max_edges=8,
                        min_progress=1.5, tag='SEMANTIC-GO'):
                self._last_committed_bearing = float(agg_w[0]) % 360.0
                self._last_committed_step = int(getattr(self, 'step', 0))
                logging.info(
                    f'[SEMANTIC-GO] warmup 语义先验即承诺 (无目击无可疑) → '
                    f'world {float(agg_w[0]) % 360:.0f}° '
                    f'(聚合 {len(getattr(self, "_semantic_votes", []) or [])} 票), '
                    f'旅程 {len(self._forced_return_path)} 边 '
                    f'(#38: 开局不跟 depth 几何走)')

        # 方案3: 预热扫描 YOLO 高置信度发现目标 → 锁定世界方位
        if angle_to_use is not None and yolo_warmup_conf is not None \
                and yolo_warmup_conf >= 0.5:
            if not hasattr(self, '_approach_bearing'):
                self._approach_bearing = None
            if not hasattr(self, '_approach_miss'):
                self._approach_miss = 0
            # #47: 同 rescan — 起始朝向 + 扫描角 + 画面内框偏移 (warmup
            #     时 current_node 本就正确, 偏移是新补的那一段)
            self._approach_bearing = self._scan_lock_bearing_rad(angle_to_use)
            self._approach_miss = 0
            # #53b: 扫描锁补记目击框面积 — 远距锁 (area<0.10) miss 预算
            # 3→6 (d01r2: 3 步预算转身未完即释锁 → REACQUIRE 循环)
            self._approach_last_area = getattr(
                self, '_scan_hit_area', None) or 1.0
            logging.info(f'[APPROACH-LOCK] warmup: angle {angle_to_use}°, '
                         f'frame offset '
                         f'{(getattr(self, "_scan_hit_offset_deg", None) or 0.0):+.0f}°, '
                         f'conf {yolo_warmup_conf:.2f}, world bearing '
                         f'{np.degrees(self._approach_bearing):.0f}° (#47)')

        if angle_to_use is not None and angle_to_use > 0:
            edge_opts = obs.get('edge_options', [])

            # #47: 瞄准角 = 扫描角 + 画面内框偏移 (YOLO 命中才有框;
            #     FIRST_DIRECTION 无框 → 原角度) — 目标转到画面中央
            aim_w = angle_to_use
            if yolo_warmup_conf is not None and (
                    getattr(self, '_ff', None) or {}).get(
                        'lock_bearing_fix', True):
                aim_w = int(round((angle_to_use + (getattr(
                    self, '_scan_hit_offset_deg', None) or 0.0)) % 360))

            # Best rotation: if angle <= 180, rotate RIGHT; else rotate LEFT is shorter
            if aim_w <= 180:
                rot_steps = aim_w // 30
                target_et = 'rotate_cw'
            else:
                rot_steps = (360 - aim_w) // 30
                target_et = 'rotate_ccw'

            # Find existing rotate option of this type (the basic 30° one)
            base_idx = None
            for i, opt in enumerate(edge_opts):
                if opt.get('chain_type') == target_et and opt.get('chain_count') == 1:
                    base_idx = i
                    break

            if base_idx is not None and rot_steps > 0:
                self._warmup_auto_action = base_idx
                self._warmup_auto_steps = rot_steps

                # === Insert a NEW option: the specific warmup rotation ===
                # Find where to insert (right after the basic rotate option)
                insert_idx = base_idx + 1
                warmup_opt = {
                    'chain_type': target_et, 'chain_count': rot_steps,
                    'direction': f'rotate_{"left" if "ccw" in target_et else "right"}',
                    'composite': None,
                }
                edge_opts.insert(insert_idx, warmup_opt)

                # Update obs edge_options
                obs['edge_options'] = edge_opts

                # Update avail_actions text: insert new line + tag existing
                avail_lines = obs.get('avail_actions', '').split('\n')
                # Insert new line for warmup option
                action_num = insert_idx  # 0-based index = action number
                rot_deg = rot_steps * 30
                turn_word = 'LEFT' if 'ccw' in target_et else 'RIGHT'
                new_line = f'  [{action_num}] turn {turn_word} ~{rot_deg}deg ★★★ WARMUP: GO THIS WAY! +20 BONUS ★★★'
                avail_lines.insert(insert_idx + 1, new_line)  # +1 for header line

                # Renumber all subsequent lines
                tagged_lines = [avail_lines[0]]  # header
                for li, line in enumerate(avail_lines[1:], start=1):
                    # Extract old number and replace with new
                    import re
                    line = re.sub(r'^\s*\[\d+\]', f'  [{li-1}]', line)
                    if li - 1 == action_num:
                        line = line.replace(' ★ WARMUP RECOMMENDED ★', '')  # clean up
                    tagged_lines.append(line)

                obs['avail_actions'] = '\n'.join(tagged_lines)
                self._warmup_auto_action = action_num  # update to new index
                logging.info(f'[WARMUP] Inserted warmup action [{action_num}] {turn_word} {rot_deg}deg, '
                           f'auto-action={action_num}, rotate_steps={rot_steps}')

        return obs

    def _post_episode(self):
        """Override: 先收集本回合计量 (父类 agent.reset() 会清 stopping_calls),
        再走父类存 df / reset / GIF 流程"""
        try:
            self._collect_episode_stats()
        except Exception as e:
            logging.warning(f'[EP-STATS] collect failed: {e}')
        super()._post_episode()

    def _collect_episode_stats(self):
        """终局计量一行: df 尾行 (finish_status/goal_reached/停距/spl) +
        停票门两级计数 + VLM 调用差分 → episode_stats_list (batch_run 转 CSV)"""
        tail = self.df.iloc[-1] if len(self.df) else None
        st = getattr(self, '_ep_stats', None) or {}

        def _t(col, default=0.0):
            try:
                return float(tail[col])
            except Exception:
                return default

        stats = {
            'episode_ndx': st.get('episode_ndx', -1),
            'category': (self.current_episode or {}).get('object_category', ''),
            'steps': int(len(self.df)),
            'finish_status': str(tail.get('finish_status')) if tail is not None else 'no_log',
            'goal_reached': bool(tail.get('goal_reached')) if tail is not None else False,
            'final_distance': _t('distance_to_goal', -1.0),
            'spl': _t('spl', 0.0),
            'initial_geodesic': float(st.get('initial_geodesic', -1.0)),
            # stopping_calls: init [-2], reset() 每回合重置 → len-1 = 实际停数
            'stop_requests': max(0, len(getattr(self.agent, 'stopping_calls', []) or []) - 1),
            'stop_vetoed': int(st.get('stop_vetoed', 0)),
            'votes_cleared': int(st.get('votes_cleared', 0)),
            'vlm_calls': self._vlm_call_total() - int(st.get('vlm_calls_start', 0)),
            # 修复2 L2 假阳性监控: triggered − final_close = 假阳性数 (CSV 直读)
            'fastpath_triggered': int(st.get('fastpath_triggered', 0)),
            'fastpath_final_close': int(st.get('fastpath_final_close', 0)),
            # A③/A⑥: VLM 判局 + 强制探索 rec-follow 诊断计量 (专家1 指标名)
            'vlm_critical_fails': int(st.get('vlm_critical_fails', 0)),
            'vlm_invalid': int(st.get('vlm_invalid', 0)) or (
                1 if getattr(self, '_vlm_episode_invalid', False) else 0),
            'vlm_invalid_reason': st.get('vlm_invalid_reason', '') or
                                  getattr(self, '_vlm_invalid_reason', ''),
            'forced_rec_used': int(getattr(self.agent, '_forced_rec_used', 0) or 0),
            'forced_rec_errors': int(getattr(self.agent, '_forced_rec_errors', 0) or 0),
            'forced_random_used': int(getattr(self.agent, '_forced_random_used', 0) or 0),
        }
        if not hasattr(self, 'episode_stats_list'):
            self.episode_stats_list = []
        self.episode_stats_list.append(stats)
        logging.info(f'[EP-STATS] {json.dumps(stats, ensure_ascii=False)}')

    def _start_forced_return(self):
        """若存在新鲜目击且当前不在目击节点 → 规划回访路径 (返回 True)"""
        sightings = getattr(self.simWrapper, 'target_sighting_nodes', [])
        if not sightings or sightings[-1][0] <= self._last_returned_sighting_step:
            return False
        s_step, s_node, s_desc = sightings[-1]
        cur = getattr(self.simWrapper, 'current_node', None)
        if cur is None or cur == s_node or (self.step - s_step) > 3:
            return False
        # 同位姿判定用位置而非节点 id: 目击节点常是同一站位的不同朝向变体,
        # 走回去 = 原地转 180° (实测烧掉 6 步/12% 预算), 而周期性重扫描
        # 本就免费覆盖全部朝向 → 直接跳过, 标记为已回访。
        # #45 (bmv8 step20 实弹): 该跳过只对"活体目击"成立 (当前帧即目击
        # 帧) — 扫描目击的目击帧是扫描照片 (如 240° 视角), 当前帧 (0°) 里
        # 目标根本不在画面: 直接配对必扑空, 0.34 真目击 (GT 夹角 0°) 就此
        # 蒸发。扫描目击 → 规划纯旋转路径转到目击朝向再看 (Dijkstra 自选
        # 短方向, 240° ≈ 4 边 ccw)。
        sw = self.simWrapper
        if hasattr(sw, '_node_pos') and cur and s_node:
            try:
                if float(np.linalg.norm(
                        np.array(sw._node_pos(cur)) - np.array(sw._node_pos(s_node)))) < 0.3:
                    if 'scan hit' in s_desc:
                        ets = self._graph_path_to(cur, s_node)
                        if ets and len(ets) <= 6:
                            self._forced_return_path = ets
                            self._last_returned_sighting_step = s_step
                            self._return_dest_node = s_node
                            logging.info(f'[FORCED-RETURN] same position but '
                                         f'sighting is a scan hit at another '
                                         f'heading → rotate {len(ets)} edges '
                                         f'to re-view (#45)')
                            return True
                    self._last_returned_sighting_step = s_step
                    # ③c: 人就在目击位置 (当前帧即目击位帧) → 当步就做配对确认
                    self._arrival_check_step = self.step
                    logging.info(f'[FORCED-RETURN] skip — already at sighting '
                                 f'position (rotation-only), no walk needed')
                    return False
            except Exception:
                pass
        path_ets = self._graph_path_to(cur, s_node)
        if path_ets and len(path_ets) <= 6:
            self._forced_return_path = path_ets
            self._last_returned_sighting_step = s_step
            self._return_dest_node = s_node
            logging.info(f'[FORCED-RETURN] Back to sighting node '
                         f'{s_node} ({len(path_ets)} edges: {path_ets})')
            return True
        return False

    def _arm_reacquire_after_miss(self):
        """#37 (bmv6 实弹): 接近锁因连续丢检释放 → 扫描目击失而复得。

        bmv6 三次释锁全链: 六图扫描 YOLO 命中 → target_priority 桥接近锁
        → 伺服 3-6 步零活体检出 → `_invalidate_lock(blacklist=False)`
        → 但 `_start_forced_return` 只认 wrapper 活体登记且新鲜度 >3 步
        即拒 → 机器人从"确曾看见目标的视角"继续盲走。
        这里在释锁当步放宽到 12 步: 走回扫描目击站位 (照片里 YOLO 真看到
        过 → 回去重扫大概率重锁; P4 释锁重扫在旅程后从该视角开火)。
        防同坑循环: 每个目击节点只补救一次 (`_reacquire_done`)。
        """
        if not (getattr(self, '_ff', None) or {}).get('scan_sighting', True):
            return
        sights = getattr(self.simWrapper, 'target_sighting_nodes', None) or []
        scan_s = next((s for s in reversed(sights) if 'scan hit' in s[2]), None)
        if scan_s is None:
            return
        s_step, s_node, _ = scan_s
        cur = getattr(self.simWrapper, 'current_node', None)
        if cur is None or cur == s_node or (self.step - s_step) > 12:
            return
        if s_node in getattr(self, '_reacquire_done', set()):
            return
        ets = self._graph_path_to(cur, s_node)
        if not ets or len(ets) > 6:
            return
        self._reacquire_done = getattr(self, '_reacquire_done', set()) | {s_node}
        self._forced_return_path = ets
        self._forced_return_dest = s_node
        self._last_returned_sighting_step = s_step
        logging.info(f'[REACQUIRE] approach lost but scan sighting {s_node} '
                     f'({self.step - s_step} steps ago) fresh → walk back to '
                     f're-detect ({len(ets)} edges, #37)')

    def _rescue_journey_edge(self, obs, next_et):
        """#34: 旅程计划边在当前节点无对应原语选项时的两级挽救, 返回选项 idx

        ① 重规划: 当前节点 → 目的地节点 (`_forced_return_dest` /
          `_return_dest_node`) 的图路径替换剩余路径, 取新首边再匹配一次
          (定位吸附/扫描替换 obs 后, 计划边序列与实际位置错位是常态,
          不是放弃旅程的理由);
        ② 方位等价边: 仍无 → 目的地世界方位 vs 各选项世界位移方位
          (`_yaw_world + _option_bearing_deg`), 差 ≤75° 中取最贴合的
          非复合选项代走 — 旅程在下一节点重新对齐。
        两级都失败 → None (真无路可走才弃程)。
        """
        cur = getattr(self.simWrapper, 'current_node', None)
        dest = getattr(self, '_forced_return_dest', None) \
            or getattr(self, '_return_dest_node', None)
        # ① 重规划 (剩余路径按当前位置重算)
        if cur and dest and cur != dest:
            ets = self._graph_path_to(cur, dest)
            if ets:
                m = next((i for i, o in enumerate(obs.get('edge_options', []))
                          if o.get('chain_type') == ets[0]
                          and o.get('chain_count') == 1
                          and not o.get('composite')), None)
                if m is not None:
                    self._forced_return_path = ets
                    n_rot = sum(1 for x in ets if 'rotate' in x)
                    logging.info(f'[FORCED-RETURN] replanned from {cur} '
                                 f'to {dest} ({len(ets)} edges: {n_rot} rot '
                                 f'+ {len(ets) - n_rot} trans, first '
                                 f'{ets[0]})')
                    return m
                next_et = ets[0]     # ② 用重规划首边的方位语义做等价回退
        # ② 方位等价边
        try:
            g = self.simWrapper.nav_graph
            if not (g and cur and dest and dest in g['nodes']
                    and cur in g['nodes']):
                return None
            cp = np.array([g['nodes'][cur]['world_pos'][0],
                           g['nodes'][cur]['world_pos'][2]])
            dp = np.array([g['nodes'][dest]['world_pos'][0],
                           g['nodes'][dest]['world_pos'][2]])
            if float(np.hypot(*(dp - cp))) < 0.8:
                return None          # 已在目的地附近, 旅程自然结束
            want = np.degrees(np.arctan2(dp[0] - cp[0], dp[1] - cp[1]))
            yaw = np.degrees(self._yaw_world())
            use_geo = (getattr(self, '_ff', None) or {}).get('servo_geometry', True)
            best, best_diff = None, 1e9
            for i, o in enumerate(obs.get('edge_options', [])):
                if o.get('composite') or not o.get('chain_type'):
                    continue
                wbd = self._option_world_bearing_deg(o) if use_geo else None
                ob = wbd if wbd is not None \
                    else (self._option_bearing_deg(o) + yaw) % 360.0
                diff = abs((ob - want + 180) % 360 - 180)
                if diff < best_diff:
                    best, best_diff = i, diff
            if best is not None and best_diff <= 75.0:
                logging.info(f'[FORCED-RETURN] bearing-equivalent edge '
                             f'[{best}] (Δ{best_diff:.0f}° toward dest) '
                             f'rescues journey (planned {next_et} missing)')
                return best
        except Exception:
            return None
        return None

    def _execute_forced_return(self, obs):
        """执行回访路径的下一步 (覆盖 VLM 决策), 路径空时返回 None"""
        if not self._forced_return_path:
            return None
        next_et = self._forced_return_path.pop(0)
        edge_opts = obs.get('edge_options', [])
        # === #41: 同向多边并步 — 旅程后续边同型 且 obs 有对应链式选项
        #     (wrapper 深度走廊通畅才生成, 零样本合规) → 一次弹 K 条走链,
        #     不再"直行一点一点挪"。走廊不通/链断 → 回落单边 (原行为)。 ===
        run = 1
        if (getattr(self, '_ff', None) or {}).get('journey_stride', True) \
                and next_et in ('forward', 'backward', 'left', 'right'):
            while (run < 3
                   and len(self._forced_return_path) > run - 1
                   and self._forced_return_path[run - 1] == next_et):
                run += 1
        idx = None
        if run > 1:
            idx = next((i for i, o in enumerate(edge_opts)
                        if o.get('chain_type') == next_et
                        and o.get('chain_count', 1) == run
                        and not o.get('composite')), None)
            if idx is not None:
                del self._forced_return_path[:run - 1]
                logging.info(f'[JOURNEY-STRIDE] {next_et} x{run} 并步 '
                             f'(走廊通畅, 剩 {len(self._forced_return_path)} 边)')
            else:
                run = 1   # 无链式选项 → 回落单边
        if run == 1:
            idx = next((i for i, o in enumerate(edge_opts)
                        if o.get('chain_type') == next_et
                        and o.get('chain_count') == 1
                        and not o.get('composite')), None)
        if idx is None:
            # === #34 (bmv4 实弹: "no option for left, aborting" ×2, 7-9 边
            #     旅程半途而废): 计划边是按规划时节点链生成的 edge_type 序列,
            #     实际执行的落点被定位吸附到别的节点 (或本步 obs 被扫描替换)
            #     → 当前节点图边里没有该原语。旅程不死, 两级挽救:
            #     ① 从当前节点到目的地节点重规划 (图边对齐当前位置);
            #     ② 仍缺 → 方位等价边: 任何把机器人往目的地方向送的可用
            #        选项 (世界方位差 ≤75°) 代走, 下一步在新节点重新对齐。 ===
            idx = self._rescue_journey_edge(obs, next_et)
            if idx is None:
                logging.warning(f'[FORCED-RETURN] no option for {next_et} '
                                f'(rescue failed), aborting')
                self._forced_return_path = []
                self._forced_return_dest = None
                return None
        # ③c: 这是回访路径最后一入边 → 执行后机器人就站在目击位置,
        #     下一步的 obs 即"到场帧" → 武装到场配对确认
        if not self._forced_return_path:
            self._arrival_check_step = self.step + 1
        # 走完整 VLM 流程 (保留日志/上下文), 然后覆盖动作为回访边
        agent_action = self._override_and_run(
            obs, idx, 'FORCED-RETURN',
            f'{next_et} (auto return to sighting node), '
            f'remaining {len(self._forced_return_path)}')
        # details.txt 补充记录
        if agent_action is not None:
            ep_dir = (f'logs/{self.outer_run_name}/{self.inner_run_name}/'
                      f'{self.curr_run_name}/step{self.step}')
            dt_path = f'{ep_dir}/details.txt'
            if os.path.exists(dt_path):
                with open(dt_path, 'a') as f:
                    f.write(f'\nFORCED_RETURN: executed action [{idx}] '
                            f'{next_et} (auto return to sighting node)\n')
        return agent_action

    def _vlm_proximity_check(self, obs, mode='yolo_box'):
        """VLM 视觉接近度判断: 目标是否足够接近 (不用任何距离信息)

        mode='yolo_box'  : YOLO 高置信检测到目标 → 画红框问 VLM 是否足够近
        mode='stop_vote' : 停止投票仲裁 — 无需 YOLO 框, 直接问 VLM
                           当前视野里目标是否大而清晰 (是否该停)
        返回 'close' / 'wrong' / 'far'; 不满足触发条件返回 False。
        """
        yolo_info = obs.get('yolo_detection') or {}
        target_bbox = None
        if yolo_info.get('target_found'):
            for d in yolo_info.get('all_detections', []):
                if d.get('class_name') == self.simWrapper.target_name:
                    target_bbox = d.get('bbox_norm')
                    break
        elif mode == 'yolo_box':
            return False  # 无检测不触发常规接近度检查
        from PIL import Image as PILImage, ImageDraw
        rgb = obs['color_sensor']
        zoom_note = ''
        zoom = None
        if target_bbox and (getattr(self, '_ff', None) or {}).get(
                'zoom_confirm', True):
            zoom = self._zoom_bbox_frame(obs, yolo_info)
        if zoom is not None:
            # P5 (修7c'): bbox 裁剪放大 — 全幅图上远距小目标 (area<1%)
            # VLM 看不清, ep1 step25 conf0.98 真目击被判 wrong 的真因
            z_rgb, (zx1, zy1, zx2, zy2) = zoom
            img = PILImage.fromarray(z_rgb[:, :, :3] if z_rgb.ndim == 3 else z_rgb)
            draw = ImageDraw.Draw(img)
            draw.rectangle([zx1, zy1, zx2, zy2], outline=(255, 0, 0), width=6)
            zoom_note = ("The image is a ZOOMED-IN crop centered on the "
                         "marked object.\n")
        else:
            img = PILImage.fromarray(rgb[:, :, :3] if rgb.ndim == 3 else rgb)
            if target_bbox:
                x1, y1, x2, y2 = target_bbox
                W, H = img.size
                draw = ImageDraw.Draw(img)
                draw.rectangle([x1 * W, y1 * H, x2 * W, y2 * H],
                               outline=(255, 0, 0), width=6)
        target_words = self.simWrapper.target_name.replace('_', ' ')
        if target_bbox:
            lead = (f"{zoom_note}"
                    f"A detector marked the target '{target_words}' with a RED BOX "
                    f"in your current view.\nLook at the marked object:\n")
        else:
            lead = (f"You are navigating to find '{target_words}' and just voted "
                    f"to STOP because you believe you see it.\n"
                    f"Look at your current view carefully:\n")
        prompt = (
            f"{lead}"
            f"- If the '{target_words}' is LARGE and clearly visible (you are "
            f"close enough that the navigation task is COMPLETE) → done=1.\n"
            f"- If it is small/far/blurry, or you cannot actually see it → done=0.\n"
            f"- If the object you see is NOT actually a '{target_words}' (wrong "
            f"object / false detection) → done=0 and write 'wrong object' in reasoning.\n"
            f"Respond ONLY with JSON: "
            f'{{"done": <0 or 1>, "reasoning": "<one sentence>"}}'
        )
        try:
            import re as _re
            response = self.agent.actionVLM.call_chat(0, [np.array(img)], prompt)
            m = _re.search(r'"done"\s*:\s*(\d)', response or '')
            done = bool(m) and m.group(1) == '1'
            rl = (response or '').lower()
            if done:
                verdict = 'close'
            elif any(w in rl for w in ['wrong object', 'not the target', 'not a ',
                                       'different object', 'not actually',
                                       'false detection', 'false positive']):
                verdict = 'wrong'
            else:
                verdict = 'far'
            logging.info(f'[PROXIMITY] VLM check ({mode}) verdict={verdict} '
                         f'({str(response)[:110]})')
            return verdict
        except Exception as e:
            logging.warning(f'[PROXIMITY] check failed: {e}')
            return 'far'

    def _stop_arbitration(self):
        """停止投票仲裁: VLM 要停 + 有新鲜目击时, 用 VLM 视觉判断该不该停

        替代旧 STOP-BLOCK 的一刀切拦截 (实测曾把 1.98m 处的正确停止拦掉)。
        'close' → 尊重停止; 'wrong' → 拉黑误报继续找; 'far' → 继续找
        (除非不可达 — 见 _target_approachable)。
        """
        obs = getattr(self, '_last_obs', None)
        if obs is None or 'color_sensor' not in obs:
            return 'far'  # 无图可判 → 保守继续找
        return self._vlm_proximity_check(obs, mode='stop_vote')

    # === 问题1 距离判断加强: 两个独立于"框内深度网格"的参考源 ===
    def _support_depth(self, d_img, x1, x2, y2, H, W):
        """问题1-A: 支撑面深度采样 — bbox 底边中点下方 +1.5%H 处

        瓶/罐站立在桌/台面上, 支撑面是实心连续表面, 深度可靠; 目标本体
        若是玻璃/反光物, 深度图上常是"洞" (打不到表面), 框内 3×3 网格
        会穿透打到背景墙。采样窗与网格采样同规格 (±4px, 中位数)。
        返回米; 采样无效返回 None。
        """
        try:
            cx = int(np.clip((x1 + x2) / 2.0 * W, 8, W - 8))
            cy = int(np.clip(y2 * H + 0.015 * H, 8, H - 8))
            win = d_img[max(0, cy - 4):cy + 4, max(0, cx - 4):cx + 4]
            valid = win[(win > 0.05) & (win < 12.0)]
            if valid.size >= 3:
                return float(np.median(valid))
        except Exception:
            pass
        return None

    def _xcheck_depth(self, grid_med, sup):
        """问题1-A: 框内网格深度 vs 支撑面深度 交叉仲裁

        两源一致 (Δ≤0.6m) → 用网格中位数 (原行为)。强分歧 → 保守取远
        (max): 遮挡物污染 (网格偏近, bm10 型) 与背景穿透 (网格偏远,
        玻璃物型) 两个方向都不会产生新的误停 — 只可能多走近一步,
        由面积≥2% 捷径或下一帧交叉再收。sup=None → 原行为。
        """
        if sup is None or abs(sup - grid_med) <= 0.6:
            return grid_med
        eff = max(grid_med, sup)
        logging.info(f'[STOP-EVIDENCE] support-surface {sup:.2f}m vs '
                     f'bbox-grid {grid_med:.2f}m (Δ>{0.6:.1f}) → '
                     f'conservative max {eff:.2f}m (surface reference)')
        return eff

    def _parallax_distance(self):
        """问题1-B: 运动视差测距 — 完全独立于深度传感器的距离估计

        针孔模型: 像面积 ∝ 1/d² → 连续两帧面积比 r=a2/a1=(d1/d2)²,
        已知位移 Δ=d1−d2 (图节点坐标 = 机器人自身里程计, 非真值) →
        d2 = Δ/(√r − 1)。深度图的玻璃洞/遮挡污染对它全部免疫。
        有效性: 面积在增 (≥1.10, 逼近中)、位移 ≥0.25m (低于则噪声
        主导)、比率 ≤25 (检测框跳变/目标切换不可信)。Δ 取两节点欧氏
        距离 (路径长 ≥ 视轴分量 → d2 只会高估 = 保守拒停方向)。
        返回米; 不满足条件返回 None。
        """
        h = getattr(self, '_bbox_area_hist', None) or []
        sw = self.simWrapper
        if len(h) < 2 or not hasattr(sw, '_node_pos'):
            return None
        (_s1, a1, n1), (_s2, a2, n2) = h[-2], h[-1]
        if a2 < a1 * 1.10 or not n1 or not n2 or n1 == n2:
            return None
        try:
            delta = float(np.linalg.norm(
                np.array(sw._node_pos(n1)) - np.array(sw._node_pos(n2))))
        except Exception:
            return None
        if delta < 0.25:
            return None
        ratio = a2 / a1
        if ratio > 25.0:
            return None
        d2 = delta / (np.sqrt(ratio) - 1.0)
        if not (0.1 < d2 < 8.0):
            return None
        return d2

    # P1 阈值: 独立几何源否决线 (米)。停票 close 的容许视觉阈 2.5m
    # (VISUAL_SUCCESS), 但 "crop 说 close" 的语义是近到看得清 — >1.5m
    # 的 crop close 一律是误判 (ep6 step35: crop close @3.95m)。
    DIST_XCHECK_FAR = 1.5

    def _forward_band_depth(self, d_img):
        """P1: 前向中央带深度 — live 深度图中央水平带网格采样中位数

        照片注入只覆盖 color_sensor (wrapper _inject_photo), depth_sensor
        保持 live — 与照片视轴同朝向 (同一相机位姿)。代表"正前方最近
        连续表面"的距离, 是无框场景 (crop close) 下唯一可用的几何源。
        带取 y∈[0.45,0.55] (避地面避天花板) × x∈[0.3,0.7] (中央)。
        返回米; 有效样本不足返回 None。
        """
        try:
            d = np.asarray(d_img, dtype=np.float32)
            if d.size == 0:
                return None
            H, W = d.shape[:2]
            samples = []
            for fx in (0.3, 0.5, 0.7):
                for fy in (0.45, 0.55):
                    cx = int(np.clip(fx * W, 8, W - 8))
                    cy = int(np.clip(fy * H, 8, H - 8))
                    win = d[max(0, cy - 4):cy + 4, max(0, cx - 4):cx + 4]
                    valid = win[(win > 0.05) & (win < 12.0)]
                    if valid.size >= 3:
                        samples.append(float(np.median(valid)))
            if samples:
                return float(np.median(samples))
        except Exception:
            pass
        return None

    def _distance_xcheck(self, obs, source):
        """P1: 停票 close 判定的独立几何交叉核 — 只否决, 不放行

        crop close (VLM 看放大裁剪说近) 与 fastpath 单票 close 的共同
        失效模式: 证据源全是"视觉" — 无框 (crop) 或框内深度被玻璃洞/
        遮挡物污染 (ep6 step37: 框内网格 0.61m, 实际 2.42m, 支撑面采样
        无效 + 视差只有 1 帧历史 → 两源全哑)。本核汇集三个独立于
        "框内网格"的几何源, 任一有效源 >1.5m → 否决 ('veto');
        全部无证据 → 维持 VLM 判定 ('none' — 防死锁: G1/G4 近距真目标
        常无任何几何证据, 否则会拒掉唯一正确停票)。
        源优先级: 视差 (照片同源, 零配准误差) > 支撑面 (bbox 底边下方
        台面) > 前向中央带 (live 深度, 无框也可用)。
        """
        _ff = getattr(self, '_ff', None) or {}
        if not _ff.get('dist_xcheck', True):
            return 'none', None
        far = self.DIST_XCHECK_FAR
        # 源1: 视差 (照片面积比 + 里程计, 与深度传感器完全独立)
        try:
            pd = self._parallax_distance()
        except Exception:
            pd = None
        if pd is not None and pd > far:
            logging.info(f'[DIST-XCHECK] {source}: parallax {pd:.2f}m > '
                         f'{far}m → veto close, keep searching')
            return 'veto', ('parallax', pd)
        # 源2: 支撑面 (bbox 底边下方实心表面; 无框则跳过)
        yolo = (obs or {}).get('yolo_detection') or {}
        bbox = None
        for dd in yolo.get('all_detections', []):
            if dd.get('class_name') == getattr(
                    self.simWrapper, 'target_name', '') \
                    and dd.get('bbox_norm'):
                bbox = [float(v) for v in dd['bbox_norm']]
                break
        d_img = (obs or {}).get('depth_sensor')
        sup = None
        if bbox is not None and d_img is not None:
            try:
                d_arr = np.asarray(d_img, dtype=np.float32)
                H, W = d_arr.shape[:2]
                sup = self._support_depth(d_arr, bbox[0], bbox[2], bbox[3],
                                          H, W)
            except Exception:
                sup = None
        if sup is not None and sup > far:
            logging.info(f'[DIST-XCHECK] {source}: support-surface {sup:.2f}m '
                         f'> {far}m → veto close, keep searching')
            return 'veto', ('support', sup)
        # 源3: 前向中央带 (live 深度, 与照片视轴同朝向)。
        #   方位守卫: 带测视轴方向 — 有框且框在侧向 (|cx-0.5|>0.25) 时
        #   带深不是目标距离的证据, 跳过; 无框 (crop close, 目标在
        #   画面中下部裁剪内 = 视轴附近) 时适用。
        band_ok = True
        if bbox is not None:
            band_ok = abs((bbox[0] + bbox[2]) / 2.0 - 0.5) <= 0.25
        fwd = self._forward_band_depth(d_img) if band_ok else None
        if fwd is not None and fwd > far:
            logging.info(f'[DIST-XCHECK] {source}: forward-band {fwd:.2f}m > '
                         f'{far}m → veto close, keep searching')
            return 'veto', ('forward-band', fwd)
        return 'none', None

    # === D2/③: 停票证据分级 — 修缺陷 A/B (G2 式 4 步无目击连票自杀) ===
    #     停止票不再只看 VLM 连续投票: 先用自家传感器做米制分级。
    #     有 YOLO 框 → bbox 中心深度探测 (合法: 机器人自己的深度传感器,
    #     不是真值); 无框 → ③ crop 放大确认门。
    def _stop_evidence(self, obs):
        """② 停票证据: YOLO 框 + 深度传感器 → ('close'|'far', yolo_info)

        'close': 框面积 ≥2% 画面 (目视级大) 或深探 ≤1.0m (1.3 硬判据)
        'far'  : 有框但深探 >1.0m — (1.0, 2.2]m 软接近区 (拒停转接近锁
                 继续逼近), >2.2m 为远; 或小框+极近物理矛盾 (探测被
                 遮挡物污染, 小框即远证据) → 均拒停, 转为接近 (AUTO-STEER)
        None   : 无框/深度不可用 → 交给 ③ crop 门
        """
        if not obs:
            return None
        yolo_info = obs.get('yolo_detection') or {}
        if not yolo_info.get('target_found'):
            # #49: 静默 None 改 INFO — 决策帧无框 vs 证据不同源 (obs 错
            #     张) 在日志里可分辨, 不再黑盒 (bmv11 step28 米制门全哑)
            logging.info('[STOP-EVIDENCE] decision obs has no target box '
                         '(yolo_detection.target_found falsy)')
            return None
        bbox = None
        for d in yolo_info.get('all_detections', []):
            if d.get('class_name') == getattr(self.simWrapper, 'target_name', '') \
                    and d.get('bbox_norm'):
                bbox = [float(v) for v in d['bbox_norm']]
                break
        if not bbox:
            return None
        x1, y1, x2, y2 = bbox
        area = max(0.0, (x2 - x1) * (y2 - y1))
        if area >= 0.02:
            logging.info(f'[STOP-EVIDENCE] bbox area {area:.3f} ≥2% → CLOSE')
            return ('close', yolo_info)
        d_img = obs.get('depth_sensor')
        if d_img is not None:
            try:
                d_img = np.asarray(d_img, dtype=np.float32)
                H, W = d_img.shape[:2]
                # 3×3 网格采整个框取中位数: 只探中心一点会被穿过框的
                # 近处遮挡物击中 (bm10 实测: 4.95m 外小框中心探到 0.47m
                # 遮挡物 → 误停 fp)
                samples = []
                for fx in (0.2, 0.5, 0.8):
                    for fy in (0.2, 0.5, 0.8):
                        cx = int(np.clip((x1 + (x2 - x1) * fx) * W, 8, W - 8))
                        cy = int(np.clip((y1 + (y2 - y1) * fy) * H, 8, H - 8))
                        win = d_img[max(0, cy - 4):cy + 4, max(0, cx - 4):cx + 4]
                        valid = win[(win > 0.05) & (win < 12.0)]
                        if valid.size >= 3:
                            samples.append(float(np.median(valid)))
                if samples:
                    med = float(np.median(samples))
                    sup = None   # #50: 支撑面源保留给小框隔离佐证用
                    # === 问题1-A (support_depth): 支撑面交叉 — bbox 底边
                    #     下方台面采样; 强分歧保守取远 (见 _xcheck_depth) ===
                    if (getattr(self, '_ff', None) or {}).get(
                            'support_depth', True):
                        sup = self._support_depth(d_img, x1, x2, y2, H, W)
                        med = self._xcheck_depth(med, sup)
                    # 尺寸-深度一致性: 极近 (<0.5m) 处的真目标必然占 >2%
                    # 画面; 小框 + 极近 = 物理矛盾 (探到的是遮挡物, 非目标)。
                    # bm14 实弹: defer crop 后放大裁剪抹掉尺寸线索, 5.43m
                    # 远处真目标被判 close → fp 停 (bm11 同病)。crop 能证
                    # present 证不了 close — 小框本身即"远"的强证据 →
                    # 直接 far: 拒停 + 接近锁走近 (近了框 ≥2% 走面积捷径)。
                    # 边界 0.5m 独立于硬判据 1.0m — 若共用 1.0, 小框在
                    # (0.5, 1.0] 的硬停通道会被矛盾分支整个吞掉 (死代码)
                    if med < self.MISMATCH_DEPTH and area < 0.02:
                        logging.info(f'[STOP-EVIDENCE] size-depth mismatch '
                                     f'(depth {med:.2f}m, area {area:.3f}) '
                                     f'→ FAR (small box = far evidence, '
                                     f'probe occluded)')
                        return ('far', yolo_info)
                    # 1.3 分级: ≤1.0m 硬判据可停; (1.0, 2.2]m 软接近 —
                    # 不放行 STOP, 拒停转接近锁继续逼近 (>2.2m 为远)
                    if med <= self.HARD_STOP_DEPTH:
                        verdict = 'close'
                    else:
                        verdict = 'far'
                        if med <= self.SOFT_APPROACH_DEPTH:
                            logging.info(f'[STOP-EVIDENCE] soft-approach zone '
                                         f'({med:.2f}m ∈ (1.0, {self.SOFT_APPROACH_DEPTH}]) '
                                         f'→ veto stop, keep approaching')
                    # === 问题1-B (parallax_gate): 视差交叉 — 独立测距
                    #     (面积比+里程计) 说还远 (>1.2m, 留 0.2m 估计容差)
                    #     时否决 close。只降不升: 视差永远不把 far 抬成
                    #     close — 新增源只收紧误停, 不放宽。 ===
                    if verdict == 'close' and (getattr(self, '_ff', None)
                                               or {}).get('parallax_gate', True):
                        pd = self._parallax_distance()
                        if pd is not None and pd > 1.2:
                            logging.info(
                                f'[STOP-EVIDENCE] parallax {pd:.2f}m > 1.2 '
                                f'(area-ratio + odometry, depth-independent) '
                                f'→ demote CLOSE to FAR')
                            verdict = 'far'
                    # === P1 (dist_xcheck) 第三源: 前向中央带 — ep6 step37
                    #     失守形态: 框内网格 0.61m (污染) close + sup=None
                    #     + 视差 1 帧历史 None → 两源全哑误停 @2.42m。
                    #     live 前向带 (与照片视轴同朝向) 是最后防线; 只
                    #     否决不放行 (带深 ≤1.5m 不抬票)。
                    #     方位守卫: 带测的是视轴方向 — 只在目标位于视轴
                    #     附近 (|cx-0.5|≤0.25 ≈ ±33°) 时才是目标距离的
                    #     证据; 侧向近目标 (框内近 + 视轴远墙) 不在
                    #     适用域, 不降级 (test_stop_gate 左近框回归)。 ===
                    if verdict == 'close' and (getattr(self, '_ff', None)
                                               or {}).get('dist_xcheck', True):
                        bx_c = (x1 + x2) / 2.0
                        if abs(bx_c - 0.5) <= 0.25:
                            fb = self._forward_band_depth(d_img)
                            if fb is not None and fb > self.DIST_XCHECK_FAR:
                                logging.info(
                                    f'[STOP-EVIDENCE] forward-band {fb:.2f}m > '
                                    f'{self.DIST_XCHECK_FAR} (live center strip) '
                                    f'→ demote CLOSE to FAR (P1 xcheck)')
                                verdict = 'far'
                    # === #50 正修 (d03r step25 实弹 fp@5.84m): 小框深度
                    #     隔离 — 小框 (area<0.5%) 的 bbox-grid 深度双向
                    #     不可信, 三件套实弹证据: bmv12 九点饿死→None,
                    #     d01r 0.42m 近探针→误 FAR, d03r 0.53m 框外近处
                    #     地板→误 CLOSE 放行 fp (真实 5.84m, 0.53 恰好
                    #     溜过 0.5m 矛盾崖)。隔离: 小框 CLOSE 须独立源
                    #     佐证 (视差/支撑面/前向带任一 ≤ 硬判据), 无佐
                    #     证 → FAR 拒停转接近 (走近框变大: 面积捷径或
                    #     佐证 CLOSE 自然到) — 与 #49b "present≠close"
                    #     同语义, 补上 proximity 路径缺口。bmv12 成功停
                    #     area 0.020 走 ≥2% 面积捷径, 不受影响。 ===
                    if verdict == 'close' \
                            and area < self.SMALL_BOX_QUARANTINE_AREA \
                            and (getattr(self, '_ff', None) or {}).get(
                                'small_box_quarantine', True):
                        corr = None
                        pd_q = self._parallax_distance()
                        if pd_q is not None and pd_q <= 1.2:
                            corr = f'parallax {pd_q:.2f}m'
                        elif sup is not None \
                                and sup <= self.HARD_STOP_DEPTH:
                            corr = f'support {sup:.2f}m'
                        else:
                            bx_c = (x1 + x2) / 2.0
                            if abs(bx_c - 0.5) <= 0.25:
                                fb = self._forward_band_depth(d_img)
                                if fb is not None \
                                        and fb <= self.HARD_STOP_DEPTH:
                                    corr = f'forward-band {fb:.2f}m'
                        if corr:
                            logging.info(
                                f'[STOP-EVIDENCE] small-box quarantine '
                                f'passed (area {area:.3f}, corroborated '
                                f'by {corr})')
                        else:
                            logging.info(
                                f'[STOP-EVIDENCE] small-box quarantine '
                                f'(area {area:.3f} grid {med:.2f}m '
                                f'uncorroborated) → FAR — approach, not '
                                f'stop (#50)')
                            verdict = 'far'
                    logging.info(f'[STOP-EVIDENCE] bbox-grid depth '
                                 f'{med:.2f}m area {area:.3f} → {verdict.upper()}')
                    return (verdict, yolo_info)
                else:
                    # #50 diag: 小框九点全无效 — 取证是哪个环节饿死采样
                    #   (双目错位 #23: 框在预渲染照片坐标系, 深度是 live
                    #   渲染? 值域/分辨率异常?), 只读自身传感器
                    try:
                        sub = d_img[max(0, int(y1 * H)):int(y2 * H) + 1,
                                    max(0, int(x1 * W)):int(x2 * W) + 1]
                        fin = d_img[np.isfinite(d_img)]
                        logging.info(
                            f'[STOP-EVIDENCE] bbox present (area {area:.3f}) '
                            f'but depth samples empty → None (#50 diag: '
                            f'd {d_img.dtype} {W}x{H} bbox['
                            f'min={float(sub.min()):.2f} '
                            f'max={float(sub.max()):.2f} n={sub.size}] '
                            f'frame[p50={float(np.median(fin)):.2f} '
                            f'valid%={100 * float((fin > 0.05).mean()):.0f}])')
                    except Exception:
                        logging.info(
                            f'[STOP-EVIDENCE] bbox present (area {area:.3f}) '
                            f'but depth samples empty → None')
            except Exception as e:
                logging.info(f'[STOP-EVIDENCE] depth probe exception: {e}')
        else:
            logging.info(f'[STOP-EVIDENCE] bbox present (area {area:.3f}) '
                         f'but obs has no depth_sensor → None')
        return None

    def _crop_confirm(self, obs, reason=''):
        """③ crop 放大确认: 中下 2/3 裁剪 ×2 放大, 一次 VLM 判目标在场与否

        触发: (a) 无框停票门 (b) 初进新 2m 区域格且前方 <2m 有台面。
        近距小目标在整幅缩略图里常糊成一团 (G1/G4 踩到 0.06m 仍不停),
        放大再看一次是最后的兜底确认。限流 ≤10 次/回合。
        返回 'close' / 'far' / 'no'; 不满足条件返回 None。
        """
        if getattr(self, '_crop_calls', 0) >= 10:
            return None
        rgb = (obs or {}).get('color_sensor')
        if rgb is None:
            return None
        try:
            from PIL import Image as PILImage
            img = np.asarray(rgb)
            if img.ndim == 3:
                img = img[:, :, :3]
            H, W = img.shape[:2]
            crop = img[int(H * 0.30):, int(W * 0.12):int(W * 0.88)]
            big = PILImage.fromarray(crop).resize(
                (crop.shape[1] * 2, crop.shape[0] * 2), PILImage.LANCZOS)
            target_words = self.simWrapper.target_name.replace('_', ' ')
            prompt = (
                f"This is a ZOOMED-IN view of the area directly in front of a "
                f"home robot. The robot is searching for '{target_words}'.\n"
                f"Look carefully at surfaces (counters, tables, shelves, floor):\n"
                f"- Is a '{target_words}' VISIBLE in this view? (present=1/0)\n"
                f"- If visible, is it CLOSE (within about 2 meters, large and "
                f"clearly recognizable)? (close=1/0)\n"
                f"Respond ONLY with JSON: "
                f'{{"present": <0 or 1>, "close": <0 or 1>, '
                f'"reasoning": "<one short sentence>"}}'
            )
            self._crop_calls = getattr(self, '_crop_calls', 0) + 1
            self._last_crop_step = self.step
            response = self.agent.actionVLM.call_chat(
                0, [np.array(big)], prompt)
            import re as _re
            m_p = _re.search(r'"present"\s*:\s*(\d)', response or '')
            m_c = _re.search(r'"close"\s*:\s*(\d)', response or '')
            present = bool(m_p) and m_p.group(1) == '1'
            close = bool(m_c) and m_c.group(1) == '1'
            verdict = 'close' if (present and close) else \
                ('far' if present else 'no')
            logging.info(f'[CROP-CONFIRM] ({reason}) verdict={verdict} '
                         f'calls={self._crop_calls} ({str(response)[:110]})')
            return verdict
        except Exception as e:
            logging.warning(f'[CROP-CONFIRM] failed: {e}')
            return None

    def _arrival_confirm(self, obs, s_rgb):
        """③(c) 到场配对确认: 目击帧(左) + 当前帧(右) 并排, 一次 VLM 判定

        场景 (G2 实测 step26-28): 目击时目标又大又近 → 强制回访原位 →
        目标却出画 (相机俯角/朝向变了)。当前帧怎么看都无法确认, 但
        "目击帧近景 + 我已回到原位" 本身就是停止证据 (0.067m 不停的修法)。
        判据只看左帧表观尺寸 — 机器人自己的历史相机帧, 合法信息, 不用真值。
        返回 'close' / 'no'; 不满足条件返回 None。限流 ≤6 次/回合。
        """
        if getattr(self, '_arrival_calls', 0) >= 6:
            return None
        rgb = (obs or {}).get('color_sensor')
        if rgb is None:
            return None
        try:
            from PIL import Image as PILImage
            tgt_w = 448

            def _sq(a):
                im = PILImage.fromarray(np.asarray(a)[:, :, :3])
                w, h = im.size
                return im.resize((tgt_w, max(1, int(h * tgt_w / w))),
                                 PILImage.LANCZOS)
            L, R = _sq(s_rgb), _sq(rgb)
            combo = PILImage.new('RGB', (L.width + R.width,
                                         max(L.height, R.height)))
            combo.paste(L, (0, 0))
            combo.paste(R, (L.width, 0))
            target_words = self.simWrapper.target_name.replace('_', ' ')
            prompt = (
                f"A home robot returned to the exact spot where it earlier SAW "
                f"the target '{target_words}'.\n"
                f"LEFT image: the robot's camera view at the moment it spotted "
                f"the target there.\n"
                f"RIGHT image: the robot's current view at that same spot.\n"
                f"Question: in the LEFT image, was the '{target_words}' clearly "
                f"visible and CLOSE-UP (large and clearly recognizable, within "
                f"about 2 meters)?\n"
                f"Respond ONLY with JSON: "
                f'{{"close": <0 or 1>, "where": "<one short phrase>"}}'
            )
            self._arrival_calls = getattr(self, '_arrival_calls', 0) + 1
            response = self.agent.actionVLM.call_chat(0, [np.array(combo)], prompt)
            import re as _re
            m_c = _re.search(r'"close"\s*:\s*(\d)', response or '')
            verdict = 'close' if (m_c and m_c.group(1) == '1') else 'no'
            logging.info(f'[ARRIVAL-CONFIRM] verdict={verdict} '
                         f'calls={self._arrival_calls} ({str(response)[:110]})')
            return verdict
        except Exception as e:
            logging.warning(f'[ARRIVAL-CONFIRM] failed: {e}')
            return None

    def _bridge_arrival_verdict(self, distance):
        """B1 目击→停票桥 (P0 断层修复): 新鲜目击 + 仍在目击点 → 到场配对

        actionVLM/YOLO 的目击到不了停票通道 (双 VLM 并联, stop_requests=0,
        9/9 失败全超时)。停票 prompt 桥 (avdb_agent.bridge_hint) 让票流起来
        后, 无框停票在此仲裁: 目击帧 (登记那一刻的近景, 尺寸/距离证据在
        拍摄时已固化) + 当前帧并排送 VLM 到场配对 — 'close' 才放行。
        目击过期 (>3 步) / 已离开目击点 (>0.5m) / 无留存帧 → None,
        落回无框一律 far 的既有轨道 (bm16 教训不破)。
        """
        sw = self.simWrapper
        sightings = getattr(sw, 'target_sighting_nodes', []) or []
        if not sightings:
            return None
        steps_ago = sw.memory.get('step_count', 0) - sightings[-1][0]
        if not (0 <= steps_ago <= 3):
            return None
        s_node = sightings[-1][1]
        s_frame = getattr(self, '_sighting_frames', {}).get(s_node)
        if s_frame is None:
            return None
        near_spot = (getattr(sw, 'current_node', None) == s_node)
        if not near_spot and hasattr(sw, '_node_pos'):
            try:
                cur_p = np.array(sw._node_pos(sw.current_node))
                sight_p = np.array(sw._node_pos(s_node))
                near_spot = float(np.linalg.norm(cur_p - sight_p)) <= 0.5
            except Exception:
                near_spot = False
        if not near_spot:
            return None
        if self._arrival_confirm(getattr(self, '_last_obs', None),
                                 s_frame) == 'close':
            logging.info(f'[BRIDGE-ARRIVAL] fresh sighting ({steps_ago} step(s) '
                         f'ago) + still at sighting spot, pairing CLOSE → '
                         f'candidate honor at {distance:.2f}m (metric '
                         f'guard pending, #49)')
            return 'close'
        return None

    def _bridge_close_guard(self, distance):
        """#49 (bmv11 step28 实弹): 桥接配对 close 的米制守卫。

        配对 VLM (目击帧+当前帧并排) 能证 present, 证不了 close — bm16
        教训同源 ("远处清晰可见的真目标, 放大后尺寸线索被抹掉, 每次都判
        close")。当前决策帧有目标框但 <2% 画面 (D2 CLOSE 同一"目视级"
        门槛) = 无近证据 → 不得放行停止; 无框/大框/消融 → 放行 (旧行为)。
        bmv11 step28: 框 area 0.003 conf 0.86 @2.42m 被配对放行 fp —
        正确行为是沿 (已修的 #47) 接近锁走 ~1.4m 再停。
        返回 True=放行, False=拦下 (verdict 降 far → 拒停转接近)。
        """
        if not (getattr(self, '_ff', None) or {}).get(
                'bridge_metric_guard', True):
            return True
        cur_y = ((getattr(self, '_last_obs', None) or {}).get(
            'yolo_detection') or {})
        if not cur_y.get('target_found'):
            return True   # 当前帧无框 → 配对是唯一证据 (B1/P0 语义保持)
        tgt = getattr(self.simWrapper, 'target_name', '')
        for d in cur_y.get('all_detections') or []:
            if d.get('class_name') == tgt and d.get('bbox_norm'):
                x1, y1, x2, y2 = d['bbox_norm']
                a = max(0.0, (x2 - x1) * (y2 - y1))
                if a < 0.02:
                    logging.info(f'[BRIDGE-GUARD] pairing close but box '
                                 f'area {a:.3f} < 2% (present≠close, bm16) '
                                 f'→ veto stop at {distance:.2f}m, walk '
                                 f'to it (#49)')
                    return False
                return True   # 目视级大框 — 近证据在, 放行
        return True   # 无目标框坐标 → 保持旧行为

    def _no_box_stop_verdict(self, obs):
        """无框停票判据 (bm16 修复): 无框 = 无可靠距离证据 → 一律 far

        历史三连 fp 实弹 (bm11 4.39m / bm14 5.43m / bm16 4.96m): VLM 看
        图判 close 全错 — 远处清晰可见的真目标, 放大裁剪后尺寸线索被抹
        掉, 每次都判 "clearly visible and close"。VLM 能证 present, 证不
        了 close。距离的可靠证据只有自身传感器 (YOLO 框面积/深度网格),
        无框时一概不存在 → 拒停。真近目标的停由 ①框路径 (出框即面积/深
        度米制) 与 ③c 到场配对 (目击帧) 兜住。
        """
        logging.info('[STOP-EVIDENCE] no yolo box → FAR (no metric '
                     'evidence, VLM close is not distance evidence)')
        return 'far'

    def _reject_stop_and_approach(self, note, yolo_info=None, vote_level=False):
        """拒停 + 证据转化: 重置连票计数, 有框则建接近锁 (下步 AUTO-STEER)

        连续 done=1 计数被清后, agent 侧的连票停止条件 (≥2/≥3) 不会立即
        复燃 — 机器人把这股"想停"的冲动转化为朝目标的接近动作。
        vote_level: True 计 votes_cleared (票被拒), False 计 stop_vetoed
        (真 stop 被拦) — batch runner 分开计量两级守卫的拦截量。
        """
        logging.info(note)
        st = getattr(self, '_ep_stats', None)
        if st is not None:
            st['votes_cleared' if vote_level else 'stop_vetoed'] += 1
        ag = getattr(self, 'agent', None)
        if ag is not None and getattr(ag, 'stop_history', None):
            hist = ag.stop_history
            cleared = 0
            for i in range(len(hist) - 1, -1, -1):
                if hist[i] and cleared < 3:
                    hist[i] = False
                    cleared += 1
                else:
                    break
        if yolo_info is not None:
            rel = self._detection_bearing_deg(yolo_info)
            if rel is not None:
                self._approach_bearing = self._yaw_world() + np.radians(rel)
                self._approach_miss = 0
                logging.info(f'[STOP-EVIDENCE] approach lock re-aimed '
                             f'rel={rel:+.0f}deg → next step AUTO-STEER')
        elif (getattr(self, '_ff', None) or {}).get('reject_sight_lock', True):
            # === #25: 拒停不失忆 — 无框时用最近目击记录定位, 不进盲走。
            #     ep6 p16 step27: ARRIVAL-CONFIRM 拒停走 vote_level 分支
            #     (无 yolo_info) → 锁全丢 → 1.17m 处盲走 12 步。有框路径
            #     (3216 fp-rejected) 早有同语义的移交, 这里补齐无框侧。 ===
            locked = False
            try:
                sw = self.simWrapper
                sights = getattr(sw, 'target_sighting_nodes', None) or []
                if sights and getattr(sw, 'current_node', None) is not None \
                        and hasattr(sw, '_node_pos'):
                    s_step, s_node, _ = sights[-1]
                    if 0 <= self.step - s_step <= 8 \
                            and s_node != sw.current_node:
                        delta = np.array(sw._node_pos(s_node)) \
                            - np.array(sw._node_pos(sw.current_node))
                        if float(np.hypot(delta[0], delta[1])) > 0.05:
                            # 世界方位 (与 _yaw_world 同约定 atan2(dx, dz))
                            self._approach_bearing = float(
                                np.arctan2(delta[0], delta[1]))
                            self._approach_miss = 0
                            locked = True
                            logging.info(
                                f'[STOP-EVIDENCE] #25 sight-memory lock → '
                                f'walk back to sighting {self.step - s_step} '
                                f'step(s) ago (world bearing '
                                f'{np.degrees(self._approach_bearing):+.0f}'
                                f'deg, not blind wandering)')
            except Exception:
                locked = False
            if not locked:
                # 站在目击点上 (方位无定义) 或无新鲜目击 → P4 释锁重扫
                # 兜底: 下一步先 360° 重新定向 (ep6 step40 周期重扫在
                # 240° 找回 conf0.80 — 早 12 步做同样的事)
                self._pending_release_rescan = True
                logging.info('[STOP-EVIDENCE] no fresh sighting memory → '
                             'P4 release-rescan armed (re-orient, not wander)')

    def _proximity_rejected_stop_guard(self, agent_action, metrics, distance):
        """bm17 绕门修复: PROXIMITY 判 close 被评估侧拒后的两级守卫

        (动作级) 连票已确立的真 stop 必须过米制停票门 _stop_evidence —
        此前该路径的 stop 不进下方仲裁块 (not _force_stop 互斥), 拒绝又
        不翻 done, 连票 stop 直接以 fp 终结回合。现在: CLOSE → 尊重停止
        (交给 fp-救援); FAR/无框 → 翻回 done + 清连票 + 接近锁 (规则 2)。
        (投票级) 被拒的票就地清零, 防止跨拒绝累积成未来的连票。
        """
        if agent_action is PolarAction.stop:
            ev = self._stop_evidence(getattr(self, '_last_obs', None))
            if ev is not None and ev[0] == 'close':
                self._arbitration_close = True
                logging.info(f'[STOP-EVIDENCE] CLOSE (proximity path) → honor '
                             f'consecutive-stop at {distance:.2f}m '
                             f'(bbox depth backed)')
            else:
                self._reject_stop_and_approach(
                    f'[STOP-EVIDENCE] FAR (proximity path) → veto '
                    f'consecutive-stop at {distance:.2f}m, walk to it instead '
                    f'(bm17 gate bypass fixed)',
                    yolo_info=(ev[1] if ev is not None else None))
                metrics['done'] = False
                metrics['finish_status'] = 'running'
                self._arbitration_close = False
        else:
            # 投票级: 这一步投的票已被拒 → 清票, 别让幽灵票攒成连票
            self._reject_stop_and_approach(
                '[PROXIMITY] vote rejected → reset consecutive votes '
                '(fp semantics preserved)', vote_level=True)

    def _coverage_exhausted(self):
        """④ 覆盖完成判据: 地图无前沿灰格 (无可走未到访) 且已有足够覆盖

        灰格来源有两路 — 图边指向未访问邻居 (update_graph_frontier) 和
        深度楔形的可走未到访段 (update/update_live 60%~100%)。两者同时
        清零 = 可达范围内已看尽走尽 → 体面结束, 不再游荡 (G4 型)。
        """
        em = getattr(self.simWrapper, 'exploration_map', None)
        if em is None:
            return False
        try:
            regions = np.asarray(em.regions)
            known = int((regions > 0).sum())
            if known < 2000:            # 地图还没画开 (<5m²) 不算枯竭
                return False
            n_front = int((regions == 2).sum())
            if n_front > 25:            # 还有前沿 (≥0.06m² 可走未到访)
                return False
            n_seen = int(np.asarray(em.seen).sum())
            logging.info(f'[EXHAUSTED-CHECK] known={known} frontier={n_front} '
                         f'seen={n_seen} → coverage complete')
            return True
        except Exception:
            return False

    def _occlusion_peek(self, obs):
        """遮挡偷看: 朝"障碍背后从没看过"的阴影换一个角度

        无明确探索方向时的兜底 — 小目标最容易藏在台面/家具背后的
        视野阴影里 (深度图: OCCUPIED 背后 UNKNOWN)。选择方位最贴合
        阴影的选项 (侧移/复合优先, 换位比原地转看得更开), 走一步,
        下一步的照片自然覆盖阴影区。
        已偷看过的方位 (±20°) 不重复; 6 步内不连发。
        """
        sw = self.simWrapper
        em = getattr(sw, 'exploration_map', None)
        cur = getattr(sw, 'current_node', None)
        if em is None or cur is None or not hasattr(sw, '_node_pos'):
            return None
        try:
            p = sw._node_pos(cur)
            pockets = em.occlusion_pockets(float(p[0]), float(p[1]), max_r=4.0)
        except Exception:
            return None
        if not pockets:
            return None
        yaw = self._yaw_world()
        seen = getattr(self, '_peeked_bearings', [])

        def wrap(a):
            return (a + np.pi) % (2 * np.pi) - np.pi

        choice = None
        for bearing, dist, size in pockets:
            if any(abs(wrap(bearing - b2)) < np.radians(20) for b2 in seen):
                continue
            choice = (bearing, dist, size)
            break
        if choice is None:
            return None
        bearing, dist, size = choice
        # 选世界方位最贴合阴影的选项; 侧移/带走的复合小加分
        best_i, best_d = None, 1e9
        for i, o in enumerate(obs.get('edge_options', [])):
            wb = yaw + np.radians(self._option_bearing_deg(o))
            d = abs(np.degrees(wrap(bearing - wb)))
            if o.get('chain_type') in ('left', 'right') or \
                    (o.get('composite')
                     and any(e in ('forward', 'left', 'right')
                             for e, _ in o['composite'])):
                d -= 3.0
            if d < best_d:
                best_d, best_i = d, i
        if best_i is None or best_d > 90:
            return None
        self._last_peek_step = self.step
        self._peeked_bearings = (seen + [bearing])[-8:]
        rel = float(np.degrees(wrap(bearing - yaw)))
        return self._override_and_run(
            obs, best_i, 'OCCLUSION-PEEK',
            f'shadow {rel:+.0f}deg rel, obstacle {dist:.1f}m '
            f'(never seen behind it — change angle to check)')

    def _target_approachable(self):
        """朝目标还能不能更近 — 纯感知判断, 不用真值距离

        核心原则: 未探索 ≠ 不可达。UNKNOWN 正是要去探索的方向,
        只有"观测到的障碍 (OCCUPIED) 挡死 + 已知空间绕不动 + 没有
        任何前沿"才判不可达。依次查:
          1. 无方向信息 → 保守视为可达, 继续找
          2. 当前有含行走的动作选项 → 还有机动余地
          3. 探索地图走廊: 朝目标方向 1.5m 内无观测障碍 → 可达
             (UNKNOWN 段不算阻挡 — 去探索!)
          4. 已知空间内绕障可规划 → 可达
          5. 当前节点还有未探索的图边 (前沿) → 可达 (勇于探索)
        """
        sw = self.simWrapper
        bearing = getattr(self, '_approach_bearing', None)
        if bearing is None:
            sights = getattr(sw, 'target_sighting_nodes', []) or []
            cur = getattr(sw, 'current_node', None)
            if sights and cur is not None and hasattr(sw, '_node_pos'):
                try:
                    fp = np.array(sw._node_pos(cur))
                    tp = np.array(sw._node_pos(sights[-1][1]))
                    delta = tp - fp
                    if float(np.linalg.norm(delta)) > 0.3:
                        bearing = float(np.arctan2(delta[0], delta[1]))
                        # 目击方向起锁, 让下一步接近模式朝它走
                        self._approach_bearing = bearing
                        self._approach_miss = 0
                except Exception:
                    return True
            else:
                return True  # 无方向信息 → 保守继续找
        # 2) 还有行走选项 → 有机动余地
        obs = getattr(self, '_last_obs', None) or {}
        has_walk = any(
            o.get('chain_type') == 'forward'
            or (o.get('composite')
                and any(e == 'forward' for e, _ in o['composite']))
            for o in obs.get('edge_options', []))
        if has_walk:
            return True
        # 3) 探索地图走廊: 只认 OCCUPIED 为阻挡 (UNKNOWN = 去探索)
        em = getattr(sw, 'exploration_map', None)
        cur = getattr(sw, 'current_node', None)
        if em is not None and cur is not None and hasattr(sw, '_node_pos'):
            try:
                p = sw._node_pos(cur)
                free_ratio, first_block, unk = em.corridor_free_ratio(
                    float(p[0]), float(p[1]), bearing, max_r=5.0)
                if first_block >= 1.5:
                    logging.info(
                        f'[STOP-ARBITRATION] reachable: no observed obstacle '
                        f'to {first_block:.1f}m (free={free_ratio:.2f}, '
                        f'unknown={unk:.2f} — unexplored is explorable)')
                    return True
                logging.info(f'[STOP-ARBITRATION] corridor observed-blocked '
                             f'at {first_block:.1f}m (free={free_ratio:.2f})')
            except Exception:
                pass
        # 4) 已知空间绕障
        if self._plan_bearing_detour():
            logging.info('[STOP-ARBITRATION] reachable: known-space detour planned')
            return True
        # 5) 前沿: 当前节点还有未探索图边 → 勇于探索
        if hasattr(sw, '_get_unvisited_directions'):
            try:
                unv = sw._get_unvisited_directions(cur)
                if unv:
                    logging.info(f'[STOP-ARBITRATION] reachable: frontier '
                                 f'edges {list(unv.keys())} — go explore')
                    return True
            except Exception:
                pass
        return False

    def _invalidate_lock(self, reason, blacklist=True, drop_sighting=True,
                         loss_strike=True):
        """及时纠偏 + 误报黑名单

        blacklist=True  (VLM 'wrong'): 走近看过且 VLM 否认 → 拉黑目击节点
            + 接近终点 (半径 REJECT_RADIUS), 3 层拦截防回坑。
        blacklist=False (3步丢失): 接近中丢检测是正常现象 (视角变化/节点
            稀疏), 只解锁不拉黑 — 实测"丢失=误报"的假设曾把真目标区域
            拉黑导致失败。目击保留, 强制回访可再去确认一次。

        drop_sighting: 是否丢弃最近目击 (wrong→丢; 丢失→留一次回访机会)。
        loss_strike  : 是否参与"丢失两击"升级 — 修复1 的 wrong 路径传
            False (wrong strike 有自己的三级体系, 不与丢失 strike 混算)。
        """
        self._approach_bearing = None
        self._approach_miss = 99
        self._forced_return_path = []
        self._forced_return_dest = None   # #34: 旅程状态一并清
        sw = self.simWrapper
        # === 两击规则 (A2-②): 同一区域 15 步内第 2 次"3步丢失" → 升级拉黑 ===
        # 单次丢失≠误报 (只解锁); 但同一片区域反复 丢→锁→丢 说明检测源
        # 不可靠 (低置信闪烁/误报), 再给机会只会原地打转 → 按误报处理。
        escalated = False
        if not blacklist and loss_strike:
            cur = getattr(sw, 'current_node', None)
            cur_pos = None
            if cur is not None and hasattr(sw, '_node_pos'):
                cur_pos = np.array(sw._node_pos(cur))
            if cur_pos is not None:
                strikes = getattr(self, '_loss_strikes', [])
                recent = [s for s in strikes
                          if self.step - s[0] <= 15
                          and float(np.linalg.norm(cur_pos - np.array(s[1]))) <= 1.5]
                if recent:
                    blacklist = True
                    escalated = True
                    reason = (f'{reason}; TWO-STRIKES: 2nd loss in same area '
                              f'within 15 steps')
                strikes.append((self.step, cur_pos.tolist()))
                self._loss_strikes = strikes[-8:]  # 只留最近 8 次防膨胀
        sightings = getattr(sw, 'target_sighting_nodes', [])
        if sightings and (blacklist or drop_sighting):
            dropped = sightings.pop()
            if blacklist:
                if hasattr(sw, 'record_rejection'):
                    sw.record_rejection(dropped[1], f'false sighting: {reason}')
                else:
                    if not hasattr(sw, 'rejected_sightings'):
                        sw.rejected_sightings = []
                    sw.rejected_sightings.append((dropped[0], dropped[1]))
                # 接近终点也拉黑 (顺着假线索走到的地方已看过且被否认)
                cur = getattr(sw, 'current_node', None)
                if cur and cur != dropped[1] and hasattr(sw, 'record_rejection'):
                    sw.record_rejection(cur, f'approach endpoint, rejected ({reason})')
            self._last_returned_sighting_step = max(
                getattr(self, '_last_returned_sighting_step', -1), dropped[0])
        if blacklist:
            tag = 'TWO-STRIKES escalation' if escalated else 'blacklisting'
            logging.info(f'[CORRECT] {reason} — releasing lock + {tag}, '
                         f'explore elsewhere')
        else:
            logging.info(f'[CORRECT] {reason} — releasing lock only '
                         f'(NO blacklist: loss ≠ false sighting)')
        # === P4 (释锁即六向重扫): 置待重扫标志 — 下一步 _step_env 决策前
        #     消费 (预算 ≤2/回合)。释锁 = 旧接近方位信念全部作废, 沿旧
        #     记忆盲走正是 ep1 释锁后游荡 26 步的主因; 所有释锁路径统一
        #     置位 (含 stop-arbitration wrong — 拉黑后同样需要重新定向),
        #     成本由消费点的预算兜底 ===
        self._pending_release_rescan = True

    def _node_known(self, node):
        """节点是否在机器人已知空间内 (去先验判定)

        已访问 (visited_nodes) 或 探索地图 FREE 区 (深度观测覆盖)。
        真实机器人只有这两种知识 — 全图坐标属于数据集先验。
        """
        sw = self.simWrapper
        if node in (getattr(sw, 'visited_nodes', None) or {}):
            return True
        g = getattr(sw, 'nav_graph', None)
        em = getattr(sw, 'exploration_map', None)
        if g is not None and em is not None and node in g.get('nodes', {}):
            wp = g['nodes'][node]['world_pos']
            try:
                return em.cell_state(wp[0], wp[2]) in (1, 2)
            except Exception:
                return False
        return False

    def _plan_bearing_path(self, bearing, max_edges=12, min_progress=0.5,
                           max_path=None, tag='DETOUR'):
        """坐标级方位路径规划 (通用原语): 沿方位方向在已知空间内找最高分
        可达节点, Dijkstra 路径写入 _forced_return_path (GOTO_WAYPOINT
        执行通道)。评分:
          progress = (节点-当前)·方位单位向量, lateral = 垂直偏差
          score = progress - 0.6*lateral - 0.1*跳数
        只在已知空间内 BFS (去先验): 真实机器人没有全屋地图, 只能沿
        "走过 (visited) / 深度看过 (探索地图 FREE 区)"的子图规划。
        全图 Dijkstra 会规划出穿过未观测区域的长路 (实测 18 边)。
        方位约定与 _yaw_world 一致 (从 +z 起算朝 +x 为正):
          方向向量 (x,z) = (sin b, cos b)
        """
        g = getattr(self.simWrapper, 'nav_graph', None)
        cur = getattr(self.simWrapper, 'current_node', None)
        if not g or not cur or bearing is None:
            return False
        unit = np.array([np.sin(bearing), np.cos(bearing)])
        cur_pos = np.array([g['nodes'][cur]['world_pos'][0],
                            g['nodes'][cur]['world_pos'][2]])
        seen = {cur: 0}
        frontier = [cur]
        for _ in range(max_edges):
            nxt = []
            for u in frontier:
                for v in g['graph'].get(u, {}):
                    if v not in seen and self._node_known(v):
                        seen[v] = seen[u] + 1
                        nxt.append(v)
            frontier = nxt
            if not frontier:
                break
        best_node, best_score = None, -float('inf')
        for v, hops in seen.items():
            if v == cur:
                continue
            p = np.array([g['nodes'][v]['world_pos'][0],
                          g['nodes'][v]['world_pos'][2]])
            delta = p - cur_pos
            progress = float(delta @ unit)
            lateral = float(np.linalg.norm(delta - progress * unit))
            score = progress - 0.6 * lateral - 0.1 * hops
            # 黑名单惩罚: 绕障终点/途经不进已排除区域
            if hasattr(self.simWrapper, '_node_in_rejected') \
                    and self.simWrapper._node_in_rejected(v)[0]:
                score -= 1.5
            if score > best_score:
                best_score, best_node = score, v
        if best_node is None or best_score < min_progress:
            return False
        path = self._graph_path_to(cur, best_node)
        if not path:
            return False
        if max_path is not None and len(path) > max_path:
            return False
        self._forced_return_path = path
        # #34: 记录目的地节点 — 计划边半途失配时 _rescue_journey_edge
        #     据此重规划 (bmv4: 7-9 边前沿旅程两次死在 "no option for left")
        self._forced_return_dest = best_node
        logging.info(f'[{tag}] path to {best_node} ({len(path)} edges, '
                     f'score {best_score:.1f}m, bearing '
                     f'{np.degrees(bearing):.0f}°)')
        return True

    def _plan_bearing_detour(self, max_edges=12):
        """坐标级绕障: 沿锁定方位方向找最远可达节点, 规划图路径绕行

        世界坐标 = 导航图节点坐标 (定位已实现)。评分与已知空间约束见
        _plan_bearing_path (通用原语)。
        """
        if self._approach_bearing is None:
            return False
        return self._plan_bearing_path(self._approach_bearing,
                                       max_edges=max_edges, tag='DETOUR')

    def _area_exhausted(self, span=10, radius=2.5, max_distinct=10):
        """区域看尽判定 (转圈检测, 纯自身轨迹 — 不用真值)

        最近 span 步 (含 override 步, 见 _step_env 顶部的 _all_nodes
        记录) 机器人节点轨迹满足两个条件 = 困在同一小区域打转:
          1. 位置包围盒对角线 < radius (直线搜索 10 步 ~6.5m 长条,
             对角线大 → 不触发; 转圈对角线 = 圈直径)
          2. 节点种类 ≤ max_distinct
        max_distinct 复测校准 (ep6 实弹): 游荡窗 10 步仍有 9-10 个
        不同节点 (密集图 ~0.65m 节点间距, 小区域内"扫新节点"≠有
        进展) — 种类门 7 会挡住真实转圈, 默认 10 = 实际交给包围盒
        判定; 逼近锁定/活体检测另有前置条件挡住合法的贴面细查。
        注: _node_pos 返回 2 维 (x,z) — 索引只能到 [1] (ep4 复测
        step19 实弹 crash: ext[2] 越界炸掉整步)。
        """
        nodes = (getattr(self, '_all_nodes', None) or [])[-span:]
        if len(nodes) < span or any(n is None for n in nodes):
            return False
        sw = self.simWrapper
        if not hasattr(sw, '_node_pos'):
            return False
        if len(set(nodes)) > max_distinct:
            return False
        pos = []
        for n in nodes:
            try:
                pos.append(np.array(sw._node_pos(n)))
            except Exception:
                return False
        pos = np.array(pos)
        ext = pos.max(0) - pos.min(0)
        if float(np.hypot(ext[0], ext[1])) >= radius:
            return False
        # === P6 (换机位门): 判"看尽"须先在本区域换过 ≥2 个机位做过 360°
        #     扫描 (scan inquiry 的机位登记)。从没扫过/只扫过一个机位就
        #     跳走 = 灯下黑 — 单视角扫一遍漏掉小目标太容易。
        #     #26 修复 (p16 cola 实弹: 判尽门被堵 11 次): ①机位数聚合当前
        #     格 ±邻格 (3×3) — 2m 格 vs 困区直径 2.5m, 转圈轨迹横跨 2-5
        #     格, 按单格查永远 0 机位 → frontier_hop 出口被堵死; ②门只堵
        #     不引 → 挡下时置换机位补扫 pending (先平移一步再 360° 扫,
        #     _step_env 消费), 而不是干等 10 步一次的周期重扫 ===
        if (getattr(self, '_ff', None) or {}).get('scan_inquiry', True):
            try:
                cell = self.simWrapper._area_key(nodes[-1])
                positions = getattr(self, '_scan_positions_by_cell', None) or {}
                n_pos = sum(len(v) for c, v in positions.items()
                            if abs(c[0] - cell[0]) <= 1
                            and abs(c[1] - cell[1]) <= 1)
                if n_pos < 2:
                    # #26b: 门挡下 → 引导去补扫 (换机位), 不堵死等周期重扫
                    if (getattr(self, '_ff', None) or {}).get(
                            'viewpoint_rescan', True) \
                            and not getattr(self, '_pending_viewpoint_shift',
                                            False) \
                            and not getattr(self, '_pending_viewpoint_rescan',
                                            False) \
                            and getattr(self, '_viewpoint_rescans', 0) < 2:
                        self._pending_viewpoint_shift = True
                        logging.info(
                            f'[AREA-EXHAUSTED] zone {cell} only {n_pos} '
                            f'viewpoint(s) (<2) — NOT exhausted; #26 viewpoint '
                            f'shift + rescan armed (先换机位再扫, 不干等)')
                    else:
                        logging.info(
                            f'[AREA-EXHAUSTED] zone {cell} 360°-scanned from '
                            f'only {n_pos} position(s) (<2) — NOT declaring '
                            f'exhausted (P6 viewpoint gate, 灯下黑防护)')
                    return False
            except Exception:
                pass
        return True

    def _pick_frontier_target(self, cur_pos, cells, em, min_dist, max_dist):
        """#32: 前沿格选点 — 已清区排除 + 语义方位偏好 → (wx,wz,d,mode)

        bmv2 mahatma 实弹: 旧版"最近灰格"永远 2.5m 外的隔壁 zone, 跳出
        又进、(-1,-2) 被 CLEAR 两次, 全程困在同一功能区 (用户: "一直在
        同一个区域确认也没啥意义")。两修:
          ① 落点 zone 在 _zone_clear_cells (判过无价值) → 出池 — 不把
            机器人派回自己刚判尽的区;
          ② VLM 扫描的 LIKELY-BEARING (世界方位) ±90° 半平面内优先取
            最近前沿 — "困惑就去更可能放目标物品的方位" (ep6 教训保持:
            只排序探索落点, 不作转向命令/不加 bonus)。
        全被排除/无语义 → 回退全局最近 (旧行为, 不死锁)。
        mode: 'semantic' | 'cleared-filtered' | 'nearest'
        """
        cleared = getattr(self, '_zone_clear_cells', None) or set()
        cell_size = getattr(self.simWrapper, 'AREA_CELL', 2.0)
        sem = None
        if (getattr(self, '_ff', None) or {}).get('semantic_frontier', True):
            # #38: 前沿落点排序也用聚合方位 (与 SEMANTIC-GO 同源, 不打架)
            agg_f = self._semantic_bearing_agg()
            if agg_f is not None:
                sem = agg_f[0]

        cands = []          # [(d, wx, wz, in_cleared, in_neg, in_scanned, aligned)]
        # === #51c (bmd01/d03 实弹, 用户指令"走过的区域其他区确认完才可再
        #     去"): 前沿候选分层 — ① 全新 zone (从未 360° 扫过) 最优;
        #     ② 扫过未否决; ③ 否决区 (identity-reject ≥2) 只在全部其他
        #     选项耗尽后才作为兜底。d03 病灶: 前沿池只认"站立点", 走过
        #     路过扫过的走廊边格子仍是灰格 → 最近灰格永远在已走过的
        #     功能区里冒出来, 152° 前沿跳把机器人送回起点旁。数据源 =
        #     自身扫描机位登记/身份判决, 零样本合规 ===
        neg_zones = getattr(self, '_zone_negative_zones', None) or set()
        scanned_cells = set((getattr(self, '_scan_positions_by_cell', None)
                             or {}).keys())
        _gate = (getattr(self, '_ff', None) or {}).get('zone_revisit_gate', True)
        stride = max(1, len(cells) // 400)
        for gz, gx in cells[::stride]:
            try:
                wx, wz = em._to_world(int(gz), int(gx))
            except Exception:
                continue
            d = float(np.hypot(wx - cur_pos[0], wz - cur_pos[1]))
            if not (min_dist <= d < max_dist):
                continue
            zone = (int(wx // cell_size), int(wz // cell_size))
            in_cleared = zone in cleared
            in_neg = _gate and zone in neg_zones
            in_scanned = _gate and (not in_neg) and zone in scanned_cells
            aligned = False
            if sem is not None:
                diff = abs((np.degrees(np.arctan2(wx - cur_pos[0],
                                                  wz - cur_pos[1]))
                            - sem + 180) % 360 - 180)
                aligned = diff <= 90.0
            cands.append((d, wx, wz, in_cleared, in_neg, in_scanned, aligned))

        alive = [c for c in cands if not c[3]]
        # #51c 分层回访门: cleared 出池 (现状) → 新 zone 池 → 扫过池 →
        # 全量兜底 (否决区只有这里才可能入选; 不死锁)
        pools = [alive]
        if _gate:
            pools = [[c for c in alive if not c[4] and not c[5]],
                     [c for c in alive if not c[4]],
                     alive]
        pool = next((p for p in pools if p), [])
        if not pool:
            pool = cands
        if not pool:
            return None
        if sem is not None:
            aligned = [c for c in pool if c[6]]
            if aligned:
                d, wx, wz = min(aligned, key=lambda c: c[0])[:3]
                return (wx, wz, d, 'semantic')
        d, wx, wz = min(pool, key=lambda c: c[0])[:3]
        return (wx, wz, d,
                'cleared-filtered' if len(alive) < len(cands) else 'nearest')

    def _plan_frontier_hop(self, min_dist=2.5, max_dist=10.0):
        """阶段1 前沿跳跃: 探索地图灰格 (可走未到访) → 已知空间 Dijkstra
        路径 → GOTO_WAYPOINT 通道强制离开旧区域

        区域看尽 (转圈) 的根治: 反应式压制 (掉头冷却/crop 兜底) 只能
        减少浪费, 机器人仍困在旧区域; 地图驱动的前沿跳跃直接把机器人
        派往"深度确认可走但从未到访"的方向。前沿格两路来源: 深度射线
        远端 + 图边指向未访问邻居 (update_graph_frontier)。
        落点选择 (#32): _pick_frontier_target — 已清区排除 + 语义方位。
        """
        sw = self.simWrapper
        em = getattr(sw, 'exploration_map', None)
        cur = getattr(sw, 'current_node', None)
        if em is None or cur is None or not hasattr(sw, '_node_pos'):
            return False
        try:
            cur_pos = np.array(sw._node_pos(cur))
        except Exception:
            return False
        try:
            cells = np.argwhere(np.asarray(em.regions) == 2)
        except Exception:
            return False
        if not len(cells):
            return False
        picked = self._pick_frontier_target(cur_pos, cells, em,
                                            min_dist, max_dist)
        if picked is None:
            return False
        wx, wz, best_d, mode = picked
        bearing = float(np.arctan2(wx - cur_pos[0], wz - cur_pos[1]))
        if self._plan_bearing_path(bearing, max_edges=10, min_progress=1.5,
                                   max_path=10, tag='FRONTIER-HOP'):
            self._last_frontier_hop_step = self.step
            logging.info(f'[FRONTIER-HOP] area exhausted → frontier '
                         f'{best_d:.1f}m away (bearing '
                         f'{np.degrees(bearing):.0f}°, mode={mode}), '
                         f'hopping out')
            return True
        return False

    def _yaw_world(self):
        """当前节点的世界朝向角 (弧度, +z 前向约定, 与深度模块一致)"""
        node = getattr(self.simWrapper, 'current_node', None)
        g = getattr(self.simWrapper, 'nav_graph', None)
        if node and g and node in g['nodes']:
            d = g['nodes'][node]['direction']
            return float(np.arctan2(d[0], d[2]))
        return 0.0

    # 数据集预渲染照片的水平视场 (1920×1080, f≈975px) — 相机内参, 零样本合规
    PHOTO_HFOV_DEG = 89.0

    def _yolo_frame_offset_deg(self, yolo_info, target):
        """#47b: YOLO 目击框中心相对画面中心的水平偏移 (度, 右正)。

        照片相机朝向 ≠ 目标方位: 目标常在画面边缘 (middle_right ≈ +41°,
        bmv10 step42-43 GT 取证)。照片 HFOV≈89° → 偏移 = (cx−0.5)×89°,
        夹到半视界内。找 all_detections 里目标类 (与 check_target 同选
        法), 无框信息 → None (caller 回落 0 = 旧行为)。
        """
        if not yolo_info:
            return None
        cx = None
        for d in yolo_info.get('all_detections') or []:
            if d.get('class_name') == target:
                cx = d.get('center_x')
                break
        if cx is None:
            cx = yolo_info.get('center_x')
        if cx is None:
            return None
        off = (float(cx) - 0.5) * self.PHOTO_HFOV_DEG
        half = self.PHOTO_HFOV_DEG / 2.0
        return float(max(-half, min(half, off)))

    def _scan_lock_bearing_rad(self, scan_angle_deg):
        """#47: 扫描目击 → 世界系接近锁方位 = 起始朝向 + 扫描角 + 画面内偏移。

        旧公式 = 锁时刻 _yaw_world() + scan_angle, bmv10 两次锁全 180° 级
        错 (终距 3.99m, 真目标区被 TWO-STRIKES 拉黑):
        ① 47a — rescan 取 obs 的途中 null 步曾把 current_node 偷换成同
           位置任意朝向变体 (step38b: −28.3° → 153.2°, 错 181.6°; wrapper
           已改途中纯读, 这里再以 _warmup_scan 入口记的基准朝向为准,
           不依赖锁时刻节点);
        ② 47b — 相机朝向 ≠ 目标方位, 漏画面内框偏移 (step42: middle_right
           +41°, 锁 304° 真 344.9°≡−15.1° ≈ GT −13.8°)。
        消融 lock_bearing_fix=False → 精确回落旧公式 (实时 _yaw_world,
        无偏移)。warmup 与 rescan 两条锁路径共用。
        """
        ff = (getattr(self, '_ff', None) or {}).get('lock_bearing_fix', True)
        base = getattr(self, '_scan_base_yaw_deg', None) if ff else None
        if base is None:
            base = float(np.degrees(self._yaw_world()))
        off = (getattr(self, '_scan_hit_offset_deg', None) or 0.0) if ff else 0.0
        return float(np.radians(base) + np.radians(scan_angle_deg)
                     + np.radians(off))

    @staticmethod
    def _option_bearing_deg(opt):
        """edge_option → 相对相机前向的近似方位角 (左负右正, 顺时针为正)"""
        if opt.get('composite'):
            et, cnt = opt['composite'][0]
        else:
            et, cnt = opt.get('chain_type'), opt.get('chain_count', 1)
        base = {'rotate_cw': 30.0, 'rotate_ccw': -30.0, 'forward': 0.0,
                'backward': 180.0, 'left': -90.0, 'right': 90.0}.get(et, 0.0)
        return base * (cnt or 1)

    def _option_world_bearing_deg(self, opt):
        """#44 (bmv8 step32-35 实弹): 选项真实世界方位 — 落点位移几何

        旧模型 yaw + 0°(forward) 假设 forward 边沿相机朝向位移 — 走廊图
        不成立: 节点 canonical 朝向 −37° 而 forward 边位移正西 (−90°),
        伺服相信 "aligned 1deg" 实偏 ~50° → 目标出画 miss 释锁 → #37
        带回 → 再偏 → TWO-STRIKES 拉黑真目标区 (0.67/0.34 目击 GT 夹角
        ≤1° 全是真目标)。改用图几何: 平移/链式/复合选项 = 当前→落点
        世界位移方位; 纯旋转 (位移 <0.15m) = 落点节点 canonical 朝向
        (转过去面对)。断链/无几何 → None (调用方回落旧模型)。
        全部来自机器人自身里程计图 (nav_graph 即自身行走数据), 零样本合规。
        """
        try:
            g = getattr(self.simWrapper, 'nav_graph', None) or {}
            nodes = g.get('nodes', {})
            cur = getattr(self.simWrapper, 'current_node', None)
            if not cur or cur not in nodes:
                return None
            legs = opt['composite'] if opt.get('composite') \
                else [(opt.get('chain_type'), opt.get('chain_count', 1) or 1)]
            node = cur
            for et, cnt in legs:
                for _ in range(cnt or 1):
                    nxt = next((t for t, e in (g.get('graph', {})
                                               .get(node, {})).items()
                                if e.get('edge_type') == et), None)
                    if nxt is None:
                        break
                    node = nxt
            if node == cur or node not in nodes:
                return None
            cp = np.array(nodes[cur]['world_pos'], float)
            lp = np.array(nodes[node]['world_pos'], float)
            dx, dz = lp[0] - cp[0], lp[2] - cp[2]
            if float(np.hypot(dx, dz)) < 0.15:   # 纯旋转: 同位异向
                dv = np.array(nodes[node].get('direction', [0, 0, 1]), float)
                return float(np.degrees(np.arctan2(dv[0], dv[2])))
            return float(np.degrees(np.arctan2(dx, dz)))
        except Exception:
            return None

    def _detection_bearing_deg(self, yolo_info):
        """bbox 中心 → 精确相对方位角 (右为正, 度)。拿不到框返回 None

        旧 9 宫格 'position' 每格横跨 ~44° (FOV 131°) — 同是 "left" 的两个
        检测可能差 40°+; 用 bbox 连续坐标 × 相机 FOV 直接算真实方向。
        """
        try:
            fov = float(self.cfg.get('sensor_cfg', {}).get('fov', 131))
        except Exception:
            fov = 131.0
        for d in yolo_info.get('all_detections', []):
            if d.get('class_name') == getattr(self.simWrapper, 'target_name', None) \
                    and d.get('bbox_norm'):
                b = d['bbox_norm']
                cx = (b[0] + b[2]) / 2.0
                return (cx - 0.5) * fov
        return None

    def _pick_steer_option_precise(self, obs, rel_deg, area_ratio):
        """精确伺服选边: 检测方位 + 框大小(距离) 连续评分, 返回 edge idx

        rel_deg   : 检测中心相对前向方位角 (右正)
        area_ratio: 框面积占比 — >0.30 近 / 0.10~0.30 中 / <0.10 远
          近 → 只转不走 (转向+前进会冲过头, 实测绕圈根因)
          远/中 → 方向贴合的复合 (转+走) 优先, 一步完成逼近
        同侧过滤: 目标明显在右 (>+12°) 时排除左转/左移, 反之亦然。
        """
        edge_opts = obs.get('edge_options', [])
        near = area_ratio > 0.30
        far = area_ratio < 0.10
        # 死区: 已基本对准 → 不再原地转 (30° 转边必过冲, 实测 step2 右转29°
        # → step3 左转30° 的来回打摆就来自这里)。
        # 分级: 上一步刚转向过 → 纯转死区扩到 30° (反转只留给真跑偏;
        # 转边自带 0.3~0.5m 位移, ±25° 内的反转指令基本都是位移噪声)
        ret = getattr(self.simWrapper, 'recent_edge_types', [])
        last_rot = ret[-1] if ret and ret[-1] in ('rotate_cw', 'rotate_ccw') else None
        rot_dead = 30.0 if last_rot else 20.0
        # #44: 选项方位优先用落点位移几何 (旧朝向启发在走廊图上可偏 ~50°)
        use_geo = (getattr(self, '_ff', None) or {}).get('servo_geometry', True)
        yaw_deg = float(np.degrees(self._yaw_world()))
        best_i, best_score = None, 1e9
        for i, o in enumerate(edge_opts):
            if use_geo:
                wb = self._option_world_bearing_deg(o)
                ob = (((wb - yaw_deg) + 180.0) % 360.0 - 180.0) \
                    if wb is not None else self._option_bearing_deg(o)
            else:
                ob = self._option_bearing_deg(o)
            pure_rot = (not o.get('composite')
                        and o.get('chain_type') in ('rotate_cw', 'rotate_ccw'))
            if pure_rot and abs(rel_deg) <= rot_dead:
                continue
            if rel_deg > 12 and ob < -5:
                continue  # 目标在右, 左向动作无意义
            if rel_deg < -12 and ob > 5:
                continue
            score = abs(ob - rel_deg)
            if pure_rot and last_rot and o.get('chain_type') != last_rot:
                score += 8.0   # 反向连转惩罚 (右转后立刻左转 = 打摆)
            has_walk = (o.get('chain_type') == 'forward'
                        or (o.get('composite')
                            and any(e == 'forward' for e, _ in o['composite'])))
            if near and has_walk:
                score += 25.0
            elif has_walk and abs(ob - rel_deg) <= 45.0:
                score -= (6.0 if far else 3.0)
            if score < best_score:
                best_score, best_i = score, i
        return best_i

    def _zoom_bbox_frame(self, obs, yolo_info, margin_mult=1.0, min_out=448):
        """P5 (修7c'): bbox 裁剪放大帧 — 远距小目标确认问询的看不清修复

        ep6 step37: 2.42m 外米盒 area=0.006 (画面 0.6%), 全幅问询 VLM
        根本看不清 → IDENTITY unsure → fastpath 单票误停; ep1 step25:
        3.5m 外 conf0.98 可乐瓶 area=0.003, 全幅红框 PROXIMITY 被判
        wrong (真目击吃 L1 处罚)。以 bbox 为中心、四周留 margin_mult×
        bbox 边长的窗口裁剪, 放大到大边 ≥min_out (LANCZOS, ≤4×)。
        返回 (zoomed_rgb, box_xyxy_px_in_zoom) 或 None (无框/无图)。

        #43 (bmv7 step27, 用户: "大概画面的3%其实已经非常大了"):
        margin_mult 2.5 时窗口 ≈ 36× bbox 面积 — conf 0.41 米袋在
        448 窗里只占 ~3%, VLM 看到的还是整个厨房 → 误判 no。收紧到
        1.0 (窗口 ≈ 9× bbox, 目标占窗 ~11%), 上下文剩"一格货架"级别。
        """
        try:
            from PIL import Image as PILImage
            rgb = obs.get('color_sensor')
            if rgb is None:
                return None
            det = None
            for d in yolo_info.get('all_detections', []):
                if d.get('class_name') == getattr(
                        self.simWrapper, 'target_name', '') \
                        and d.get('bbox_norm') \
                        and (det is None or d.get('confidence', 0)
                             >= det.get('confidence', 0)):
                    det = d
            if det is None:
                return None
            x1, y1, x2, y2 = [float(v) for v in det['bbox_norm']]
            img = PILImage.fromarray(
                rgb[:, :, :3] if rgb.ndim == 3 else rgb)
            W, H = img.size
            bx1, by1, bx2, by2 = x1 * W, y1 * H, x2 * W, y2 * H
            m = margin_mult * max(bx2 - bx1, by2 - by1, 4.0)
            cx1, cy1 = max(0, int(bx1 - m)), max(0, int(by1 - m))
            cx2, cy2 = min(W, int(bx2 + m)), min(H, int(by2 + m))
            crop = img.crop((cx1, cy1, cx2, cy2))
            cw, ch = crop.size
            if cw < 8 or ch < 8:
                return None
            scale = min(max(min_out / cw, min_out / ch), 4.0)
            if scale > 1.0:
                crop = crop.resize((int(cw * scale), int(ch * scale)),
                                   PILImage.LANCZOS)
            zbox = ((bx1 - cx1) * scale, (by1 - cy1) * scale,
                    (bx2 - cx1) * scale, (by2 - cy1) * scale)
            return np.array(crop), zbox
        except Exception as e:
            logging.debug(f'[ZOOM-CONFIRM] crop failed: {e}')
            return None

    def _vlm_identity_check(self, obs, yolo_info):
        """VLM 看 YOLO 框内图确认身份: 框是不是目标本身 (不是同类邻居/背景)

        YOLO 中置信 (0.30~0.60) 且目标很小时, 分类并不稳 (0.51 的可乐瓶
        检测可能只是厨房台面上的酱料瓶)。把框放大画出来问 VLM, 返回:
        'yes' / 'no' / 'unsure' (解析失败按 'unsure' 处理, 走近再验)。
        P5 (zoom_confirm): 有框时改为 bbox 裁剪放大帧问询 — 全幅图上
        0.6% 画面的目标 VLM 看不清 (ep6 step37 unsure 的真因)。
        """
        dets = [d for d in yolo_info.get('all_detections', [])
                if d.get('class_name') == getattr(self.simWrapper, 'target_name', None)
                and d.get('bbox_norm')]
        if not dets:
            return 'unsure'
        b = max(dets, key=lambda d: d.get('confidence', 0))['bbox_norm']
        conf = yolo_info.get('confidence', 0)
        tname = getattr(self.simWrapper, 'target_name', 'target').replace('_', ' ')
        zoom_note = ''
        zoom = self._zoom_bbox_frame(obs, yolo_info) \
            if (getattr(self, '_ff', None) or {}).get('zoom_confirm', True) \
            else None
        if zoom is not None:
            z_rgb, (zx1, zy1, zx2, zy2) = zoom
            img = z_rgb.copy()
            H, W = img.shape[:2]
            cv2.rectangle(img, (int(zx1), int(zy1)), (int(zx2), int(zy2)),
                          (0, 0, 255), max(3, W // 150))
            zoom_note = ("The image is a ZOOMED-IN crop centered on the "
                         "detected object (detector's box in red).\n")
            logging.info(f'[IDENTITY] zoomed confirm frame {W}x{H} '
                         f'(bbox area {((b[2]-b[0])*(b[3]-b[1])):.4f})')
        else:
            img = obs['color_sensor'].copy()
            H, W = img.shape[:2]
            x1, y1, x2, y2 = (int(b[0] * W), int(b[1] * H),
                              int(b[2] * W), int(b[3] * H))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), max(3, W // 200))
        prompt = (
            f"{zoom_note}"
            f"The red box marks an object that an object detector believes is a "
            f"'{tname}' (confidence {conf:.0%}).\n"
            f"Look ONLY at the object inside the red box (ignore everything else).\n"
            f"Is the object inside the red box actually a {tname}?\n"
            f"Answer with JSON only:\n"
            f'{{"verdict": "yes"}}  — definitely a {tname}\n'
            f'{{"verdict": "no"}}   — definitely NOT a {tname}\n'
            f'{{"verdict": "unsure"}} — cannot tell from this view'
        )
        try:
            resp = self.agent.actionVLM.call_chat(0, [np.array(img)], prompt)
        except Exception as e:
            logging.warning(f'[IDENTITY] VLM call failed: {e}')
            return 'unsure'
        m = re.search(r'"verdict"\s*:\s*"(yes|no|unsure)"', str(resp).lower())
        verdict = m.group(1) if m else 'unsure'
        logging.info(f'[IDENTITY] conf={conf:.2f} → VLM verdict: {verdict}')
        return verdict

    def _pick_confirm_option(self, obs, yolo_info):
        """身份不确定 → 选一个"看得更清"的动作: 近 → 走近, 远 → 转向对准

        - 框很小 (area < 0.02, 距离远) → 走近最有效: 方位 ±45° 内的
          平移/复合优先
        - 框不小 → 换角度: ≤35° 的转向把目标转到画面中央再看一次
        返回 edge idx 或 None。
        """
        area = float(yolo_info.get('area_ratio', 0) or 0)
        rel = self._detection_bearing_deg(yolo_info) or 0.0
        best_i, best_score = None, 1e9
        for i, o in enumerate(obs.get('edge_options', [])):
            ob = self._option_bearing_deg(o)
            has_walk = (o.get('chain_type') == 'forward'
                        or (o.get('composite')
                            and any(e == 'forward' for e, _ in o['composite'])))
            if area < 0.02:
                # 远: 靠近优先 (平移/复合), 方向别偏离目标太多
                if not has_walk or abs(ob - rel) > 45.0:
                    continue
                score = abs(ob - rel) * 0.5
            else:
                # 近: 换角度 — 小幅转向把目标摆到画面中间
                if has_walk or abs(ob) > 35.0 or ob * rel < 0:
                    continue
                score = abs(ob - rel)
            if score < best_score:
                best_score, best_i = score, i
        return best_i

    def _pick_reposition_option(self, obs):
        """低置信误检换机位动作 (用户规则, ep1 灯下黑): 优先平移
        (left/right — 位置变朝向不变, 视野平移一格), 其次 forward。
        纯旋转不算换机位 (灯下黑要的是新站位, 不是新朝向)。
        """
        opts = obs.get('edge_options', [])
        for et in ('left', 'right', 'forward'):
            for i, o in enumerate(opts):
                if o.get('chain_type') == et and not o.get('composite'):
                    return i
        for i, o in enumerate(opts):
            if o.get('composite') and any(
                    e in ('left', 'right', 'forward')
                    for e, _ in o['composite']):
                return i
        return None

    def _consume_viewpoint_shift(self, obs):
        """#26b: 判尽门挡下 (#26a 聚合后仍 <2 机位) 的换机位补扫。

        用户思路原话: "避免灯下黑问题, 也可以换个机位来再扫描一次啊" —
        门挡下不该干等 10 步一次的周期重扫, 而是主动: 本步平移一格到新
        站位 (灯下黑要新站位不是新朝向), 下一步原地 360° 六向重扫并登记
        新机位 (P4 执行通道, 预算独立 — p16 cola 实弹 P4 预算开局耗尽
        后门被堵 11 次直到回合结束)。
        预算 ≤2/回合, 只在真正平移时消耗; 无平移边 → 不耗预算等周期。
        返回 override 动作或 None。
        """
        self._pending_viewpoint_shift = False
        if not (getattr(self, '_ff', None) or {}).get(
                'viewpoint_rescan', True):
            return None
        if getattr(self, '_viewpoint_rescans', 0) >= 2:
            return None
        idx = self._pick_reposition_option(obs)
        if idx is None:
            logging.info('[VIEWPOINT-SHIFT] no translation edge — wait for '
                         'periodic rescan (#26, no budget spent)')
            return None
        self._viewpoint_rescans = getattr(self, '_viewpoint_rescans', 0) + 1
        self._pending_viewpoint_rescan = True   # 下一步六向扫 (P4 通道)
        logging.info(f'[VIEWPOINT-SHIFT] step {getattr(self, "step", 0)}: '
                     f'zone under-covered (<2 scan viewpoints) → shift '
                     f'to a new standpoint, 360° rescan from there next '
                     f'step (#26, budget {self._viewpoint_rescans}/2)')
        return self._override_and_run(
            obs, idx, 'VIEWPOINT-SHIFT',
            'under-covered zone — new standpoint then 360° re-scan '
            '(#26 viewpoint coverage, 灯下黑防护)')

    def _reposition_after_reject(self, obs, conf, note):
        """IDENTITY no + 低置信 → 换机位重看一次 (ep1: goal 就在 0.06m
        脚边, 当前机位看不清被否; 平移一格新视角自然重新检测)。

        预算每回合 ≤4 次; 无可换动作/超预算 → None (接受误检, 正常探索)。
        返回 override 动作或 None。
        """
        if not (getattr(self, '_ff', None) or {}).get(
                'low_conf_reposition', True):
            return None
        if getattr(self, '_reposition_calls', 0) >= 4:
            return None
        idx = self._pick_reposition_option(obs)
        if idx is None:
            return None
        self._reposition_calls = getattr(self, '_reposition_calls', 0) + 1
        return self._override_and_run(
            obs, idx, 'REPOSITION',
            f'low-conf ({conf:.2f}) sighting rejected but target may be '
            f'"hidden in plain sight" ({note}) — shift position and '
            f're-check from a new viewpoint')

    def _approach_step(self, obs):
        """方案3 全自动接近: 返回覆盖动作 (edge_idx 已设置) 或 None

        - 当前帧 YOLO 看到目标 (conf ≥ 0.3): 伺服朝它走, 并更新锁定方位
        - 否则: 选与锁定方位最对齐的图边 (绕障: 没有正对边时选偏差最小的)
        - 连续 miss > 10 步 → 放弃锁定 (周期重扫描会重新锁定)
        """
        yolo_info = obs.get('yolo_detection') or {}
        if yolo_info.get('target_found') and yolo_info.get('confidence', 0) >= 0.3:
            pos = yolo_info.get('position', '')
            conf = yolo_info.get('confidence', 0)
            area = float(yolo_info.get('area_ratio', 0) or 0)
            rel = self._detection_bearing_deg(yolo_info)
            # === 🕵️ 中置信小目标 → VLM 看框身份确认 (用户规则) ===
            # YOLO 置信一般且目标小 → 框给 VLM 复核:
            #   no     → 基本确定不是: 该节点记为已排除, 解锁去探索新地方
            #   unsure → 拿不准: 走近 (远) / 换角度 (近) 再确认
            #   yes    → 正常伺服接近
            # 同节点 10 步内不重复问; 3 步内不连续问 (防 VLM 调用风暴)
            if 0.30 <= conf < 0.60 and area < 0.10:
                cur = getattr(self.simWrapper, 'current_node', None)
                if getattr(self, '_identity_reject', {}).get(cur, -99) > self.step - 10:
                    logging.info(f'[IDENTITY] node {cur} already rejected — explore')
                    return None
                if self.step - getattr(self, '_last_identity_step', -99) >= 3:
                    self._last_identity_step = self.step
                    verdict = self._vlm_identity_check(obs, yolo_info)
                    if verdict == 'no':
                        # === #42 (bmv7 step27 实弹): 低置信否认 ≈ unsure ===
                        # conf 0.41 真米袋被 no → 记 strike → 与接近链丢失
                        # 凑 TWO-STRIKES 拉黑 → 扫描重检被压制 → 语义改道
                        # 4.67m max_steps (用户: "YOLO都看到了，不过去确认
                        # 下吗? 应该去前进二次确认下")。低置信时 VLM 的 no
                        # 证据不足, 前进/换角再看 (≤3 次/节点, 超限回落
                        # 原拒绝路径 — 保住 ep8 类真误报的止损)。
                        if (conf < 0.50 or area < 0.01) and (getattr(self, '_ff', None)
                                                            or {}).get(
                                                                'identity_lowconf_confirm', True):
                            used = getattr(self, '_lowconf_confirm_used', {}).get(cur, 0)
                            if used < 3:
                                self._lowconf_confirm_used = {
                                    **getattr(self, '_lowconf_confirm_used', {}),
                                    cur: used + 1}
                                ci = self._pick_confirm_option(obs, yolo_info)
                                if ci is not None:
                                    logging.info(
                                        f'[IDENTITY] no @ conf {conf:.2f} '
                                        f'area {area:.3f} <0.50/<1% — '
                                        f'not trusted (YOLO saw it) → forward '
                                        f'second confirmation ({used + 1}/3) (#42/#51a)')
                                    return self._override_and_run(
                                        obs, ci, 'CONFIRM',
                                        f'identity no at LOW conf/tiny box '
                                        f'({conf:.2f}/{area:.3f}) — '
                                        f'YOLO detection present, go closer and '
                                        f're-check (second confirmation) (#42)')
                                logging.info(
                                    f'[IDENTITY] no @ conf {conf:.2f} but no '
                                    f'confirm option — fall through to reject (#42)')
                            else:
                                logging.info(
                                    f'[IDENTITY] no @ conf {conf:.2f}, confirm '
                                    f'budget {used}/3 exhausted at {cur} → '
                                    f'reject as before (#42)')
                        self._identity_reject = {
                            **getattr(self, '_identity_reject', {}), cur: self.step}
                        self._zone_neg_strike()   # #51b: 区域级否决记账
                        # #42b: wrong 判决走自己的三级体系 (_identity_reject
                        # 节点冷却 + identity_gate_v2), 不混入"丢失两击" —
                        # bmv7 step27 正是 identity-no strike × 接近链丢失
                        # strike 拼成 TWO-STRIKES 黑名单 (docstring 2230 行
                        # 本就规定 wrong 路径传 False, 此处漏传)
                        self._invalidate_lock(
                            f'VLM identity: NOT the target (YOLO conf {conf:.2f})',
                            blacklist=False, drop_sighting=True,
                            loss_strike=False)
                        logging.info('[IDENTITY] verdict=no → unlock, explore new areas')
                        # 用户规则: 低置信误检先换机位重看 (ep1 灯下黑 —
                        # 0.06m 脚边目标当前机位看不清; 换站位新视角重检)
                        rep = self._reposition_after_reject(obs, conf, 'ep1')
                        if rep is not None:
                            return rep
                        return None
                    if verdict == 'unsure':
                        # === 修复2 Layer 1 (vote_fastpath): 停票链保护 —
                        # 上一步已投 done=1 (streak≥1) 且当前帧有框 (本分支
                        # 前提即 target_found) → 跳过 CONFIRM 相机动作, 走
                        # 正常步让票链在同一视野上继续 (设计稿 §4) ===
                        _vote_protect = bool(
                            (getattr(self, '_ff', None) or {}).get(
                                'vote_fastpath', True)
                            and (getattr(getattr(self, 'agent', None),
                                         'stop_history', None)
                                 or [False])[-1])
                        if _vote_protect:
                            logging.info(
                                f'[VOTE-PROTECT] step={self.step} stop '
                                f'streak≥1 + box in frame → CONFIRM camera '
                                f'move suppressed, keep vote chain on same '
                                f'view')
                        else:
                            ci = self._pick_confirm_option(obs, yolo_info)
                            if ci is not None:
                                return self._override_and_run(
                                    obs, ci, 'CONFIRM',
                                    f'identity unsure (conf={conf:.2f}, '
                                    f'area={area:.2f}) → better view '
                                    f'(closer/new angle)')
                            logging.info('[IDENTITY] unsure but no confirm '
                                         'option — proceed with servo')
            if rel is not None:
                # 精确锁定: 用检测真实方向建世界系锁 (旧代码从选中选项反推,
                # 只有 30° 选项粒度, 复合动作还会带偏锁定)
                self._approach_bearing = self._yaw_world() + np.radians(rel)
                self._approach_miss = 0
                self._approach_last_area = area
                band = 'near' if area > 0.30 else ('far' if area < 0.10 else 'mid')
                idx = self._pick_steer_option_precise(obs, rel, area)
                if idx is not None:
                    return self._override_and_run(
                        obs, idx, 'AUTO-STEER',
                        f"YOLO {pos} conf={conf:.2f} → rel={rel:+.0f}deg "
                        f"area={area:.2f} ({band})")
            else:
                idx = self._pick_steer_option(obs, pos)
                if idx is not None:
                    rel_o = self._option_bearing_deg(obs['edge_options'][idx])
                    self._approach_bearing = self._yaw_world() + np.radians(rel_o)
                    self._approach_miss = 0
                    return self._override_and_run(
                        obs, idx, 'AUTO-STEER',
                        f"YOLO {pos} conf={conf:.2f} (coarse 9-grid)")
            # 无可用伺服动作 (如节点没有对应边) → 用检测精确方位重瞄锁定,
            # 落到下面的对齐选边 (绕圈由图边兜底, 不回落给 VLM)
            horiz = rel if rel is not None else \
                (-20.0 if 'left' in pos else (20.0 if 'right' in pos else 0.0))
            self._approach_bearing = self._yaw_world() + np.radians(horiz)
            self._approach_miss = 0
            logging.info(f'[AUTO-STEER] no steer option at {pos} '
                         f'(rel={rel}), re-aiming lock')
        else:
            # 无活体检测 → 按锁定方位选边
            self._approach_miss += 1
        # 远距追踪容忍更长盲走 (bm12 实测: 4m+ 距离 YOLO 闪烁, 3 步丢检
        # 即释锁 → 接近被反复打断, 追到 5m 又被弹回 9m)。锁定是世界系的,
        # 末次检测远 (area<0.10) 时盲走预算 3→6; 近距维持 3 (近处丢检=真丢)
        miss_budget = 6 if getattr(self, '_approach_last_area', 1.0) < 0.10 else 3
        if self._approach_miss >= miss_budget:
            # 及时纠偏: 连续多步看不到目标 → 解除锁定。注意不拉黑 —
            # 接近中丢检测多为视角/节点稀疏所致, 拉黑会误杀真目标区域
            self._invalidate_lock(
                f'target lost for {self._approach_miss} steps — detection stale',
                blacklist=False, drop_sighting=False)
            # #37: 锁丢了 ≠ 证据蒸发 — 扫描目击 ≤12 步内 → 走回目击视角
            # 重看一次 (须在 _invalidate_lock 之后: 它会清 _forced_return_path)
            self._arm_reacquire_after_miss()
            return None
        # 绕圈保护: 最近 12 步内同一节点出现 ≥3 次 → 立即重扫描重新锁定
        cur_node = getattr(self.simWrapper, 'current_node', None)
        if not hasattr(self, '_approach_nodes'):
            self._approach_nodes = []
        self._approach_nodes.append(cur_node)
        if len(self._approach_nodes) > 12:
            self._approach_nodes = self._approach_nodes[-12:]
        if cur_node is not None and self._approach_nodes.count(cur_node) >= 3 \
                and self.step not in getattr(self, '_periodic_scanned_at', set()):
            logging.info(f'[APPROACH] cycle detected at node {cur_node}')
            # 先尝试坐标级绕障 (沿锁定方位的最远可达节点)
            if self._plan_bearing_detour():
                return self._execute_forced_return(obs)
            # 绕障规划失败 → 重扫描重新锁定
            self._periodic_scanned_at.add(self.step)
            self._re_scan(self.step, reset_memory=False)
            if hasattr(self, '_scan_obs') and self._scan_obs is not None:
                self._scan_obs = None  # 让本步回落到正常流程, 下一步用新扫描
            self._approach_bearing = None
            return None
        edge_opts = obs.get('edge_options', [])
        # 无路可走保护: 选项里没有任何行走成分 (全被墙/家具挡死, 实测
        # NO_SAFE_DIRECTION 节点只剩旋转) → 与其在锁定方位上原地转,
        # 直接坐标级绕障 (图上找朝锁定方向的最远可达节点绕过去)
        has_walk_opt = any(
            (o.get('chain_type') == 'forward'
             or (o.get('composite')
                 and any(e == 'forward' for e, _ in o['composite'])))
            for o in edge_opts)
        if not has_walk_opt and self._plan_bearing_detour():
            logging.info('[APPROACH] no walkable option here — bearing detour')
            return self._execute_forced_return(obs)
        yaw = self._yaw_world()
        # === #53a (d01r2 实弹): 锁方位与当前朝向差 >45° — 对齐选边一步
        # 只能挪 30° 且每步涨 miss (#52a 拆除角度折算拐杖后暴露);
        # miss 预算耗尽释锁 → #37 拉回同位重锁同方位 → 无限循环
        # (d01r2 step25-31: 锁 −124° vs 朝向 −3°, 3 步耗尽 ×2 轮)。
        # 大角度差改沿锁方位图路径走 (带平移, GOTO 通道 — 上方调用点
        # `not self._forced_return_path` 旅程执行期接近让路, 锁保持
        # 世界系不丢); 规划失败回落原对齐选边。活体重瞄 ≤44.7°
        # (半视界) 天然不触发, 只救陈旧扫描锁 ===
        d_lock = (np.degrees(self._approach_bearing) - np.degrees(yaw)
                  + 180.0) % 360.0 - 180.0
        if abs(d_lock) > 45.0 and (getattr(self, '_ff', None) or {}).get(
                'approach_bearing_path', True) \
                and self._plan_bearing_path(
                    self._approach_bearing, max_edges=8, min_progress=0.5,
                    max_path=8, tag='APPROACH-TURN'):
            logging.info(f'[APPROACH-TURN] lock '
                         f'{np.degrees(self._approach_bearing):.0f}° vs heading '
                         f'{np.degrees(yaw):.0f}° ({d_lock:+.0f}°) → graph path '
                         f'{len(self._forced_return_path)} edges instead of '
                         f'30°-edge nibbling (#53a)')
            return self._execute_forced_return(obs)
        # #44: 世界方位对齐用落点位移几何 (bmv8: 旧模型 "aligned 1deg"
        # 实偏 ~50° 向西走出画 → 释锁循环 → TWO-STRIKES 拉黑真目标区)
        use_geo = (getattr(self, '_ff', None) or {}).get('servo_geometry', True)
        best_i, best_d = None, 181.0
        for i, o in enumerate(edge_opts):
            if use_geo:
                wbd = self._option_world_bearing_deg(o)
                wb = np.radians(wbd) if wbd is not None \
                    else yaw + np.radians(self._option_bearing_deg(o))
            else:
                wb = yaw + np.radians(self._option_bearing_deg(o))
            d = np.degrees(abs(((self._approach_bearing - wb + np.pi) % (2 * np.pi)) - np.pi))
            # 抗重复/抗原地打转: 与上一步相同动作 +5°, 180° 掉头 +10° 惩罚
            if i == getattr(self, '_last_approach_idx', -1):
                d += 5.0
            if abs(self._option_bearing_deg(o)) >= 150:
                d += 10.0
            if d < best_d:
                best_d, best_i = d, i
        if best_i is not None and best_d <= 90:
            self._last_approach_idx = best_i
            return self._override_and_run(
                obs, best_i, 'APPROACH', f'aligned {best_d:.0f}deg from locked bearing')
        return None

    def _override_and_run(self, obs, idx, tag, note=''):
        """跑完整 VLM 流程 (保留日志/上下文) 后覆盖动作为 idx"""
        # 接近/确认/偷看等主动追击期间设置标志 — wrapper 的 HARD RULE 3
        # 口袋逃逸不劫持追击动作 (否则正在逼近目标时被拉走去探索)
        self.simWrapper._approach_active = True
        # P3 (修7a): miss 同步 — 1b/1c 只在 miss==0 (持续目击的精确伺服)
        # 时旁路; miss>0 (跟丢计数) 后恢复硬规则, 防圈保护不失守
        self.simWrapper._approach_miss = getattr(self, '_approach_miss', 99)
        # B⑦: 机制意图标签 — wrapper 1b/1c 豁免 (peek 的旋转是"转过去看",
        # 不是扫视打转) + FINAL-EXEC trace (consume-once, 单点覆盖 PEEK/
        # REPOSITION/FRONTIER-HOP/FORCED-RETURN 全部机制)
        self.simWrapper._action_intent = tag
        try:
            agent_action = super()._step_env(obs)
        finally:
            self.simWrapper._approach_active = False
        if agent_action is not None:
            # === 修复2 Layer 2 防吞票 (ep2 step25 型): 停票票在 override
            #     内部产生 — 快通道 stop 不得被 _attach_exec_idx 改写成
            #     覆盖动作 (那正是 ep2 票链被 CONFIRM 掐断的机制)。
            #     super 返回 stop 单例 ⟺ 米制门已否决 (done=True 时 super
            #     返回 None) → 与正常步尾部同规则转当步移动。 ===
            if agent_action is PolarAction.stop \
                    and getattr(self.agent, 'last_stop_was_fastpath', False):
                agent_action = self._vetoed_stop_action(obs)
                logging.info(f'[FASTPATH] step={self.step}: stop request '
                             f'vetoed by metric gate inside {tag} override → '
                             f'walk closer (votes reset)')
                return agent_action
            # 物理生效 (bm7/bm12 大坑修复): wrapper 按 edge_idx 执行,
            # 之前只改账面 → 世界一直按 VLM raw 选择动, 日志/图在说谎
            agent_action = self._attach_exec_idx(agent_action, idx, obs)
            raw_idx = None
            if len(self.df) and 'action_number' in self.df.columns:
                raw_idx = self.df.iloc[-1]['action_number']
                self.df.iloc[-1, self.df.columns.get_loc('action_number')] = idx
            logging.info(f'[{tag}] step {self.step}: executing [{idx}] {note}')
            # 选中图重画为实际执行动作 + 覆盖角标 (图与下一帧画面一致)
            self._redraw_chosen_as_executed(obs, idx, tag, raw_idx)
        return agent_action

    def _redraw_chosen_as_executed(self, obs, executed_idx, tag, raw_idx=None):
        """env 覆盖步: 重画 color_sensor_chosen.png 为实际执行的动作。

        agent.step 里画的选中图是 VLM 原始选择; env 层覆盖 (warmup 自动
        动作 / APPROACH / OCCLUSION-PEEK 等) 发生在图片落盘之后 — 不重画
        的话图会说谎 (实测: step0 图画左移, 实际右转 60°)。纯可视化,
        不影响任何决策逻辑。图片已在 super()._step_env() 内保存, 这里
        直接覆盖同名 PNG。
        """
        try:
            ep_dir = (f'logs/{self.outer_run_name}/{self.inner_run_name}/'
                      f'{self.curr_run_name}/step{self.step}')
            png = f'{ep_dir}/color_sensor_chosen.png'
            if not os.path.exists(png):
                return
            img = obs['color_sensor'].copy()
            self.agent._draw_chosen_action_nav_demo(
                img, executed_idx, obs.get('avail_actions', ''), self.step)
            # 底部红色角标: 说明这一步被 env 覆盖
            H, W = img.shape[:2]
            s = H / 480.0
            bh = int(30 * s)
            roi = img[H - bh:H, 0:W]
            blk = np.zeros_like(roi)
            cv2.addWeighted(blk, 0.75, roi, 0.25, 0, roi)
            txt = f'AUTO OVERRIDE ({tag}): executed [{executed_idx}]'
            if raw_idx is not None and raw_idx != executed_idx:
                txt += f' | VLM raw choice was [{raw_idx}]'
            cv2.putText(img, txt, (int(8 * s), H - int(9 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, (0, 0, 255),
                        max(1, round(1 * s)), cv2.LINE_AA)
            cv2.imwrite(png, img)
            # details.txt 同步补一行 (warmup 块已有, 其余覆盖路径补齐审计链)
            dt = f'{ep_dir}/details.txt'
            if os.path.exists(dt):
                with open(dt, 'a') as f:
                    raw_note = f' (VLM raw choice [{raw_idx}] overridden)' \
                        if raw_idx is not None and raw_idx != executed_idx else ''
                    f.write(f'\nOVERRIDE ({tag}): executed action [{executed_idx}]'
                            f'{raw_note}\n')
        except Exception as e:
            logging.debug(f'[redraw chosen] skipped: {e}')

    def _attach_exec_idx(self, agent_action, idx, obs=None):
        """把实际执行编号挂到返回动作上 (物理通道: wrapper 按 edge_idx 执行)

        bm7/bm12 实测大坑: env 层覆盖只改账面 (df/图/日志), 从不设置
        edge_idx → 世界一直按 VLM raw 选择动, "executed [X]" 全在说谎
        (bm7 step4: 标 executed[5], 实际动了 raw[6]); stop 单例不带
        edge_idx 还会掉进 wrapper 的 vlm_a=0 回退 = 180° turn_around
        (bm12 step22→23)。这里统一: stop/null 单例 → 新建动作对象
        (绝不在单例上挂属性, 会永久污染), 再挂 edge_idx。

        #52b (d01r step43-46, #27 复合残留同族): 覆盖 idx 时 action 上
        残留的 rotate_steps/forward_steps (VLM raw 折算写入) 不清 →
        wrapper 用残留步数覆盖被选选项的 chain_count, "30° 微调"物理
        转成 180° 甩头。覆盖 = 全量替换: 带 obs 时把 steps 同步成被选
        选项自身的 chain_count (复合选项两残留全清, wrapper 走
        composite 分支不读它们)。
        """
        if agent_action is PolarAction.stop or agent_action is PolarAction.null:
            agent_action = PolarAction(0, 0)
        try:
            agent_action.edge_idx = int(idx)
        except Exception:
            agent_action.edge_idx = 0
        if obs is not None:
            try:
                o = (obs.get('edge_options') or [])[agent_action.edge_idx]
            except Exception:
                o = None
            if o is not None:
                if o.get('chain_type') in ('rotate_cw', 'rotate_ccw'):
                    agent_action.rotate_steps = o.get('chain_count', 1)
                    if hasattr(agent_action, 'forward_steps'):
                        del agent_action.forward_steps
                elif o.get('chain_type') == 'forward':
                    agent_action.forward_steps = o.get('chain_count', 1)
                    if hasattr(agent_action, 'rotate_steps'):
                        del agent_action.rotate_steps
                else:   # composite / 其他: 残留全清
                    if hasattr(agent_action, 'rotate_steps'):
                        del agent_action.rotate_steps
                    if hasattr(agent_action, 'forward_steps'):
                        del agent_action.forward_steps
        return agent_action

    def _vetoed_stop_action(self, obs):
        """被否决的停止票当步的动作: stop 单例不能直接回传 (无 edge_idx
        → wrapper 回退 [0] turn_around 180°)。转为当步就走:
        1) 有接近锁 → 锁定方位最对齐的选项 ("拒停转接近"当步就接近)
        2) 否则 → VLM 本步原始移动选择 (df 账面号)
        3) 都没有 → [0] (与旧行为一致)
        """
        edge_opts = obs.get('edge_options') or []
        idx = None
        if getattr(self, '_approach_bearing', None) is not None:
            try:
                yaw = self._yaw_world()
                best_i, best_d = None, 181.0
                for i, o in enumerate(edge_opts):
                    wb = yaw + np.radians(self._option_bearing_deg(o))
                    d = np.degrees(abs(((self._approach_bearing - wb + np.pi)
                                        % (2 * np.pi)) - np.pi))
                    if i == getattr(self, '_last_approach_idx', -1):
                        d += 5.0
                    if abs(self._option_bearing_deg(o)) >= 150:
                        d += 10.0
                    if d < best_d:
                        best_d, best_i = d, i
                if best_i is not None and best_d <= 90:
                    idx = best_i
            except Exception:
                idx = None
        if idx is None and len(self.df) and 'action_number' in self.df.columns:
            try:
                n = int(self.df.iloc[-1]['action_number'])
                if 0 <= n < len(edge_opts):
                    idx = n
            except Exception:
                idx = None
        act = self._attach_exec_idx(PolarAction(0, 0),
                                    idx if idx is not None else 0, obs)
        logging.info(f'[STOP-VETO] vetoed stop → execute move [{act.edge_idx}] '
                     f'this step (not 180° fallback)')
        return act

    def _pick_steer_option(self, obs, yolo_pos):
        """YOLO 目标位置 → 朝向它的 edge_option 编号 (纵向距离感知)

        bottom_*  = 目标很近 → 只原地转向 (转向+前进会冲过头, 实测绕圈)
        middle_*  = 中距离   → 先转再走 (复合) 或 转
        top_*     = 较远     → 先转再走 (复合, 优先走更远) 或 转
        *_center  = 前进
        """
        edge_opts = obs.get('edge_options', [])
        if 'left' in yolo_pos:
            rot = 'rotate_ccw'
        elif 'right' in yolo_pos:
            rot = 'rotate_cw'
        else:
            rot = None
        if rot is None:
            for i, o in enumerate(edge_opts):
                if o.get('chain_type') == 'forward' and o.get('chain_count') == 1 \
                        and not o.get('composite'):
                    return i
            return None
        if 'bottom' in yolo_pos:
            # 很近: 只转不走
            for i, o in enumerate(edge_opts):
                if o.get('chain_type') == rot and o.get('chain_count') == 1 \
                        and not o.get('composite'):
                    return i
            return None
        if 'top' in yolo_pos:
            # 较远: 优先复合×2 (转+走两步)
            for i, o in enumerate(edge_opts):
                if o.get('composite') and o['composite'][0] == (rot, 1) \
                        and len(o['composite']) >= 2 and o['composite'][1] == ('forward', 2):
                    return i
        # 中距离: 复合 (转+走)
        for i, o in enumerate(edge_opts):
            if o.get('composite') and o['composite'][0] == (rot, 1):
                return i
        for i, o in enumerate(edge_opts):
            if o.get('chain_type') == rot and o.get('chain_count') == 1 \
                    and not o.get('composite'):
                return i
        return None

    def _graph_path_to(self, from_node, to_node):
        """导航图上 from_node → to_node 的最短路径, 返回 [edge_type, ...]

        Dijkstra, 供目击强制回访/语义旅程/救援重规划共用。
        #46 (bmv9 实弹): 代价从"边距离(米)"改为"步数等价" — 图里旋转边
        distance 均值 0.014m (128 条 literally 0.0), 按米计价 = 旋转免费,
        Dijkstra 拿免费旋转洪泛路径: v9 step18 救援重规划 17 边含 7×rotate_ccw
        ≈ 原地转 ~330° (step19-33 烧 15 步近零位移, 全程零目击终距 6.95m)。
        回合预算烧的是步数: 旋转 1 步/边; 平移可 #41 并步 (≤3 边/步) →
        0.45 步/边。真实图复算 v9 病例: 17 边/est ~15 步 → 14 边/est 10 步,
        且旋转只在真需要改朝向处出现; #45 同位转向补全路径不变 (4×rotate_ccw)。
        ablation: --no-journey-stepcost 回落旧米数代价。
        """
        g = getattr(self.simWrapper, 'nav_graph', None)
        if not g or from_node not in g['graph'] or to_node not in g['graph']:
            return None
        use_step = (getattr(self, '_ff', None) or {}).get(
            'journey_stepcost', True)
        import heapq
        dist = {from_node: 0.0}
        prev = {}
        pq = [(0.0, from_node)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float('inf')):
                continue
            if u == to_node:
                break
            for v, e in g['graph'].get(u, {}).items():
                et = e['edge_type']
                c = (1.0 if 'rotate' in et else 0.45) if use_step \
                    else e['distance']
                nd = d + c
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    prev[v] = (u, et)
                    heapq.heappush(pq, (nd, v))
        if to_node not in prev:
            return None
        ets = []
        cur = to_node
        while cur != from_node:
            u, et = prev[cur]
            ets.append(et)
            cur = u
        return ets[::-1]

    def _p6_resolve_suspicious(self, summary, step_num):
        """P6 (修9) 执行端: SCAN_SUSPICIOUS 方位裁决 → 返回转向角或 None

        #33/#34/#35/#36 四层语义 (逐次实弹演化):
          已查桶      → 不转身销账 (bmv3 沙发/电视死循环);
          已查 ≥2 桶  → 新可疑不追, 查尽离区 (bmv5: VLM 每扫必出新可疑,
                       追不完; 两处亲查无检出 = 查尽, 用户判据);
          转身 0 步   → 当场销账不空转 (bmv5 step6: final=rotate_cw x0);
          未查过     → 转身查看 + 记桶出队。
        zone 查尽 (无可疑/全查过/≥2 桶无果) + 本次扫描有 LIKELY-BEARING
        → #35 SEMANTIC-GO: 方向立即变旅程 (用户指令"scan 以后就是要找出
        探索的思路, 而不是继续困惑")。ep6 教训不破: 非目击/无 +20 bonus,
        是困惑出口的方向承诺, 路径仍是已知图验证过的节点。
        """
        m_sus = re.search(r'SCAN_SUSPICIOUS:\s*(\d+)', summary)
        _cell = None
        try:
            _cell = self.simWrapper._area_key(
                self.simWrapper.current_node)
        except Exception:
            pass
        _insp = ((getattr(self, '_inspected_spots', None) or {})
                 .get(_cell, set())) if _cell else set()

        def _mark_inspected(bucket):
            """#33/#34: 记已查桶 + 同桶排队条目出队"""
            if not _cell:
                return
            if not hasattr(self, '_inspected_spots'):
                self._inspected_spots = {}
            self._inspected_spots.setdefault(_cell, set()).add(bucket)
            if not hasattr(self, '_suspicious_spots'):
                self._suspicious_spots = {}
            self._suspicious_spots[_cell] = [
                e for e in (self._suspicious_spots.get(_cell) or [])
                if e[1] // 60 != bucket]

        def _zone_queue_empty():
            return not ((getattr(self, '_suspicious_spots', None)
                         or {}).get(_cell) or []) if _cell else True

        def _arm_semantic_go(reason):
            if not (getattr(self, '_ff', None) or {}).get(
                    'semantic_go', True):
                return
            agg = self._semantic_bearing_agg()
            if agg is None \
                    or int(agg[1]) != int(step_num) \
                    or self._approach_bearing is not None \
                    or getattr(self, '_forced_return_path', None):
                return
            if self._plan_bearing_path(
                    np.radians(float(agg[0])), max_edges=8,
                    min_progress=1.5, tag='SEMANTIC-GO'):
                # #38: 记录已承诺方向 — 翻向守卫的比对基准
                self._last_committed_bearing = float(agg[0]) % 360.0
                self._last_committed_step = int(step_num)
                logging.info(f'[SEMANTIC-GO] {reason} → 扫描定方向 '
                             f'world {float(agg[0]) % 360:.0f}° '
                             f'(聚合 {len(getattr(self, "_semantic_votes", []) or [])} 票), '
                             f'旅程 {len(self._forced_return_path)} 边 '
                             f'(困惑出口 = 承诺方向, 不是再磨)')

        _sus_ang = int(m_sus.group(1)) if m_sus else None
        # 与 auto-turn 同式的转身步数 (330-359° 也是 0 步)
        _rot_needed = ((360 - _sus_ang) // 30
                       if (_sus_ang or 0) > 180
                       else (_sus_ang or 0) // 30)
        # === #39 (用户拍板"语义优先, 可疑顺路查"): 可疑点与语义先验同
        #     级都是推测 (照片里目标不可见, 可见则 YOLO 通道更早触发);
        #     4 局实弹 P6 追查 0 命中, 真命中全来自扫描本身。本次扫描
        #     语义方向新鲜时: 顺路可疑 (±60°, 查它=往语义方向走) 照旧
        #     转身查看; 不顺路 → 留 zone 记忆不追, 方向成行。 ===
        _agg39 = self._semantic_bearing_agg() \
            if (getattr(self, '_ff', None) or {}).get(
                'semantic_over_suspicious', True) else None
        _prior_fresh = _agg39 is not None and int(_agg39[1]) == int(step_num)
        _on_way = False
        if _sus_ang is not None and _agg39 is not None:
            try:
                _sw_deg = (float(np.degrees(self._yaw_world()))
                           + float(_sus_ang)) % 360.0
                _on_way = abs((_sw_deg - _agg39[0] + 180.0) % 360.0
                              - 180.0) <= 60.0
            except Exception:
                _on_way = False
        if _sus_ang is not None and _prior_fresh and not _on_way:
            logging.info(f'[RE-SCAN] P6: suspicious {_sus_ang}° 不顺路 '
                         f'(语义 world {_agg39[0] % 360:.0f}°) → 留记忆'
                         f'不追, 方向成行 (#39 语义优先)')
            _arm_semantic_go('语义先验 > 不顺路可疑')
            return None
        if _sus_ang is not None and _sus_ang // 60 in _insp:
            logging.info(f'[RE-SCAN] P6: suspicious '
                         f'{_sus_ang}° 已查过 → 不再转身 '
                         f'(销账, 等待离区)')
            if _zone_queue_empty() or _prior_fresh:
                _arm_semantic_go('zone 可疑点全查过' if _zone_queue_empty()
                                 else '语义先验 > 剩余排队可疑')
            return None
        if _sus_ang is not None and len(_insp) >= 2:
            # #35: 已查过 2 个方位无果 → 不追第 3 个"新可疑"
            logging.info(f'[RE-SCAN] P6: zone 已查 '
                         f'{sorted(_insp)} 桶无果, 新可疑 '
                         f'{_sus_ang}° 不再追 → 查尽离区')
            _arm_semantic_go('zone 已查 2+ 方位无果')
            return None
        if _sus_ang is not None and _rot_needed == 0:
            # #36 (bmv5 step6 实弹): <30° 的可疑就在当前视野正前 —
            #   转身 0 步 = 白烧一步, 本帧 YOLO 已看过 → 当场销账
            logging.info(f'[RE-SCAN] P6: suspicious {_sus_ang}° 在'
                         f'当前视野内 (<30°) → 不转身, 当场销账')
            _mark_inspected(_sus_ang // 60)
            if _zone_queue_empty() or _prior_fresh:
                _arm_semantic_go('zone 可疑点全查过' if _zone_queue_empty()
                                 else '语义先验 > 剩余排队可疑')
            return None
        if m_sus:
            new_angle = int(m_sus.group(1))
            logging.info(f'[RE-SCAN] P6 inquiry: suspicious spot at '
                         f'{new_angle}° → auto-turn to inspect '
                         f'(no lock — suspicion ≠ confirmation)')
            # #33: 记"已查" (60° 桶) — 转过看过后若再报同方位可疑 →
            #   销账; 若真有目标, YOLO/目击会接管且 #31 守卫挡住 CLEAR
            _mark_inspected(new_angle // 60)
            return new_angle
        # 本次扫描无可疑 → 扫描的 LIKELY-BEARING 直接成行
        _arm_semantic_go('扫描无可疑点')
        return None

    def _re_scan(self, step_num, reset_memory=True):
        """Mid-navigation 360° re-scan when VLM is stuck. Same as warmup but resets state.
        Saves PNG frames + GIF with step number in filename for visibility.

        reset_memory=False (周期性重定向): 保留 visited 访问记忆, 防止"一切重新变新"
        导致 VLM 放弃已搜索区域记忆而漂移。
        """
        self._last_scan_step = step_num   # 扫描机制 v2: 任意扫描重置保底计时
        logging.info(f'[RE-SCAN] Starting mid-navigation re-scan at step {step_num}...')
        saved_node = self.simWrapper.current_node

        # Run the scan (reuse warmup logic; 帧目录 tag=step{N}_rescan —
        # 不再经过 warmup_scan 中转, 初始扫描帧永不被覆盖)
        tag = f'step{step_num}_rescan'
        summary, target_priority = self._warmup_scan(step_tag=tag)

        # RE-SCAN banner: 各 rescan 帧目录的 color_sensor.png 加红条
        # (GIF 里可辨识哪段是途中重扫描)
        ep_dir = f'logs/{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'
        if os.path.isdir(f'{ep_dir}/{tag}'):
            from PIL import Image, ImageDraw
            for fi in range(6):
                png = f'{ep_dir}/{tag}_{fi:02d}/color_sensor.png'
                if os.path.exists(png):
                    img = Image.open(png)
                    draw = ImageDraw.Draw(img)
                    draw.rectangle([(0, 0), (img.width, 36)], fill=(200, 50, 50))
                    draw.text((5, 4),
                              f"RE-SCAN at step {step_num}  |  direction {fi*60} deg",
                              fill=(255, 255, 255))
                    img.save(png)
            logging.info(f'[RE-SCAN] Saved {ep_dir}/{tag}/ + step frames '
                         f'for GIF visibility')

        if summary:
            if reset_memory:
                # Reset visited-node counts to give fresh exploration perspective
                self.simWrapper.visited_nodes = {}
                self.simWrapper.visited_edges = {}
                self.simWrapper.target_sighting_nodes = []
                self.simWrapper.memory['step_count'] = 0  # reset memory counter
                self.simWrapper.memory['rooms'] = {}
                self.simWrapper.memory['short_term'] = []

            # Inject re-scan analysis
            obs = self.simWrapper.step(PolarAction.null)
            existing_mem = obs.get('memory_context', '')
            obs['memory_context'] = f'## 🔄 MID-NAVIGATION RE-SCAN\n{summary}\n' + existing_mem

            import re
            # === 解析目标方向 (Bug修复: 之前只解析 FIRST_DIRECTION 行,
            #     丢弃了 YOLO 扫描发现块 target_priority, 导致 97% 的发现
            #     没有任何自动动作。现在: YOLO 发现优先, FIRST_DIRECTION 兜底) ===
            new_angle = None
            yolo_conf = None
            if target_priority:
                m = re.search(r'at \*\*(\d+)°', target_priority)
                mc = re.search(r'confidence \*\*([\d.]+)%', target_priority)
                if m:
                    new_angle = int(m.group(1))
                    yolo_conf = float(mc.group(1)) / 100.0 if mc else 0.0
            if new_angle is None:
                fd_match = re.search(r'FIRST_DIRECTION:\s*(\d+)°', summary)
                if fd_match:
                    new_angle = int(fd_match.group(1))
            # === P6 (问询层兜底): YOLO/FIRST_DIRECTION 都没给方位时,
            #     SCAN_SUSPICIOUS 的可疑方位作转向目标 — 只转身去看,
            #     不锁 APPROACH bearing (yolo_conf 保持 None: 可疑 ≠
            #     证实)。summary 含完整 VLM response, 行总在 ===
            if new_angle is None \
                    and (getattr(self, '_ff', None) or {}).get(
                        'scan_inquiry', True):
                new_angle = self._p6_resolve_suspicious(summary, step_num)

            if new_angle is not None:
                # 方案3: 锁定目标方位 (世界坐标系) → 进入全自动接近模式
                # 扫描方向约定: 顺时针 = 正角 (与 rotate_cw/右转一致)
                if yolo_conf is not None and yolo_conf >= 0.5:
                    # #47: 锁方位 = 扫描起始朝向 + 扫描角 + 画面内框偏移
                    #     (旧两错: 锁时刻 _yaw_world 被 null 步偷换 + 漏
                    #     画面内偏移 — bmv10 两次锁全 180° 级错)
                    self._approach_bearing = self._scan_lock_bearing_rad(new_angle)
                    self._approach_miss = 0
                    # #53b: 扫描锁补记目击框面积 — 远距锁 (area<0.10) miss
                    # 预算 3→6 (d01r2 step28: 锁 −124° 需 4+ 步转身, 3 步
                    # 预算耗尽释锁 → #37 拉回同位重锁同方位无限循环)
                    self._approach_last_area = getattr(
                        self, '_scan_hit_area', None) or 1.0
                    logging.info(f'[APPROACH-LOCK] bearing locked: scan angle '
                                 f'{new_angle}°, frame offset '
                                 f'{(getattr(self, "_scan_hit_offset_deg", None) or 0.0):+.0f}°, '
                                 f'conf {yolo_conf:.2f}, world bearing '
                                 f'{np.degrees(self._approach_bearing):.0f}° (#47)')
                # #47: 瞄准角 = 扫描角 + 画面内偏移 (YOLO 命中才有框;
                #     P6 可疑/FIRST_DIRECTION 无框 → 原角度)。把目标转到
                #     画面中央而非留在 ±44.7° 半视界边缘 (边缘处伺服易丢)
                aim = new_angle
                if yolo_conf is not None and (getattr(self, '_ff', None) or {})\
                        .get('lock_bearing_fix', True):
                    aim = int(round((new_angle + (getattr(
                        self, '_scan_hit_offset_deg', None) or 0.0)) % 360))
                rot_steps = (360 - aim) // 30 if aim > 180 else aim // 30
                if rot_steps == 0:
                    # #36: <30° (或 >330°) 无需转身 — 不设 0 步 auto-action
                    #     白烧一步 (bmv5 step6: final=rotate_cw x0 空转)
                    logging.info(f'[RE-SCAN] direction {aim}° already '
                                 f'ahead — no turn needed')
                else:
                    target_et = 'rotate_ccw' if aim > 180 else 'rotate_cw'
                    edge_opts = obs.get('edge_options', [])
                    for i, opt in enumerate(edge_opts):
                        if opt.get('chain_type') == target_et and opt.get('chain_count') == 1:
                            self._warmup_auto_action = i
                            self._warmup_auto_steps = rot_steps
                            logging.info(f'[RE-SCAN] New auto-action: {i} toward {aim}°')
                            break
                    # Tag AVAILABLE DIRECTIONS
                    avail_lines = obs.get('avail_actions', '').split('\n')
                    for i, line in enumerate(avail_lines):
                        if i > 0 and i-1 < len(edge_opts):
                            if edge_opts[i-1].get('chain_type') == target_et:
                                avail_lines[i] += '  ★★★ RE-SCAN: GO THIS WAY! ★★★'
                                break
                    obs['avail_actions'] = '\n'.join(avail_lines)

            self._scan_obs = obs  # store for next step_env
            # #49 (bmv11 step28 实弹): 停票证据与投票证据同源 (#30a 补全)
            #   — 中途重扫在 _step_env 入口存 _last_obs 之后才替换决策 obs,
            #   停票仲裁 (_stop_evidence 读 _last_obs) 看的还是替换前的旧
            #   图 → 决策帧有框 (area 0.003 conf 0.86) 米制门却静默 None,
            #   停票落到桥接配对纯 VLM 判 close 放行 @2.42m fp。
            self._last_obs = obs
            logging.info('[RE-SCAN] Complete. Fresh perspective injected.')
        else:
            self._scan_obs = None

    HARD_FILTER_CONE_DEG = 45.0  # 推荐方位半锥角 (平移候选剔除边界)

    def _vlm_grounding_detect(self, rgb, target_name):
        """Direct-VLM / max-conf 共享检测原语: 全幅图问 VLM present + bbox

        Qwen-VL grounding 输出 → 合成与 yolo.check_target 同构的 info dict,
        下游伺服 (bbox 方位) / 停票门 (bbox 深探) / 黑名单零改动复用
        (与 gt_detect 同一注入模式, 换的是检测器)。
        present 但无 bbox → 视同未检出 (无框无法伺服/过米制门)。
        返回 (found, info)。
        """
        try:
            img = np.asarray(rgb)
            if img.ndim == 3:
                img = img[:, :, :3]
            tw = target_name.replace('_', ' ')
            prompt = (
                f"You are a robot's vision system. Target object: '{tw}'.\n"
                f"Look at the current camera view:\n"
                f"- Is the '{tw}' VISIBLE anywhere in the image? (present=1/0)\n"
                f"- If visible: your confidence (0.00-1.00) and its bounding box.\n"
                f"Respond ONLY with JSON: "
                f'{{"present": <0 or 1>, "confidence": <0.00-1.00>, '
                f'"bbox": [<x1>, <y1>, <x2>, <y2>]}}\n'
                f"bbox = left/top/right/bottom, normalized to 0-1000 of "
                f"image width/height; null if not visible."
            )
            resp = self.agent.actionVLM.call_chat(0, [img], prompt)
            miss = {'target_found': False, 'target_name': target_name,
                    'all_detections': []}
            m_p = re.search(r'"present"\s*:\s*(\d)', resp or '')
            if not m_p or m_p.group(1) != '1':
                return False, miss
            mb = re.search(
                r'"bbox"\s*:\s*\[\s*([\d.]+)[,\s]+([\d.]+)[,\s]+'
                r'([\d.]+)[,\s]+([\d.]+)', resp or '')
            if not mb:
                logging.info('[DIRECT-VLM] present but no bbox → treated '
                             'as miss (cannot servo/gate without a box)')
                return False, miss
            vals = [float(v) for v in mb.groups()]
            if max(vals) > 1.05:          # 0-1000 口径 → 归一化
                vals = [min(1000.0, max(0.0, v)) / 1000.0 for v in vals]
            x1, y1, x2, y2 = [min(1.0, max(0.0, v)) for v in vals]
            if x2 <= x1 or y2 <= y1:
                return False, miss
            mc = re.search(r'"confidence"\s*:\s*([\d.]+)', resp or '')
            conf = min(1.0, max(0.30, float(mc.group(1)))) if mc else 0.60
            area = (x2 - x1) * (y2 - y1)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            hl = 'left' if cx < 0.33 else ('right' if cx > 0.67 else 'center')
            vv = 'top' if cy < 0.33 else ('bottom' if cy > 0.67 else 'middle')
            det = {'class_id': -2, 'class_name': target_name,
                   'confidence': conf, 'bbox_norm': [x1, y1, x2, y2],
                   'area_ratio': area, 'center_x': cx, 'center_y': cy}
            return True, {'target_found': True, 'target_name': target_name,
                          'confidence': conf, 'position': f'{vv}_{hl}',
                          'area_ratio': area, 'all_detections': [det]}
        except Exception as e:
            logging.warning(f'[DIRECT-VLM] grounding failed: {e}')
            return False, {'target_found': False,
                           'target_name': target_name, 'all_detections': []}

    def _apply_detector_variants(self, obs):
        """表 IV 检测器变体分派 (决策前, 全部 obs 换血点之后):

        direct_vlm   : 每步 VLM grounding 替换 YOLO (含目击登记), yolo_text
                       同步重建 — 行语义即"每步送 VLM 判 present", VLM 调用
                       量的增加是该行的测量对象 (表 VIII 计量列)。
        maxconf_fusion: YOLO 检出目标 (conf≥0.25) 时问一次 VLM 置信,
                       fused = max(yolo, vlm) 替换置信路由 — VLFM 式融合;
                       YOLO 无检出不问 (召回侧不变, 差异集中在置信路由)。
        """
        _ff = getattr(self, '_ff', None) or {}
        sw = self.simWrapper
        tgt = getattr(sw, 'target_name', None)
        rgb = (obs or {}).get('color_sensor')
        if not tgt or rgb is None:
            return
        if _ff.get('direct_vlm'):
            found, info = self._vlm_grounding_detect(rgb, tgt)
            obs['yolo_detection'] = info
            if found:
                # 目击/状态机登记 (与 wrapper found 块同构, 换了检测源)
                desc = f"DirectVLM {info['position']} conf={info['confidence']:.2f}"
                sw._record_detection(sw.current_node, desc)
                sw.target_sighting_nodes.append(
                    (sw.memory['step_count'], sw.current_node, desc))
                if len(sw.target_sighting_nodes) > 5:
                    sw.target_sighting_nodes = sw.target_sighting_nodes[-5:]
            obs['yolo_text'] = sw.yolo.format_for_prompt(
                info, tgt, yolo_ever_found=found)
        elif _ff.get('maxconf_fusion'):
            cur = obs.get('yolo_detection') or {}
            if cur.get('target_found') \
                    and 0.25 <= cur.get('confidence', 0) < 1.0:
                vfound, vinfo = self._vlm_grounding_detect(rgb, tgt)
                vconf = vinfo.get('confidence', 0) if vfound else 0.0
                fused = max(cur.get('confidence', 0), vconf)
                if abs(fused - cur.get('confidence', 0)) > 1e-9:
                    cur = dict(cur)
                    cur['confidence'] = fused
                    dets = []
                    for d in cur.get('all_detections', []):
                        d = dict(d)
                        if d.get('class_name') == tgt:
                            d['confidence'] = fused
                        dets.append(d)
                    cur['all_detections'] = dets
                    obs['yolo_detection'] = cur
                    obs['yolo_text'] = sw.yolo.format_for_prompt(
                        cur, tgt, yolo_ever_found=True)
                    logging.info(f'[MAXCONF] yolo={cur.get("confidence", 0):.2f}'
                                 f'→fused={fused:.2f} (vlm={vconf:.2f})')

    def _hard_filter_depth_rec(self, obs):
        """表 III 对照: DEPTH 推荐硬剔除 — 推荐方位 ±45° 锥外的平移类候选
        (forward/backward/left/right/带走复合) 从动作集中剔除, 旋转类保留
        (锥内无可走项时仍可转向重探); 全部平移候选出锥 → 放行全部 (不困死)。

        剪枝必须同步 simWrapper.last_edge_options: wrapper 执行按该列表
        索引 edge_idx, 只改 obs 会造成"agent 按 pruned 列表选 idx, 世界按
        full 列表执行"的错位 (bm7/bm12 执行通道教训的同型坑)。
        """
        rec = obs.get('depth_rec_deg')
        opts = obs.get('edge_options') or []
        if rec is None or not opts:
            return
        geo = obs.get('option_geo') or []

        def _keep(i):
            o = opts[i]
            if not (o.get('chain_type') in ('forward', 'backward', 'left', 'right')
                    or (o.get('composite')
                        and any(e in ('forward', 'backward', 'left', 'right')
                                for e, _ in o['composite']))):
                return True   # 旋转类不属于"推荐锥"约束
            b = self._option_bearing_deg(o)
            return abs(((b - rec + 180.0) % 360.0) - 180.0) <= self.HARD_FILTER_CONE_DEG

        keep_idx = [i for i in range(len(opts)) if _keep(i)]
        n_walk_kept = sum(
            1 for i in keep_idx
            if opts[i].get('chain_type') in ('forward', 'backward', 'left', 'right')
            or (opts[i].get('composite')
                and any(e in ('forward', 'backward', 'left', 'right')
                        for e, _ in opts[i]['composite'])))
        if len(keep_idx) == len(opts) or n_walk_kept == 0:
            self._hf_pre_opts = None  # 本步未剪枝, 无需重定位
            return  # 无可剔除 / 锥内无可走项 → 放行 (不困死机器人)
        self._hf_pre_opts = list(opts)  # 剪枝前列表 (自动动作身份重定位用)
        dropped = [i for i in range(len(opts)) if i not in set(keep_idx)]
        obs['edge_options'] = [opts[i] for i in keep_idx]
        if len(geo) == len(opts):
            obs['option_geo'] = [geo[i] for i in keep_idx]
        # avail_actions 重编号: 首行表头, 行 i+1 ↔ edge_options[i]
        lines = obs.get('avail_actions', '').split('\n')
        if len(lines) >= len(opts) + 1:
            obs['avail_actions'] = '\n'.join(
                [lines[0]]
                + [re.sub(r'^\s*\[\d+\]', f'  [{j}]', lines[i + 1])
                   for j, i in enumerate(keep_idx)])
        if getattr(self.simWrapper, 'last_edge_options', None) is not None:
            self.simWrapper.last_edge_options = list(obs['edge_options'])
        logging.info(f'[FEATURE-FLAGS] hard_depth_filter: rec {rec:+.0f}° '
                     f'cone ±{self.HARD_FILTER_CONE_DEG:.0f}° → dropped '
                     f'options {dropped} ({len(keep_idx)}/{len(opts)} kept)')

    # ================= 阶段0: FSM 显式化 + 三修复 (设计稿 v1.2) =================

    def _fsm_state(self):
        """FSM 状态派生纯函数 — 由现有属性组合求值, 无新增持久状态 (设计稿 §2.1)

        派生序 (首命中): GOTO_WAYPOINT → CONFIRM_SIGHTING → APPROACH → EXPLORE。
        与 §2.3 仲裁表不矛盾: 表中 APPROACH 高于 GOTO_WAYPOINT 仅指非回访
        活动期间 — GOTO 活动时 APPROACH 机制整体让位 (现代码
        `if not self._forced_return_path and ...` 保留), 故派生先查 GOTO。
        ESCAPE 无 env 侧入口: 由 wrapper 硬规则触发时自行打
        `[FSM] state=ESCAPE (hard rule ...)` 日志 (环境反应, 仅诊断)。
        """
        if getattr(self, '_forced_return_path', None):
            return 'GOTO_WAYPOINT', 'forced-return path active'
        if getattr(self, '_arrival_check_step', -99) == self.step:
            return 'CONFIRM_SIGHTING', 'arrival pairing armed this step'
        if getattr(self, '_approach_bearing', None) is not None \
                and getattr(self, '_approach_miss', 99) <= 10:
            return 'APPROACH', f'bearing locked (miss={self._approach_miss})'
        return 'EXPLORE', 'no lock / no return path / no arrival check'

    def _step_concluded(self):
        """本步是否已在某个 super()._step_env 内部终局 (df 尾行非 running)

        覆盖路径 (CONFIRM/AUTO-STEER 等) 内部的停票成功会令 super 返回
        None, caller 无法与"无覆盖动作"区分 → 用来防止终局后再叠一个
        agent 步 (ep2 step25 型: 快通道停票在 CONFIRM override 内成功)。
        """
        try:
            return bool(len(self.df)) and str(
                self.df.iloc[-1].get('finish_status')) not in (
                'running', 'nan', 'None')
        except Exception:
            return False

    def _arm_vote_fastpath(self, obs):
        """修复2 Layer 2 (vote_fastpath): 桥接快通道武装 (设计稿 §4)

        新鲜目击活跃 (B1 桥条件, ≤3 步) + 当前帧有框 + 质量门
        (conf≥0.30 OR area≥0.01 — OR 不是 AND: ep2 实测 area=0.00/conf=0.36,
        AND 会误杀 ep2) → agent 连票门槛 2→1 (单张 done=1 票即发停票请求)。
        停票仍过 _calculate_metrics 全部米制门 — 连票数只是防幻觉票的
        第二道保险。武装结果挂 agent.fastpath_armed, 上下文留
        self._fastpath_ctx 供 [FASTPATH] 触发日志五字段。
        """
        ctx = None
        _ff = getattr(self, '_ff', None) or {}
        yolo_f = (obs or {}).get('yolo_detection') or {}
        if _ff.get('vote_fastpath', True) and yolo_f.get('target_found'):
            fs = None
            if hasattr(self.simWrapper, '_fresh_sighting_info'):
                fs = self.simWrapper._fresh_sighting_info()
            if fs is not None:
                conf_f = float(yolo_f.get('confidence', 0) or 0)
                area_f = 0.0
                tgt = getattr(self.simWrapper, 'target_name', '')
                for d_ in yolo_f.get('all_detections', []):
                    if d_.get('class_name') == tgt and d_.get('bbox_norm'):
                        x1_, y1_, x2_, y2_ = d_['bbox_norm']
                        area_f = max(area_f, max(0.0, (x2_ - x1_) * (y2_ - y1_)))
                if conf_f >= _ff.get('vote_fastpath_min_conf', 0.30) \
                        or area_f >= _ff.get('vote_fastpath_min_area', 0.01):
                    ctx = {'area': area_f, 'conf': conf_f,
                           'bridge_left': 3 - fs['steps_ago']}
        self._fastpath_ctx = ctx
        if getattr(self, 'agent', None) is not None:
            self.agent.fastpath_armed = ctx is not None

    # 修复1: L1 检测抑制窗 = L2/L3 升格最小间隔 (设计稿 §3)
    IDENTITY_GATE_COOLDOWN = 10  # steps

    def _wrong_strike_metrics(self, obs):
        """当前帧目标框的 (conf, bbox面积) — wrong-strike 证据强度度量"""
        yolo = (obs or {}).get('yolo_detection') or {}
        conf = float(yolo.get('confidence', 0) or 0)
        area = 0.0
        tgt = getattr(self.simWrapper, 'target_name', '')
        for d in yolo.get('all_detections', []):
            if d.get('class_name') == tgt and d.get('bbox_norm'):
                x1, y1, x2, y2 = d['bbox_norm']
                area = max(area, max(0.0, (x2 - x1) * (y2 - y1)))
        return conf, area

    def _penalize_proximity_wrong(self, obs):
        """修复1 (identity_gate_v2): PROXIMITY 判 'wrong' 的三级处罚 (设计稿 §3)

        旧行为 (一次 wrong → 永久拉黑) 把 ep8 的真可乐瓶 (0.06m) 拉黑,
        机器人被推到 10.30m。三级化:
          L1 首次 wrong       : 只解锁 + 节点检测抑制 10 步 (不拉黑), 记 strike
          L2 第二次 wrong     : 四条件 (面积≥2% + conf≥0.5 + 距上次≥冷却
                                10 步 + L1 先行) → 永久拉黑
          L3 第三次 wrong     : 每次间隔≥冷却 → 永久拉黑 + 全量 strike 日志
          间隔<冷却的重复 wrong: 不计新 strike (连续重复错不升格), 再冷却一次
        conf≥0.5 由 PROXIMITY 入口门隐含 (触发检查即要求 conf≥0.5),
        写入 L2 是防御性不变量。IDENTITY 'no' 路径本就 blacklist=False, 不动。
        """
        node = getattr(self.simWrapper, 'current_node', None)
        conf, area = self._wrong_strike_metrics(obs)
        strikes = getattr(self, '_wrong_strikes', None) or {}
        hist = strikes.get(node, []) if node else []

        def _l1(note):
            # L1: 解锁 (丢目击不拉黑) + 复用 _identity_reject 窗做节点检测抑制
            if node:
                self._identity_reject = {**getattr(self, '_identity_reject', {}),
                                         node: self.step}
            self._invalidate_lock(note, blacklist=False, drop_sighting=True,
                                  loss_strike=False)

        if hist and self.step - hist[-1][0] < self.IDENTITY_GATE_COOLDOWN:
            logging.info(f'[IDENTITY-GATE] wrong at {node} within cooldown '
                         f'({self.step - hist[-1][0]} < '
                         f'{self.IDENTITY_GATE_COOLDOWN} steps) — not a new '
                         f'strike, L1 again')
            _l1(f'VLM wrong (repeat within cooldown, area={area:.3f} '
                f'conf={conf:.2f}) — L1 cooldown again, NO blacklist')
            return
        if node:
            hist = hist + [(self.step, area, conf)]
            strikes[node] = hist[-6:]
            self._wrong_strikes = strikes
        if len(hist) >= 3:
            # L3: 重查耗尽 (防回环上界) — 全量 strike 日志供事后审计 (v1.2)
            dump = '; '.join(f'#{i + 1} step={s} area={a:.3f} conf={c:.2f}'
                             for i, (s, a, c) in enumerate(hist))
            intervals = ', '.join(str(hist[i + 1][0] - hist[i][0])
                                  for i in range(len(hist) - 1))
            logging.info(f'[IDENTITY-GATE] L3 EXHAUSTED at {node} — 3 wrong '
                         f'strikes [{dump}] intervals [{intervals}] → '
                         f'permanent blacklist (anti-loop bound)')
            self._invalidate_lock(
                f'VLM wrong #3 at {node} (L3 exhausted, see strike dump)')
            return
        if len(hist) == 2:
            # L2 四条件: 面积≥2% + conf≥0.5 (入口门隐含, 防御性) + 间隔≥冷却
            # (上面已保证) + L1 先行 (构造保证: hist[0] 走的就是 L1)
            if area >= 0.02 and conf >= 0.5:
                logging.info(
                    f'[IDENTITY-GATE] L2 ESCALATE at {node} — 2nd wrong with '
                    f'strong evidence (area={area:.3f}≥2%, conf={conf:.2f}≥0.5, '
                    f'interval={hist[1][0] - hist[0][0]}≥'
                    f'{self.IDENTITY_GATE_COOLDOWN}) → permanent blacklist')
                self._invalidate_lock(
                    f'VLM wrong #2 at {node} (big clear box denied twice '
                    f'after cooldown — L2 escalate)')
                return
            logging.info(f'[IDENTITY-GATE] wrong #2 at {node} WITHOUT strong '
                         f'evidence (area={area:.3f}, conf={conf:.2f}) — '
                         f'stay L1, no blacklist')
        _l1(f'VLM wrong at {node} (strike #{len(hist)}, area={area:.3f} '
            f'conf={conf:.2f}) — L1 cooldown, NO blacklist')

    def _proximity_suppressed(self):
        """修复1: L1 抑制窗内的节点不再触发 PROXIMITY 检查
        (VLM 刚否认过的同一节点 10 步内不重复问 — 防 VLM 调用浪费 + 防抖)"""
        node = getattr(self.simWrapper, 'current_node', None)
        return bool(
            (getattr(self, '_ff', None) or {}).get('identity_gate_v2', True)
            and node is not None
            and getattr(self, '_identity_reject', {}).get(node, -99)
            > self.step - self.IDENTITY_GATE_COOLDOWN)

    def _step_env(self, obs: dict):
        """Override: warmup auto-action + re-scan + 目击强制回访 + 全自动接近."""
        # === Check if re-scan obs is pending ===
        if hasattr(self, '_scan_obs') and self._scan_obs is not None:
            obs = self._scan_obs
            self._scan_obs = None

        # #30a (coca-ab 实弹, 2026-09-13): 暂存"决策实际用的观测"。
        #   旧版在 scan-obs 替换之前存原始 obs → 停票仲裁 (_stop_evidence
        #   读 _last_obs) 与 agent 决策 (stopping VLM / fastpath 武装读
        #   替换后 obs) 看的是两张不同的图: coca 终局步 fastpath 拿着
        #   扫描帧的框 (area 0.009/conf 0.37) 发单票停, 米制门却在原始
        #   前视图里找不到框 → "no yolo box → FAR" → UNREACHABLE 兜底
        #   放行 1.98m 停止 (本应软接近区拒停走近)。停票证据必须与投票
        #   证据同源 — 移到替换之后。
        self._last_obs = obs
        self._inject_zone_caution(obs)   # #51b: 否决区警示 (见方法 docstring)

        # === 阶段0 FSM: 每步开头求值并打日志 (派生纯函数, 设计稿 §2) ===
        _fsm, _fsm_reason = self._fsm_state()
        logging.info(f'[FSM] step={self.step} state={_fsm} ({_fsm_reason})')

        # === 阶段1: 全路径节点轨迹记录 (含 override 步 — _recent_nodes
        #     只记正常 VLM 步, 转圈恰发生在 AUTO-STEER/PEEK 等 override
        #     步上) → _area_exhausted 的数据源 ===
        self._all_nodes = (getattr(self, '_all_nodes', [])
                           + [getattr(self.simWrapper, 'current_node',
                                      None)])[-12:]

        # === 档位剥离 (消融): cascade:F → agent 不见 YOLO 文本 (纯 VLM
        #     决策, base 复现 VLMnav); grid_hub:F → 不见结构化深度推荐
        #     (G 档差异)。env 自身机制用 obs['yolo_detection'] (dict),
        #     停票门证据不受剥离影响 ===
        _ff = getattr(self, '_ff', None)
        if _ff is not None:
            if not _ff.get('cascade', True):
                obs.pop('yolo_text', None)
            if not _ff.get('grid_hub', True):
                obs.pop('depth_rec_deg', None)

        # === 周期重扫描 (决策前!): 扫描结果当步立即生效, 不再浪费一步
        #     (Bug修复: 之前扫描在 VLM 决策之后触发, 结果要下一步才用上) ===
        if not hasattr(self, '_periodic_scanned_at'):
            self._periodic_scanned_at = set()
        # === #26b (换机位补扫): 判尽门挡下置位 → 本步先平移一格到新
        #     站位 (占动作), 下一步原地 360° 六向重扫登记新机位 (走 P4
        #     执行通道, 预算独立)。放在 P4 消费之前: 平移必须发生在
        #     补扫之前, 否则登记的还是旧节点 (set 去重 = 白扫)。
        #     #34: 强制旅程进行中不消费 (平移会把机器人带离旅程路径,
        #     pending 保留到旅程结束) ===
        if getattr(self, '_pending_viewpoint_shift', False) \
                and not getattr(self, '_forced_return_path', None):
            shift_act = self._consume_viewpoint_shift(obs)
            if shift_act is not None:
                return shift_act

        # === P4 (释锁即六向重扫): _invalidate_lock 置位后, 这一步决策前
        #     先 360° 重扫一次重新定向 (预算 ≤2/回合, 保已访记忆 — 不然
        #     "一切重新变新"会让 VLM 放弃已搜索区域)。与周期重扫互斥:
        #     本步已扫则周期块跳过 (防同步双跑双倍 VLM 成本)。
        #     #26b: _pending_viewpoint_rescan (换机位补扫来源) 共用此
        #     执行体但只查 viewpoint_rescan 预算 — p16 cola 实弹 P4
        #     预算开局 2 次耗尽后, 补扫来源被连带堵死。
        #     #34: 强制旅程 (前沿跳/回访) 进行中不扫 — 旅程本身就是
        #     "困惑的出路", 中途扫描 = 打断换向 + obs 替换致计划边失配
        #     (bmv4 step18: 9 边旅程第 2 边后被困惑扫掐死) ===
        _pend_vp = getattr(self, '_pending_viewpoint_rescan', False)
        if (getattr(self, '_pending_release_rescan', False) or _pend_vp) \
                and not getattr(self, '_forced_return_path', None):
            self._pending_release_rescan = False
            self._pending_viewpoint_rescan = False
            if _pend_vp:
                ok = (_ff or {}).get('viewpoint_rescan', True)
                src = 'viewpoint coverage re-scan (#26 灯下黑补扫)'
            else:
                ok = (_ff or {}).get('release_rescan', True) \
                    and getattr(self, '_release_rescans', 0) < 2
                if ok:
                    self._release_rescans = getattr(self, '_release_rescans', 0) + 1
                src = 'lock release re-orientation (pre-action, P4)'
            if ok:
                logging.info(f'[RE-SCAN] Triggered: {src} '
                             f'at step {self.step}')
                self._periodic_scanned_at.add(self.step)
                self._re_scan(self.step, reset_memory=False)
                if getattr(self, '_scan_obs', None) is not None:
                    obs = self._scan_obs
                    self._scan_obs = None
        # === 扫描机制 v2 (用户指令 2026-09-13): 按需扫描 — "实在困惑就
        #     原地 scan, 不用等整 10 步"。困惑信号全部机器人自身状态
        #     (零样本合规): ① VLM 连续 3 次同选择 ② 零检出 ≥6 步 ③ 当前
        #     节点无新方向 ④ 本 zone 已 360° 扫过仍无发现; ≥2 信号同时
        #     成立即扫 (冷却 4 步 + 预算 ≤4/回合, 六图扫描 ~7 次 VLM 调用)。
        #     周期扫描降级为保底: 距上次任意扫描 ≥10 步才触发 (任意扫描
        #     重置计时 — 不再死等整 10 倍数步, 用户实弹 mahatma step6 案:
        #     warmup 后连选 4 次 forward + 零检出 7 步, 旧制要等到 step10) ===
        if not hasattr(self, '_confusion_rescans'):
            self._confusion_rescans = 0
        sig = []
        # #40c: APPROACH 伺服进行中 (有锁且 miss≤10) 困惑扫让路 — "零检出/
        #   无新方向"对接近目标本是常态, bmv6 step44 miss=3、伺服 6° 对准
        #   中被 120° 重扫打断。与 #34c 旅程守卫同族; miss>10 说明真跟丢,
        #   重扫恢复合法。
        _approach_alive = not (
            getattr(self, '_approach_bearing', None) is not None
            and getattr(self, '_approach_miss', 0) <= 10)
        if self.step > 0 \
                and self.step - getattr(self, '_last_scan_step', 0) >= 4 \
                and self._confusion_rescans < 4 \
                and not getattr(self, '_forced_return_path', None) \
                and _approach_alive \
                and self.step not in self._periodic_scanned_at:
            ah = (getattr(self.agent, 'action_history', None) or [])[-3:]
            if len(ah) == 3 and ah[0] == ah[1] == ah[2]:
                sig.append(f'vlm repeat choice x3 (action {ah[0]})')
            if getattr(self, '_no_detect_streak', 0) >= 6:
                sig.append(f'no-detect {self._no_detect_streak} steps')
            try:
                if not self.simWrapper._get_unvisited_directions(
                        self.simWrapper.current_node):
                    sig.append('no fresh direction')
            except Exception:
                pass
            try:
                if self.simWrapper._area_key(self.simWrapper.current_node) \
                        in (getattr(self, '_scan_positions_by_cell', None) or {}):
                    sig.append('zone already 360-scanned')
            except Exception:
                pass
            if len(sig) >= 2:
                sig_txt = ' | '.join(sig)
                self._confusion_rescans += 1
                logging.info(f'[RE-SCAN] Triggered: CONFUSION ({sig_txt}) '
                             f'at step {self.step}')
                self._periodic_scanned_at.add(self.step)
                self._re_scan(self.step, reset_memory=False)
                if getattr(self, '_scan_obs', None) is not None:
                    obs = self._scan_obs
                    self._scan_obs = None
        if self.step > 0 and self.step - getattr(self, '_last_scan_step', 0) >= 10 \
                and not getattr(self, '_forced_return_path', None) \
                and _approach_alive \
                and self.step not in self._periodic_scanned_at:
            sightings = getattr(self.simWrapper, 'target_sighting_nodes', [])
            recent_sighting = bool(sightings) and (self.step - sightings[-1][0]) <= 8
            if not recent_sighting:
                logging.info(f'[RE-SCAN] Triggered: periodic re-orientation '
                             f'at step {self.step} (pre-action, ≥10 since '
                             f'last scan)')
                self._periodic_scanned_at.add(self.step)
                self._re_scan(self.step, reset_memory=False)
                if hasattr(self, '_scan_obs') and self._scan_obs is not None:
                    obs = self._scan_obs
                    self._scan_obs = None

        # === 表 IV 检测器变体 (direct_vlm / maxconf_fusion): 同样置于全部
        #     obs 换血点之后 — 后续 PROXIMITY/接近/停票门读到的都是变体后
        #     的 yolo_detection ===
        if (_ff or {}).get('direct_vlm') or (_ff or {}).get('maxconf_fusion'):
            self._apply_detector_variants(obs)

        # === 表 III 对照行: DEPTH 推荐硬剔除 — 置于全部 obs 换血点之后
        #     (首换 _scan_obs / 周期重扫描), 保证 agent 所见即过滤后动作集 ===
        if (_ff or {}).get('hard_depth_filter'):
            self._hard_filter_depth_rec(obs)
            # warmup/re-scan 自动动作按身份重定位: 其目标必为纯旋转选项
            # (过滤不删旋转), 但删除前方平移项会使索引前移
            if getattr(self, '_warmup_auto_action', None) is not None \
                    and getattr(self, '_hf_pre_opts', None):
                tgt = self._hf_pre_opts[self._warmup_auto_action] \
                    if self._warmup_auto_action < len(self._hf_pre_opts) else None
                new_idx = next((i for i, o in enumerate(obs['edge_options'])
                                if o is tgt), None)
                if new_idx is None:
                    logging.info('[FEATURE-FLAGS] hard_depth_filter: warmup '
                                 'auto-action dropped (option pruned)')
                self._warmup_auto_action = new_idx

        # === 修复2 Layer 2 (vote_fastpath): 桥接快通道武装 — 必须在首个
        #     super()._step_env (投票在其中发生) 之前; 检测器变体/硬过滤
        #     已换血完毕, 读的是最终 yolo_detection ===
        self._arm_vote_fastpath(obs)

        # === Auto-action from warmup/re-scan ===
        if hasattr(self, '_warmup_auto_action') and self._warmup_auto_action is not None:
            auto_idx = self._warmup_auto_action
            self._warmup_auto_action = None  # only fire once
            logging.info(f'[AUTO] Step {self.step}: executing warmup/re-scan auto-action {auto_idx}')

            obs['goal'] = self.current_episode['object'] if isinstance(self.current_episode, dict) else self.current_episode

            # Use parent step_env logic but override the VLM decision
            # Simpler: just call parent but replace the action
            agent_action = super()._step_env(obs)

            # Override with auto-action
            if agent_action is not None:
                self.simWrapper._action_intent = 'warmup_auto'   # B⑦
                # stop/null 单例先换新对象再挂 edge_idx (在单例上挂属性
                # 会永久污染 — 后续所有 stop 都会被 wrapper 当作这个边执行)
                agent_action = self._attach_exec_idx(agent_action, auto_idx)
                if hasattr(self, '_warmup_auto_steps'):
                    agent_action.rotate_steps = self._warmup_auto_steps
                    logging.info(f'[AUTO] Overriding with action {auto_idx}, rotate_steps={self._warmup_auto_steps}')
                else:
                    logging.info(f'[AUTO] Overriding VLM action with auto-action {auto_idx}')
                # 修正显示: VLM 原始选择会留在 df/details 里造成误导
                # (实测用户看到 "action 5 SIDESTEP" 以为走错方向, 实际执行的是 [3] RIGHT 60°)
                raw_idx = None
                if len(self.df) and 'action_number' in self.df.columns:
                    raw_idx = self.df.iloc[-1]['action_number']
                    self.df.iloc[-1, self.df.columns.get_loc('action_number')] = auto_idx
                ep_dir = (f'logs/{self.outer_run_name}/{self.inner_run_name}/'
                          f'{self.curr_run_name}/step{self.step}')
                dt_path = f'{ep_dir}/details.txt'
                if os.path.exists(dt_path):
                    with open(dt_path, 'a') as f:
                        f.write(f'\nAUTO (warmup/re-scan): executed action [{auto_idx}] '
                                f'(VLM raw choice overridden)\n')
                # 选中图重画为实际执行动作 + 覆盖角标
                self._redraw_chosen_as_executed(
                    obs, auto_idx, 'warmup/re-scan', raw_idx)

            return agent_action

        # === 状态初始化 (每步首用) ===
        if not hasattr(self, '_forced_return_path'):
            self._forced_return_path = []
        if not hasattr(self, '_last_returned_sighting_step'):
            self._last_returned_sighting_step = -1

        # === A: VLM 接近度判断 — 成功判定不依赖距离信息, 由 VLM 视觉判断
        #     "目标是否足够接近" (YOLO 中高置信度检测到目标时触发检查,
        #     给 VLM 看画了目标框的照片) ===
        if not hasattr(self, '_last_proximity_step'):
            self._last_proximity_step = -99
        if not hasattr(self, '_force_stop'):
            self._force_stop = False
        yolo_now = obs.get('yolo_detection') or {}
        # 修2B (ep4): 零检出连击计数 — 未见目标 +1 / 见到清零;
        # ≥12 步触发 crop 兜底 (见 ③(b) 后的触发块)
        self._no_detect_streak = 0 if yolo_now.get('target_found') \
            else getattr(self, '_no_detect_streak', 0) + 1
        # 问题1-B (parallax_gate): 目标框面积历史 — 连续两帧面积比 + 已知
        # 位移 → 独立距离估计 (视差测距), 供 _stop_evidence 交叉
        if yolo_now.get('target_found'):
            _tgt_a, _tgt_nm = 0.0, getattr(self.simWrapper, 'target_name', '')
            for _d_ in yolo_now.get('all_detections', []):
                if _d_.get('class_name') == _tgt_nm and _d_.get('bbox_norm'):
                    _x1_, _y1_, _x2_, _y2_ = _d_['bbox_norm']
                    _tgt_a = max(_tgt_a, max(0.0, (_x2_ - _x1_) * (_y2_ - _y1_)))
            if _tgt_a > 1e-4:
                _h = getattr(self, '_bbox_area_hist', None) or []
                _h.append((self.step, _tgt_a,
                           getattr(self.simWrapper, 'current_node', None)))
                self._bbox_area_hist = _h[-5:]
        # 修复1 (identity_gate_v2): L1 抑制窗内的节点不再触发 PROXIMITY 检查
        if self._proximity_suppressed():
            logging.debug(f'[IDENTITY-GATE] PROXIMITY check suppressed at '
                          f'{getattr(self.simWrapper, "current_node", None)} '
                          f'(L1 window active)')
        elif (_ff or {}).get('metric_gate', True) \
                and yolo_now.get('target_found') and yolo_now.get('confidence', 0) >= 0.5 \
                and self.step - self._last_proximity_step >= 2:
            self._last_proximity_step = self.step
            verdict = self._vlm_proximity_check(obs)
            # 记录带框判定结果: 有 YOLO 框指引的判定比无框 stop_vote 可靠,
            # 供停止仲裁参考 (实测: 带框判 'far'(真目标远望) 后 1 步 stop_vote
            # 误判 'wrong' 拉黑了真目标区域, 抑制后续所有检测 → 卡死)
            if verdict in ('close', 'far', 'wrong'):
                self._boxed_verdict = (self.step, verdict)
            if verdict == 'close':
                self._force_stop = True
                if hasattr(self.simWrapper, '_record_confirmed'):
                    self.simWrapper._record_confirmed(self.simWrapper.current_node)
                agent_action = super()._step_env(obs)
                self._force_stop = False
                # bm11 接力: 判 close 但评估侧 ≥2.5m 拒了 (fp 语义) → 目标
                # 看着近实际远 → 检测框方位移交接近锁, 下步 AUTO-STEER 走过去
                # (实测缺这环: VLM 4 步连票停在 4.39m 处活活 fp)
                if getattr(self, '_force_stop_rejected', False):
                    self._force_stop_rejected = False
                    rel = self._detection_bearing_deg(yolo_now)
                    if rel is not None:
                        self._approach_bearing = self._yaw_world() + np.radians(rel)
                        self._approach_miss = 0
                        logging.info(f'[PROXIMITY] fp-rejected → approach lock '
                                     f'rel={rel:+.0f}deg (walk to it, not stop)')
                # 被拒后动作可能是 stop 单例 → 转 180° 回退为当步移动
                if agent_action is PolarAction.stop:
                    agent_action = self._vetoed_stop_action(obs)
                return agent_action
            elif verdict == 'wrong':
                # 修复1 (identity_gate_v2): 三级化处罚 (旧行为一次 wrong 即
                # 永久拉黑, ep8 真目标 0.06m 被拉黑推到 10.30m);
                # 开关关 → else 分支保留旧行为 (消融对照)
                if (_ff or {}).get('identity_gate_v2', True):
                    self._penalize_proximity_wrong(obs)
                else:
                    self._invalidate_lock('VLM rejected detection (wrong object)')

        # === 方案3 全自动接近模式: 已锁定目标方位时, 不咨询 VLM,
        #     每步自动朝目标走 (活体检测伺服 / 按锁定方位选边绕障)。
        #     当前帧 YOLO 高置信度看到目标也会触发/延续接近模式 ===
        if not hasattr(self, '_approach_miss'):
            self._approach_miss = 0
        if not hasattr(self, '_approach_bearing'):
            self._approach_bearing = None
        # re-arm 门槛 0.5 (未锁定时要起新锁必须中高置信度);
        # 已锁定 (bearing 有效) 延续伺服仍用 0.3 — 修实测 conf 0.33~0.63
        # 厨房闪烁反复 re-arm, 劫持 40 步的死区 (arbitration 要 0.5, re-arm 却只要 0.3)
        _armed = self._approach_bearing is not None and self._approach_miss <= 10
        yolo_live = bool(yolo_now.get('target_found')) \
            and yolo_now.get('confidence', 0) >= (0.3 if _armed else 0.5)
        # 绕障路径执行期间, 接近模式让位 (先走完绕障, 否则绕障会被活体
        # 伺服打断而永远走不完 — 实测绕圈原因之一)
        if not self._forced_return_path and (
                yolo_live or (self._approach_bearing is not None and self._approach_miss <= 10)):
            app_action = self._approach_step(obs)
            if app_action is not None:
                return app_action
            # 修复2: 覆盖路径 (CONFIRM/AUTO-STEER) 内部已终局 (如快通道停票
            # 成功 → super 返回 None) → 不再叠第二个 agent 步
            if self._step_concluded():
                logging.info(f'[FSM] step={self.step} episode concluded '
                             f'inside override → end step')
                return None

        # === 🕵️ 未锁定的中置信目击也要过身份检查 (口袋陷阱主因拆除) ===
        # 实测: 口袋里 YOLO 反复 0.34~0.47 误报 → 目击登记 → FORCED-RETURN
        # 把机器人拖回同一节点 → 离开 → 又拖回, 28 步出不去。
        # VLM 判 "no" → 该节点记入排除窗 + 撤销它的目击登记 (不再强制回访)。
        _armed_now = self._approach_bearing is not None and self._approach_miss <= 10
        if (not self._forced_return_path and not _armed_now
                and bool(yolo_now.get('target_found'))):
            conf_u = yolo_now.get('confidence', 0)
            area_u = float(yolo_now.get('area_ratio', 0) or 0)
            if 0.30 <= conf_u < 0.60 and area_u < 0.10:
                cur = getattr(self.simWrapper, 'current_node', None)
                sw = self.simWrapper
                if getattr(self, '_identity_reject', {}).get(cur, -99) > self.step - 10:
                    # 排除窗内: 同一坑的重复误报, 直接撤销目击义务
                    sw.target_sighting_nodes = [
                        s for s in (getattr(sw, 'target_sighting_nodes', []) or [])
                        if s[1] != cur]
                elif (_ff or {}).get('cascade', True) \
                        and self.step - getattr(self, '_last_identity_step', -99) >= 3:
                    self._last_identity_step = self.step
                    _id_verdict = self._vlm_identity_check(obs, yolo_now)
                    if _id_verdict == 'no':
                        # === #51a (bmd01 实弹 ×4): 远距真瓶被 unarmed
                        # 路径直接放走 — conf 0.37/0.48 + area 0.003-0.006
                        # 的框 VLM 根本看不清, no 不是证据 → 与 #42 同源
                        # (armed 路径有二确认, 此路径漏接)。放走后目击区
                        # 因 #31 守卫永不 CLEAR, 记忆拽回再看再否 = 用户
                        # 报障 "不停去探索同一区域" 的 d01 形态。
                        # conf<0.50 或框 <1% 画面 → 前进二次确认 (走近框
                        # 变大再判), 预算与 #42 共用 ≤3/节点, 超限回落原
                        # 拒绝路径 (保真误报止损) ===
                        if (conf_u < 0.50 or area_u < 0.01) and _ff.get(
                                'identity_lowconf_confirm', True):
                            used = getattr(self, '_lowconf_confirm_used', {}).get(cur, 0)
                            if used < 3:
                                self._lowconf_confirm_used = {
                                    **getattr(self, '_lowconf_confirm_used', {}),
                                    cur: used + 1}
                                ci = self._pick_confirm_option(obs, yolo_now)
                                if ci is not None:
                                    logging.info(
                                        f'[IDENTITY] no @ conf {conf_u:.2f} '
                                        f'area {area_u:.3f} (unarmed, 远距小框 '
                                        f'看不清) → forward second confirmation '
                                        f'({used + 1}/3) (#51a)')
                                    return self._override_and_run(
                                        obs, ci, 'CONFIRM',
                                        f'identity no at LOW conf/tiny box '
                                        f'({conf_u:.2f}/{area_u:.3f}) — go '
                                        f'closer and re-check (#51a)')
                                logging.info(
                                    f'[IDENTITY] no @ conf {conf_u:.2f} but no '
                                    f'confirm option — fall through to reject (#51a)')
                            else:
                                logging.info(
                                    f'[IDENTITY] no @ conf {conf_u:.2f}, confirm '
                                    f'budget {used}/3 exhausted at {cur} → '
                                    f'reject as before (#51a)')
                        self._identity_reject = {
                            **getattr(self, '_identity_reject', {}), cur: self.step}
                        self._zone_neg_strike()   # #51b: 区域级否决记账
                        sw.target_sighting_nodes = [
                            s for s in (getattr(sw, 'target_sighting_nodes', []) or [])
                            if s[1] != cur]
                        logging.info(
                            f'[IDENTITY] unarmed sighting at {cur} rejected — '
                            f'drop forced-return obligation, keep exploring')
                        # 同上: 低置信误检先换机位重看 (灯下黑保护)
                        rep = self._reposition_after_reject(obs, conf_u, 'ep1')
                        if rep is not None:
                            return rep
                    elif _id_verdict == 'yes':
                        # bm15 实弹: conf 0.39 真目标 IDENTITY yes 后无任何
                        # 下文 (不建锁/不登记) → yolo_live 门槛 0.5 又不够 →
                        # 落到 OCCLUSION-PEEK 转身, 目标出画, 再没回来。
                        # VLM 都确认是目标了 → 检测方位建接近锁, 下步伺服
                        rel_yes = self._detection_bearing_deg(yolo_now)
                        if rel_yes is not None:
                            self._approach_bearing = self._yaw_world() \
                                + np.radians(rel_yes)
                            self._approach_miss = 0
                            self._approach_last_area = float(
                                yolo_now.get('area_ratio', 0) or 0)
                            logging.info(
                                f'[IDENTITY] VLM confirmed target (conf '
                                f'{conf_u:.2f}) → approach lock rel='
                                f'{rel_yes:+.0f}deg, next step AUTO-STEER')

        # === 目击强制回访 (方案A): YOLO/VLM 看到过目标后若离开目击节点,
        #     沿图路径自动走回 (不咨询 VLM — 实测 VLM 会无视"回去查"提示) ===
        if not self._forced_return_path:
            self._start_forced_return()

        ret_action = self._execute_forced_return(obs)
        if ret_action is not None:
            return ret_action

        # === ③(c) 到场配对确认: 强制回访抵达目击位置 (或本就在目击位) →
        #     目击帧 + 当前帧并排送 VLM 一次。G2 实测: 目击时目标又大又近,
        #     回到原位后目标却出画 (相机俯角/朝向), 当前帧怎么确认都失败 →
        #     但 "目击帧近景 + 已回原位" 就是停止证据。一次性消费, ≤6 次。 ===
        if (getattr(self, '_arrival_check_step', -99) == self.step
                and not self._forced_return_path):
            self._arrival_check_step = -99  # 一次性消费
            sights_c = getattr(self.simWrapper, 'target_sighting_nodes', []) or []
            if sights_c and not yolo_live and self._approach_bearing is None:
                s_frame = getattr(self, '_sighting_frames', {}).get(
                    sights_c[-1][1])
                if s_frame is not None:
                    verdict = self._arrival_confirm(obs, s_frame)
                    if verdict == 'close':
                        logging.info('[ARRIVAL-CONFIRM] sighting-frame CLOSE + '
                                     'back at sighting spot → force stop')
                        self._force_stop = True
                        if hasattr(self.simWrapper, '_record_confirmed'):
                            self.simWrapper._record_confirmed(
                                self.simWrapper.current_node)
                        agent_action = super()._step_env(obs)
                        self._force_stop = False
                        if agent_action is PolarAction.stop:
                            agent_action = self._vetoed_stop_action(obs)
                        return agent_action

        # === ③(b) 几何触发 crop 确认: 初进新 2m 区域格 + 前方 <2m 有台面。
        #     近距小目标 YOLO 常漏检 (E), 整幅缩略图 VLM 常看不清 (G1/G4
        #     实测踩到 0.06m 仍不停) — 到了新功能区、面前就有个台子,
        #     放大裁剪确认一次, 'close' 直接以 _force_stop 收官。
        #     限流: 每区域格一次, 全回合 ≤10 次, 间隔 ≥2 步。 ===
        if (not yolo_live
                and self._approach_bearing is None
                and self.step - getattr(self, '_last_crop_step', -99) >= 2
                and getattr(self, '_crop_calls', 0) < 10
                and not self._forced_return_path):
            sw_c = self.simWrapper
            try:
                cur_cell = sw_c._area_key(getattr(sw_c, 'current_node', None))
            except Exception:
                cur_cell = None
            if cur_cell is not None \
                    and cur_cell not in getattr(self, '_crop_cells', set()):
                fwd_near = False
                for o in (obs.get('edge_options') or []):
                    is_fwd = o.get('chain_type') == 'forward' or (
                        o.get('composite')
                        and any(e == 'forward' for e, _ in o['composite']))
                    if is_fwd and float(o.get('safe_dist_m', 9) or 9) < 2.0:
                        fwd_near = True
                        break
                if fwd_near:
                    if not hasattr(self, '_crop_cells'):
                        self._crop_cells = set()
                    self._crop_cells.add(cur_cell)
                    verdict = self._crop_confirm(obs, f'new-area {cur_cell}')
                    if verdict == 'close' \
                            and self._distance_xcheck(obs, 'crop-close')[0] == 'veto':
                        # P1: 几何核否决 — crop 看着近但独立源说远
                        # (ep6 step35 型), 不停, 本步继续正常决策
                        verdict = None
                    if verdict == 'close':
                        logging.info('[CROP-GATE] geometric trigger CLOSE → '
                                     'force stop (G1/G4 step-on-target fix)')
                        self._force_stop = True
                        if hasattr(self.simWrapper, '_record_confirmed'):
                            self.simWrapper._record_confirmed(
                                self.simWrapper.current_node)
                        agent_action = super()._step_env(obs)
                        self._force_stop = False
                        if agent_action is PolarAction.stop:
                            agent_action = self._vetoed_stop_action(obs)
                        return agent_action

        # === 修2B (ep4 实弹): 零检出感知兜底 — softsoap 全程零 YOLO 检出
        #     (小目标类别盲区) → 无线索纯扫视 → 硬规则误判振荡 → 区域转圈。
        #     连续 ≥12 步未见目标时每 6 步 crop 放大看一次前方近处:
        #     'close' 收官; 'far' (present 但远) 在日志留线索。与 ③(b)
        #     几何触发共用 _crop_calls ≤10 限流与限流间隔。 ===
        if (not yolo_live and self._approach_bearing is None
                and (_ff or {}).get('zero_detect_crop', True)
                and getattr(self, '_no_detect_streak', 0) >= 12
                and self.step - getattr(self, '_last_zerodetect_crop', -99) >= 6
                and getattr(self, '_crop_calls', 0) < 10
                and not self._forced_return_path):
            self._last_zerodetect_crop = self.step
            verdict = self._crop_confirm(
                obs, f'zero-detect {self._no_detect_streak} steps')
            if verdict == 'close' \
                    and self._distance_xcheck(obs, 'crop-close')[0] == 'veto':
                # P1: 几何核否决 (ep6 step35: zero-detect crop close @3.95m
                # — 正是此路径; 前向带 >1.5m → 不停, 继续走近再核)
                verdict = None
            if verdict == 'close':
                logging.info('[CROP-GATE] zero-detect CLOSE → force stop '
                             '(ep4 no-clue wandering fix)')
                self._force_stop = True
                if hasattr(self.simWrapper, '_record_confirmed'):
                    self.simWrapper._record_confirmed(
                        self.simWrapper.current_node)
                agent_action = super()._step_env(obs)
                self._force_stop = False
                if agent_action is PolarAction.stop:
                    agent_action = self._vetoed_stop_action(obs)
                return agent_action

        # === ④ 覆盖完成停: 地图无前沿灰格 (无可走未到访) 且覆盖足够 →
        #     体面结束搜索 (not-found-exhausted)。5 步连续确认防单帧误判;
        #     比 max_steps 游荡到死省步数省 VLM 调用 (G4 型游荡)。 ===
        if self._coverage_exhausted():
            self._exhausted_streak = getattr(self, '_exhausted_streak', 0) + 1
            if self._exhausted_streak >= 5:
                logging.info(f'[EXHAUSTED-STOP] coverage complete for '
                             f'{self._exhausted_streak} consecutive steps '
                             f'→ dignified not-found stop')
                # 置位后走父类正常一步 (保留 df/日志行), _calculate_metrics
                # 里的 EXHAUSTED 块负责强制 done=True 结束回合
                self._exhausted_stop = True
                return super()._step_env(obs)
        else:
            self._exhausted_streak = 0

        # === 扫描机制 v2 (用户指令 2026-09-13): zone 判 CLEAR (无可疑、
        #     无价值) → 第一时间机制强制离开 — 不再只是 prompt 注记 (实测
        #     VLM 无视 "move on" 提示继续原地磨)。复用前沿跳跃规划走向
        #     最近前沿; 与目击追击/接近锁互斥 (有活证据优先追) ===
        if (not yolo_live and self._approach_bearing is None
                and not self._forced_return_path
                and (_ff or {}).get('zone_clear_exit', True)
                and self._in_cleared_zone(obs)):
            if self._plan_frontier_hop():
                logging.info(f'[ZONE-CLEAR] step {self.step}: 当前 zone 已判'
                             f'无价值 → 立即离区 (最高优先)')
                ret_action = self._execute_forced_return(obs)
                if ret_action is not None:
                    return ret_action

        # === 阶段1 前沿跳跃 (转圈根治): 无目标线索 + 最近 10 步困在
        #     <2.5m 小区域 (区域看尽) → 探索地图前沿格 (可走未到访)
        #     最近一个 → GOTO_WAYPOINT 强制离开。6 步冷却防连发;
        #     规划失败 (无前沿/无已知空间路径) 静默回退现有行为 ===
        if (not yolo_live
                and self._approach_bearing is None
                and not self._forced_return_path
                and (_ff or {}).get('frontier_hop', True)
                and self.step - getattr(self, '_last_frontier_hop_step',
                                        -99) >= 6
                and self._area_exhausted()):
            if self._plan_frontier_hop():
                ret_action = self._execute_forced_return(obs)
                if ret_action is not None:
                    return ret_action

        # === 遮挡偷看 (兜底探索): 无活体检测 + 无锁定 + 无新区域方向 +
        #     6 步内没偷看过 → 朝深度图上的"障碍背后阴影"换角度查看 ===
        sw = self.simWrapper
        if (not yolo_live
                and self._approach_bearing is None
                and self.step - getattr(self, '_last_peek_step', -99) >= 6
                and hasattr(sw, '_get_unvisited_directions')):
            try:
                if not sw._get_unvisited_directions(
                        getattr(sw, 'current_node', None)):
                    peek = self._occlusion_peek(obs)
                    if peek is not None:
                        return peek
            except Exception:
                pass

        # Normal step
        agent_action = super()._step_env(obs)

        # B⑦: agent 强制探索 (rec-follow/安全池随机) 的意图传播 —
        # wrapper FINAL-EXEC trace 记 forced_rec/forced_random (与 env
        # 机制标签同通道; agent 每步开头置 None, 无残留)
        _fk = getattr(self.agent, 'last_forced_kind', None)
        if _fk:
            self.simWrapper._action_intent = f'forced_{_fk}'

        # === VLM 假停止拦截: 有近期目击时, stop 不能提前结束任务 →
        #     转为强制回访目击节点 (继续找) ===
        if agent_action is PolarAction.stop and self._start_forced_return():
            ret_action = self._execute_forced_return(obs)
            if ret_action is not None:
                return ret_action

        # === 被否决的停止票: D2/仲裁/crop 门在 _calculate_metrics 里否决了
        #     stop (done=False) 时, super 仍返回 stop 单例 → wrapper 无
        #     edge_idx 会回退 [0] = 180° turn_around (bm12 step22→23 实测)。
        #     转为当步就走 (接近锁对齐 / VLM 原始移动选择)。 ===
        if agent_action is PolarAction.stop:
            agent_action = self._vetoed_stop_action(obs)

        # === Post-step hooks: record VLM target sightings for next step ===
        if agent_action is not None:
            # Hook 1: If agent's target_memory was just updated, record to simWrapper
            # (agent 侧已做否定句过滤, 这里按目击记录; 注意用空格形式的名字
            #  以匹配 record_vlm_sighting 的模式)
            if hasattr(self.agent, 'target_memory') and self.agent.target_memory:
                mem_step, mem_desc, mem_action = self.agent.target_memory
                if mem_step >= self.step - 1:  # freshly updated this step
                    if hasattr(self.simWrapper, 'record_vlm_sighting'):
                        tname_sp = self.simWrapper.target_name.replace('_', ' ')
                        self.simWrapper.record_vlm_sighting(
                            f"i see {tname_sp} {mem_desc}"
                        )

            # Hook 2: If stopping voted done=1, also record
            if hasattr(self.agent, 'stop_history') and self.agent.stop_history:
                if self.agent.stop_history[-1]:
                    if hasattr(self.simWrapper, 'record_vlm_sighting'):
                        tname_sp = self.simWrapper.target_name.replace('_', ' ')
                        self.simWrapper.record_vlm_sighting(
                            f"{tname_sp} is visible"
                        )

            # === ③c 目击帧留存: 本步目击列表有新增 → 存当前帧 (目击视野),
            #     供回访到场后的配对确认 (LEFT=目击帧)。容量留最近 8 帧。 ===
            sights_f = getattr(self.simWrapper, 'target_sighting_nodes', []) or []
            if sights_f and sights_f[-1][0] > getattr(self, '_sight_cap_step', -1):
                self._sight_cap_step = sights_f[-1][0]
                rgb_f = (getattr(self, '_last_obs', None) or {}).get('color_sensor')
                if rgb_f is not None:
                    if not hasattr(self, '_sighting_frames'):
                        self._sighting_frames = {}
                    self._sighting_frames[sights_f[-1][1]] = \
                        np.asarray(rgb_f).copy()
                    while len(self._sighting_frames) > 8:
                        self._sighting_frames.pop(next(iter(self._sighting_frames)))

        # === Re-scan trigger check ===
        if agent_action is not None:
            # 记录节点轨迹: 重扫描只在"真的没动"时触发
            # (连续 forward 是有进展的正常行为, 不能误触发; done=0 只说明
            #  当前没看到目标, 搜索过程中属正常)
            if not hasattr(self, '_recent_nodes'):
                self._recent_nodes = []
            self._recent_nodes.append(getattr(self.simWrapper, 'current_node', None))
            if len(self._recent_nodes) > 8:
                self._recent_nodes = self._recent_nodes[-8:]

            need_rescan = False
            reason = ""

            # Trigger 1: 6 consecutive done=0 AND 最近6步节点无变化 (原地打转)
            if hasattr(self.agent, 'stop_history') and len(self.agent.stop_history) >= 6 \
                    and len(self._recent_nodes) >= 6:
                recent6 = self.agent.stop_history[-6:]
                nodes6 = self._recent_nodes[-6:]
                if all(not v for v in recent6) and len(set(nodes6)) == 1:
                    need_rescan = True
                    reason = "6 consecutive done=0 with no node progress"

            # Trigger 2: 4 consecutive same action AND 最近4步节点无变化
            if not need_rescan and hasattr(self.agent, 'action_history') \
                    and len(self.agent.action_history) >= 4 \
                    and len(self._recent_nodes) >= 4:
                recent4 = self.agent.action_history[-4:]
                nodes4 = self._recent_nodes[-4:]
                if len(set(recent4)) == 1 and len(set(nodes4)) == 1:
                    need_rescan = True
                    reason = f"4x same action ({recent4[0]}) with no node progress"

            # (周期重扫描已移到决策前, 见 _step_env 顶部)

            if need_rescan:
                logging.info(f'[RE-SCAN] Triggered: {reason} at step {self.step}')
                self._re_scan(self.step, reset_memory=True)

        # === 物理通道兜底: 任何要回传给外层循环的动作必须带 edge_idx,
        #     否则 wrapper 回退 [0] = 180° turn_around (avdb_agent 正常
        #     路径总会挂; 这里兜住 env 各覆盖路径的漏网之鱼) ===
        if agent_action is not None \
                and agent_action is not PolarAction.stop \
                and agent_action is not PolarAction.null \
                and not hasattr(agent_action, 'edge_idx'):
            try:
                n = int(self.df.iloc[-1]['action_number']) \
                    if len(self.df) and 'action_number' in self.df.columns else 0
            except Exception:
                n = 0
            logging.warning(f'[EXEC-CHANNEL] action without edge_idx at step '
                            f'{self.step} → attaching [{n}]')
            agent_action.edge_idx = n

        return agent_action

    def _calculate_metrics(self, agent_state, agent_action, geodesic_path, max_steps):
        """Override: use relaxed threshold when VLM visually confirms target.

        In AVDB, graph nodes are fixed positions — the VLM may correctly identify
        the target from a node that's further from the registered goal position.
        When visual confirmation exists, we accept a wider distance threshold.
        """
        # Call parent for standard calculation
        metrics = super()._calculate_metrics(agent_state, agent_action, geodesic_path, max_steps)

        # === VLM 接近度成功: 由 VLM 视觉判定"目标足够接近" (专用接近度检查
        #     _vlm_proximity_check 通过后置位 _force_stop)。不用任何距离信息
        #     参与判定 (机器人不知道真实距离)。评估侧距离仅用于放宽阈值的
        #     最终校验 (VISUAL_SUCCESS_THRESHOLD 2.5m)。 ===
        distance = metrics.get('distance_to_goal', 999)
        if getattr(self, '_force_stop', False):
            if distance < self.VISUAL_SUCCESS_THRESHOLD:
                metrics['done'] = True
                metrics['finish_status'] = 'success'
                metrics['goal_reached'] = True
                metrics['spl'] = geodesic_path / max(geodesic_path, self.agent_distance_traveled)
                self.wandb_log_data.update({
                    'spl': metrics['spl'],
                    'goal_reached': metrics['goal_reached']
                })
                logging.info(f"✅ VLM PROXIMITY SUCCESS: VLM judged target close "
                             f"enough (eval distance={distance:.2f}m)")
            else:
                # bm11 接力标志: 判 close 但评估侧拒了 → 目标"看着近实际远",
                # A-block 返回后把检测框方位移交给接近锁 (走过去, 别停)
                self._force_stop_rejected = True
                logging.info(f"[PROXIMITY] VLM judged close, but eval distance "
                             f"{distance:.2f}m >= {self.VISUAL_SUCCESS_THRESHOLD}m "
                             f"→ keep searching (fp semantics preserved)")
                # bm17 绕门修复: 本路径的 stop 此前被下方仲裁块的
                # `not _force_stop` 互斥条件排除在停票门外, 拒绝又只设标志
                # 不翻 done → 连票 stop 带着 done=True 直接终结回合
                # (bm17 Step12 于 4.95m fp, 全程零 [STOP-EVIDENCE])
                self._proximity_rejected_stop_guard(agent_action, metrics, distance)

        # === 停止仲裁 (替代旧 STOP-BLOCK 一刀切拦截): VLM 要停 + 有新鲜
        #     目击时, 用 VLM 视觉判断"该不该停"。
        #     旧逻辑无条件拦掉一切停止 — 实测把 1.98m 处的正确停止拦掉,
        #     机器人随后转身走远导致失败。仲裁规则:
        #     'close' → 尊重停止 (走下面 fp-救援的放宽判定)
        #     'wrong' → 拉黑误报, 继续找
        #     'far'   → 继续找 (原拦截行为) ===
        if (agent_action is PolarAction.stop and metrics.get('done')
                and distance >= self.cfg['success_threshold']
                and (getattr(self, '_ff', None) or {}).get('metric_gate', True)
                and not getattr(self, '_force_stop', False)
                and not getattr(self, '_exhausted_stop', False)):
            # === D2 第一关: 有 YOLO 框 → 自家深度传感器米制判 (零 VLM 调用)。
            #     近 (面积≥2% 或深探≤1.0m 硬判据) → 尊重停止; (1.0,2.2]m
            #     软接近区与远 → 拒停转接近 (1.3 收紧)。
            #     G2 式 5.43m 自杀在此被拦: 框在但深探远 → AUTO-STEER 靠近。 ===
            ev = self._stop_evidence(getattr(self, '_last_obs', None))
            if ev is not None:
                ev_verdict, ev_yolo = ev
                if ev_verdict == 'close':
                    self._arbitration_close = True
                    logging.info(f'[STOP-EVIDENCE] CLOSE → honor stop at '
                                 f'{distance:.2f}m (bbox depth backed)')
                else:
                    self._reject_stop_and_approach(
                        f'[STOP-EVIDENCE] FAR → veto stop at {distance:.2f}m, '
                        f'convert vote into approach', yolo_info=ev_yolo)
                    metrics['done'] = False
                    metrics['finish_status'] = 'running'
                    self._arbitration_close = False
            else:
                sightings = getattr(self.simWrapper, 'target_sighting_nodes', [])
                arbitrated = False
                if sightings:
                    steps_ago = self.simWrapper.memory['step_count'] - sightings[-1][0]
                    if steps_ago <= 8:
                        arbitrated = True
                        # === B1 目击→停票桥 (P0): 新鲜目击 (≤3 步) 且仍在目击点
                        #     → 目击帧+当前帧到场配对, 'close' 放行停止;
                        #     'no'/不可配对 → 落回无框一律 far 的既有轨道 ===
                        verdict = None
                        if steps_ago <= 3:
                            verdict = self._bridge_arrival_verdict(distance)
                            # #49: 配对 close 须过米制守卫 — 小框=present
                            # ≠close, 拦下转接近 (锁已指对方向)
                            if verdict == 'close' \
                                    and not self._bridge_close_guard(distance):
                                verdict = 'far'
                        if verdict is None:
                            # bm11 修复: 无框停票不用整幅目测 (4.39m 判 close → fp),
                            # 改 crop 放大尺寸判据; 'close' 才放行
                            verdict = self._no_box_stop_verdict(
                                getattr(self, '_last_obs', None))
                        if verdict == 'close':
                            self._arbitration_close = True
                            logging.info(f'[STOP-ARBITRATION] VLM stop at {distance:.2f}m '
                                         f'HONORED (VLM judged target close enough)')
                            # 不拦截: 走下方 fp-救援, 距离 < 2.5m 即成功
                        else:
                            if verdict == 'far' and not self._target_approachable():
                                # 不够近, 但感知上已无法更近 (无行走选项 +
                                # 探索地图走廊阻断 + 已知空间绕障失败) →
                                # 当前视角是物理极限, 尊重停止。
                                # 系统全程未用真值距离 — 只用图像+深度可达性。
                                logging.info(
                                    '[STOP-ARBITRATION] UNREACHABLE — target visible '
                                    'but not closeable; honoring stop as best '
                                    'achievable view')
                            else:
                                if verdict == 'wrong':
                                    # 矛盾保护: 2 步内的带框判定说目标可见 (close/far)
                                    # 时, 无框 stop_vote 的 'wrong' 不足为信 — 只拒停,
                                    # 不拉黑 (拉黑会抑制整个区域的后续检测)
                                    bv = getattr(self, '_boxed_verdict', None)
                                    if bv and self.step - bv[0] <= 2 \
                                            and bv[1] in ('close', 'far'):
                                        logging.info(
                                            f'[STOP-ARBITRATION] wrong verdict overridden '
                                            f'by boxed {bv[1]} verdict {self.step - bv[0]} '
                                            f'step(s) ago — no blacklist, keep lock')
                                    else:
                                        self._invalidate_lock(
                                            'stop arbitration: VLM saw wrong object')
                                metrics['done'] = False
                                metrics['finish_status'] = 'running'
                                self._arbitration_close = False
                                logging.info(
                                    f'[STOP-ARBITRATION] VLM stop at {distance:.2f}m '
                                    f'rejected (verdict={verdict}, sighting '
                                    f'{steps_ago} step(s) ago) — keep searching')
                if not arbitrated:
                    # === D2 第三关 (③a): 无框 + 无新鲜目击 — G2 正是这种
                    #     情况 (YOLO 盲区, VLM 连票误停)。整幅缩略图里小
                    #     目标糊成一团 → crop 放大确认, 'close' 才放行。 ===
                    crop = self._crop_confirm(getattr(self, '_last_obs', None),
                                              'stop-vote-gate')
                    if crop == 'close' and self._distance_xcheck(
                            getattr(self, '_last_obs', None),
                            'crop-close')[0] == 'veto':
                        # P1: 几何核否决 — 同 'far' 轨道拒停转接近
                        crop = 'far'
                    if crop == 'close':
                        self._arbitration_close = True
                        logging.info(f'[CROP-GATE] CLOSE → honor stop at '
                                     f'{distance:.2f}m (zoomed confirm)')
                    elif crop is not None:
                        self._reject_stop_and_approach(
                            f'[CROP-GATE] {crop} → veto stop at {distance:.2f}m, '
                            f'reset consecutive votes, keep searching')
                        metrics['done'] = False
                        metrics['finish_status'] = 'running'
                        self._arbitration_close = False

        # If parent marked as false-positive (stop called but distance > threshold):
        if metrics.get('finish_status') == 'fp' and agent_action is PolarAction.stop:
            distance = metrics.get('distance_to_goal', 999)
            visual_confirm = False

            # Check 0: 停止仲裁已判 close (VLM 刚看过当前视野确认接近)
            if getattr(self, '_arbitration_close', False):
                visual_confirm = True
                logging.info("Visual confirm: stop arbitration verdict was 'close'")

            # Check 1: simWrapper has recent target sightings (YOLO or VLM)
            if hasattr(self.simWrapper, 'target_sighting_nodes') and self.simWrapper.target_sighting_nodes:
                last_sighting = self.simWrapper.target_sighting_nodes[-1]
                sight_step, sight_node, sight_desc = last_sighting
                steps_ago = self.simWrapper.memory['step_count'] - sight_step
                if steps_ago <= 3:
                    visual_confirm = True
                    logging.info(f"Visual confirm: target sighted {steps_ago} steps ago at {sight_node}")

            # Check 2: agent's stop_history shows confident recent done=1 votes
            if not visual_confirm and hasattr(self.agent, 'stop_history') and self.agent.stop_history:
                recent_stops = self.agent.stop_history[-3:]
                if sum(1 for v in recent_stops if v) >= 2:
                    visual_confirm = True
                    logging.info(f"Visual confirm: {sum(1 for v in recent_stops if v)}/3 recent done=1 votes")

            # Apply relaxed threshold when VLM visually confirmed
            if visual_confirm and distance < self.VISUAL_SUCCESS_THRESHOLD:
                metrics['finish_status'] = 'success'
                metrics['goal_reached'] = True
                metrics['spl'] = geodesic_path / max(geodesic_path, self.agent_distance_traveled)
                self.wandb_log_data.update({
                    'spl': metrics['spl'],
                    'goal_reached': metrics['goal_reached']
                })
                logging.info(f"✅ VISUAL SUCCESS: distance={distance:.2f}m < relaxed threshold "
                           f"{self.VISUAL_SUCCESS_THRESHOLD}m, VLM confirmed target")

        # === ④ 覆盖完成停: _step_env 置位的 _exhausted_stop → 强制结束
        #     回合 (not_found_exhausted, 不算成功 — 只是体面收场)。 ===
        if getattr(self, '_exhausted_stop', False) \
                and metrics.get('finish_status') != 'success':
            metrics['done'] = True
            metrics['finish_status'] = 'not_found_exhausted'
            logging.info('[EXHAUSTED-STOP] episode ends — searchable area '
                         'fully covered, target not found')

        # === 修复2 Layer 2: 快通道触发日志 (五字段) + 假阳性计量 (v1.2
        #     第四轮点4: triggered/final_close 进 episode_stats → CSV) ===
        self._fastpath_metrics(agent_action, metrics)

        return metrics

    def _fastpath_metrics(self, agent_action, metrics):
        """[FASTPATH] 触发日志五字段 + 假阳性计数 (fastpath_triggered /
        fastpath_final_close → episode_stats → batch CSV, 设计稿 §4)"""
        if agent_action is not PolarAction.stop \
                or not getattr(self.agent, 'last_stop_was_fastpath', False):
            return
        ctx = getattr(self, '_fastpath_ctx', None) or {}
        final = 'close' if (metrics.get('finish_status') == 'success'
                            or getattr(self, '_arbitration_close', False)) \
            else 'far'
        st_fp = getattr(self, '_ep_stats', None)
        if st_fp is not None:
            st_fp['fastpath_triggered'] = \
                st_fp.get('fastpath_triggered', 0) + 1
            if final == 'close':
                st_fp['fastpath_final_close'] = \
                    st_fp.get('fastpath_final_close', 0) + 1
        logging.info(
            f"[FASTPATH] step={self.step} area={ctx.get('area', 0):.3f} "
            f"conf={ctx.get('conf', 0):.2f} "
            f"bridge_left={ctx.get('bridge_left', -1)} vote=1 → "
            f"stop_request(最终判定 {final})")
