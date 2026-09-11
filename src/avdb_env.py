"""AVDB Env — VLM navigates, graph validates. Extends ObjectNavEnv."""
import sys, os, logging, json, gzip, re, numpy as np, pandas as pd
import cv2
sys.path.insert(0, '/home/tao_h/avdb_habitat_converter/scripts')
from avdb_sim_wrapper import AVDBSimWrapper, PolarAction
from avdb_agent import AVDBAgent  # make AVDBAgent available for globals() lookup
from env import ObjectNavEnv
from utils import create_gif, log_exception
import habitat_sim

class AVDBEnv(ObjectNavEnv):
    NAV_GRAPH_PATH = "/home/tao_h/avdb_habitat_converter/data/processed/navigation_graph.json"

    # Relaxed threshold when VLM visually confirms target
    VISUAL_SUCCESS_THRESHOLD = 2.5  # meters

    def _initialize_agent(self, cfg: dict):
        """Override: use AVDBAgent which injects YOLO into stopping prompt."""
        PolarAction.default = PolarAction(cfg['agent_cfg']['default_action'], 0, 'default')
        cfg['agent_cfg']['sensor_cfg'] = cfg['sim_cfg']['sensor_cfg']
        self.agent: AVDBAgent = AVDBAgent(cfg['agent_cfg'])

    def _initialize_experiment(self):
        self.all_episodes = []
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

    def _warmup_scan(self):
        """360° environment scan with YOLO detection. Returns (summary_str, target_direction_priority).

        If YOLO detects the target, the scan records which direction and confidence,
        then generates a priority instruction to go that way first.
        Also saves a scan GIF with YOLO bounding boxes.
        """
        node = self.simWrapper.current_node
        if not node or not self.simWrapper.nav_graph:
            return "", None

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
        ep_dir = f'logs/{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'
        if gif_frames:
            scan_dir = f'{ep_dir}/warmup_scan'
            os.makedirs(scan_dir, exist_ok=True)
            for fi, frame in enumerate(gif_frames):
                frame.save(os.path.join(scan_dir, f'scan_{fi:02d}.png'))
                # Also save as step-like dirs for main GIF inclusion
                step_dir = f'{ep_dir}/warmup_scan_{fi:02d}'
                os.makedirs(step_dir, exist_ok=True)
                frame.save(os.path.join(step_dir, 'color_sensor.png'))
            # GIF
            gif_path = f'{scan_dir}/warmup_scan.gif'
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
            prompt = (
                f"You are at the starting position. Target: {target}.\n"
                f"Below are 6 photos at 60° intervals (0°/60°/120°/180°/240°/300°).\n\n"
                f"Analyze:\n"
                f"1. What ROOMS in each direction?\n"
                f"2. Where is '{target}' most likely located? At which angle?\n"
                f"3. Output exactly: FIRST_DIRECTION: <angle>°\n"
                f"   (pick one of: 0°, 60°, 120°, 180°, 240°, 300°)\n"
                f"Be concise. The FIRST_DIRECTION line is required."
            )
            imgs = [np.array(f) for f in gif_frames]
            response = self.agent.actionVLM.call_chat(0, imgs, prompt)

            # Parse FIRST_DIRECTION from VLM response
            import re
            fd_match = re.search(r'FIRST_DIRECTION:\s*(\d+)°', response)
            first_dir_angle = int(fd_match.group(1)) if fd_match else None

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
            summary += (
                f"\n**MUST-DO: Go toward the warmup-identified direction NOW.**\n"
            )
            logging.info(f'[WARMUP] Analysis complete ({len(response)} chars)')
            return summary, target_priority

        except Exception as e:
            logging.warning(f'[WARMUP] VLM analysis failed: {e}')
            if target_priority:
                return target_priority, target_priority
            return "", None

    def _initialize_episode(self, episode_ndx: int):
        self.step = 0; self.init_pos = None; self.df = pd.DataFrame({})
        self.agent_distance_traveled = 0; self.prev_agent_position = None
        self._arbitration_close = False  # 停止仲裁结果 (每 episode 复位)
        # D2/③/④ 状态复位 (跨回合不残留)
        self._exhausted_stop = False
        self._exhausted_streak = 0
        self._crop_calls = 0
        self._crop_cells = set()
        self._last_crop_step = -99
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
        self.path_calculator.requested_start = sp
        self.path_calculator.requested_ends = [ep['goals'][0]['position']]
        self.curr_shortest_path = self.simWrapper.get_path(self.path_calculator)
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

        # 方案3: 预热扫描 YOLO 高置信度发现目标 → 锁定世界方位
        if angle_to_use is not None and yolo_warmup_conf is not None \
                and yolo_warmup_conf >= 0.5:
            if not hasattr(self, '_approach_bearing'):
                self._approach_bearing = None
            if not hasattr(self, '_approach_miss'):
                self._approach_miss = 0
            yaw = self._yaw_world()
            self._approach_bearing = yaw + np.radians(angle_to_use)
            self._approach_miss = 0
            logging.info(f'[APPROACH-LOCK] warmup: angle {angle_to_use}°, '
                         f'conf {yolo_warmup_conf:.2f}, world bearing '
                         f'{np.degrees(self._approach_bearing):.0f}°')

        if angle_to_use is not None and angle_to_use > 0:
            edge_opts = obs.get('edge_options', [])

            # Best rotation: if angle <= 180, rotate RIGHT; else rotate LEFT is shorter
            if angle_to_use <= 180:
                rot_steps = angle_to_use // 30
                target_et = 'rotate_cw'
            else:
                rot_steps = (360 - angle_to_use) // 30
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
        sw = self.simWrapper
        if hasattr(sw, '_node_pos') and cur and s_node:
            try:
                if float(np.linalg.norm(
                        np.array(sw._node_pos(cur)) - np.array(sw._node_pos(s_node)))) < 0.3:
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

    def _execute_forced_return(self, obs):
        """执行回访路径的下一步 (覆盖 VLM 决策), 路径空时返回 None"""
        if not self._forced_return_path:
            return None
        next_et = self._forced_return_path.pop(0)
        edge_opts = obs.get('edge_options', [])
        idx = next((i for i, o in enumerate(edge_opts)
                    if o.get('chain_type') == next_et
                    and o.get('chain_count') == 1
                    and not o.get('composite')), None)
        if idx is None:
            logging.warning(f'[FORCED-RETURN] no option for {next_et}, aborting')
            self._forced_return_path = []
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
        img = PILImage.fromarray(rgb[:, :, :3] if rgb.ndim == 3 else rgb)
        if target_bbox:
            x1, y1, x2, y2 = target_bbox
            W, H = img.size
            draw = ImageDraw.Draw(img)
            draw.rectangle([x1 * W, y1 * H, x2 * W, y2 * H],
                           outline=(255, 0, 0), width=6)
        target_words = self.simWrapper.target_name.replace('_', ' ')
        if target_bbox:
            lead = (f"A detector marked the target '{target_words}' with a RED BOX "
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

    # === D2/③: 停票证据分级 — 修缺陷 A/B (G2 式 4 步无目击连票自杀) ===
    #     停止票不再只看 VLM 连续投票: 先用自家传感器做米制分级。
    #     有 YOLO 框 → bbox 中心深度探测 (合法: 机器人自己的深度传感器,
    #     不是真值); 无框 → ③ crop 放大确认门。
    def _stop_evidence(self, obs):
        """② 停票证据: YOLO 框 + 深度传感器 → ('close'|'far', yolo_info)

        'close': 框面积 ≥2% 画面 (目视级大) 或 bbox 中心深度 ≤2.2m
        'far'  : 有框但深度探测远, 或小框+极近物理矛盾 (探测被遮挡物
                 污染, 小框即远证据) → 拒停, 转为接近 (AUTO-STEER)
        None   : 无框/深度不可用 → 交给 ③ crop 门
        """
        if not obs:
            return None
        yolo_info = obs.get('yolo_detection') or {}
        if not yolo_info.get('target_found'):
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
                    # 尺寸-深度一致性: <1m 处的 ≥10cm 杂货必然占 >2% 画面;
                    # 小框 + 极近 = 物理矛盾 (探到的是遮挡物, 非目标)。
                    # bm14 实弹: defer crop 后放大裁剪抹掉尺寸线索, 5.43m
                    # 远处真目标被判 close → fp 停 (bm11 同病)。crop 能证
                    # present 证不了 close — 小框本身即"远"的强证据 →
                    # 直接 far: 拒停 + 接近锁走近 (近了框 ≥2% 走面积捷径)
                    if med < 1.0 and area < 0.02:
                        logging.info(f'[STOP-EVIDENCE] size-depth mismatch '
                                     f'(depth {med:.2f}m, area {area:.3f}) '
                                     f'→ FAR (small box = far evidence, '
                                     f'probe occluded)')
                        return ('far', yolo_info)
                    verdict = 'close' if med <= 2.2 else 'far'
                    logging.info(f'[STOP-EVIDENCE] bbox-grid depth '
                                 f'{med:.2f}m area {area:.3f} → {verdict.upper()}')
                    return (verdict, yolo_info)
            except Exception as e:
                logging.debug(f'[STOP-EVIDENCE] depth probe failed: {e}')
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

    def _reject_stop_and_approach(self, note, yolo_info=None):
        """拒停 + 证据转化: 重置连票计数, 有框则建接近锁 (下步 AUTO-STEER)

        连续 done=1 计数被清后, agent 侧的连票停止条件 (≥2/≥3) 不会立即
        复燃 — 机器人把这股"想停"的冲动转化为朝目标的接近动作。
        """
        logging.info(note)
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

    def _invalidate_lock(self, reason, blacklist=True, drop_sighting=True):
        """及时纠偏 + 误报黑名单

        blacklist=True  (VLM 'wrong'): 走近看过且 VLM 否认 → 拉黑目击节点
            + 接近终点 (半径 REJECT_RADIUS), 3 层拦截防回坑。
        blacklist=False (3步丢失): 接近中丢检测是正常现象 (视角变化/节点
            稀疏), 只解锁不拉黑 — 实测"丢失=误报"的假设曾把真目标区域
            拉黑导致失败。目击保留, 强制回访可再去确认一次。

        drop_sighting: 是否丢弃最近目击 (wrong→丢; 丢失→留一次回访机会)。
        """
        self._approach_bearing = None
        self._approach_miss = 99
        self._forced_return_path = []
        sw = self.simWrapper
        # === 两击规则 (A2-②): 同一区域 15 步内第 2 次"3步丢失" → 升级拉黑 ===
        # 单次丢失≠误报 (只解锁); 但同一片区域反复 丢→锁→丢 说明检测源
        # 不可靠 (低置信闪烁/误报), 再给机会只会原地打转 → 按误报处理。
        escalated = False
        if not blacklist:
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

    def _plan_bearing_detour(self, max_edges=12):
        """坐标级绕障: 沿锁定方位方向找最远可达节点, 规划图路径绕行

        世界坐标 = 导航图节点坐标 (定位已实现)。评分:
          progress = (节点-当前)·方位单位向量, lateral = 垂直偏差
          score = progress - 0.6*lateral - 0.1*跳数
        选最高分节点 → Dijkstra 路径 → 复用强制回访机制执行。
        方位约定与 _yaw_world 一致 (从 +z 起算朝 +x 为正):
          方向向量 (x,z) = (sin b, cos b)
        """
        g = getattr(self.simWrapper, 'nav_graph', None)
        cur = getattr(self.simWrapper, 'current_node', None)
        if not g or not cur or self._approach_bearing is None:
            return False
        unit = np.array([np.sin(self._approach_bearing),
                         np.cos(self._approach_bearing)])
        cur_pos = np.array([g['nodes'][cur]['world_pos'][0],
                            g['nodes'][cur]['world_pos'][2]])
        # BFS 收集 max_edges 跳内的节点 — 只在已知空间内 (去先验):
        # 真实机器人没有全屋地图, 只能沿"走过 (visited) / 深度看过
        # (探索地图 FREE 区)"的子图绕障。全图 Dijkstra 会规划出穿过
        # 未观测区域的长路 (实测 18 边), 那是数据集先验不是机器人知识。
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
        if best_node is None or best_score < 0.5:
            return False
        path = self._graph_path_to(cur, best_node)
        if not path:
            return False
        self._forced_return_path = path
        logging.info(f'[DETOUR] bearing detour to {best_node} '
                     f'({len(path)} edges, score {best_score:.1f}m)')
        return True

    def _yaw_world(self):
        """当前节点的世界朝向角 (弧度, +z 前向约定, 与深度模块一致)"""
        node = getattr(self.simWrapper, 'current_node', None)
        g = getattr(self.simWrapper, 'nav_graph', None)
        if node and g and node in g['nodes']:
            d = g['nodes'][node]['direction']
            return float(np.arctan2(d[0], d[2]))
        return 0.0

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
        best_i, best_score = None, 1e9
        for i, o in enumerate(edge_opts):
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

    def _vlm_identity_check(self, obs, yolo_info):
        """VLM 看 YOLO 框内图确认身份: 框是不是目标本身 (不是同类邻居/背景)

        YOLO 中置信 (0.30~0.60) 且目标很小时, 分类并不稳 (0.51 的可乐瓶
        检测可能只是厨房台面上的酱料瓶)。把框放大画出来问 VLM, 返回:
        'yes' / 'no' / 'unsure' (解析失败按 'unsure' 处理, 走近再验)。
        """
        dets = [d for d in yolo_info.get('all_detections', [])
                if d.get('class_name') == getattr(self.simWrapper, 'target_name', None)
                and d.get('bbox_norm')]
        if not dets:
            return 'unsure'
        b = max(dets, key=lambda d: d.get('confidence', 0))['bbox_norm']
        conf = yolo_info.get('confidence', 0)
        img = obs['color_sensor'].copy()
        H, W = img.shape[:2]
        x1, y1, x2, y2 = (int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), max(3, W // 200))
        tname = getattr(self.simWrapper, 'target_name', 'target').replace('_', ' ')
        prompt = (
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
                        self._identity_reject = {
                            **getattr(self, '_identity_reject', {}), cur: self.step}
                        self._invalidate_lock(
                            f'VLM identity: NOT the target (YOLO conf {conf:.2f})',
                            blacklist=False, drop_sighting=True)
                        logging.info('[IDENTITY] verdict=no → unlock, explore new areas')
                        return None
                    if verdict == 'unsure':
                        ci = self._pick_confirm_option(obs, yolo_info)
                        if ci is not None:
                            return self._override_and_run(
                                obs, ci, 'CONFIRM',
                                f'identity unsure (conf={conf:.2f}, area={area:.2f}) '
                                f'→ better view (closer/new angle)')
                        logging.info('[IDENTITY] unsure but no confirm option — '
                                     'proceed with servo')
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
        best_i, best_d = None, 181.0
        for i, o in enumerate(edge_opts):
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
        try:
            agent_action = super()._step_env(obs)
        finally:
            self.simWrapper._approach_active = False
        if agent_action is not None:
            # 物理生效 (bm7/bm12 大坑修复): wrapper 按 edge_idx 执行,
            # 之前只改账面 → 世界一直按 VLM raw 选择动, 日志/图在说谎
            agent_action = self._attach_exec_idx(agent_action, idx)
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

    def _attach_exec_idx(self, agent_action, idx):
        """把实际执行编号挂到返回动作上 (物理通道: wrapper 按 edge_idx 执行)

        bm7/bm12 实测大坑: env 层覆盖只改账面 (df/图/日志), 从不设置
        edge_idx → 世界一直按 VLM raw 选择动, "executed [X]" 全在说谎
        (bm7 step4: 标 executed[5], 实际动了 raw[6]); stop 单例不带
        edge_idx 还会掉进 wrapper 的 vlm_a=0 回退 = 180° turn_around
        (bm12 step22→23)。这里统一: stop/null 单例 → 新建动作对象
        (绝不在单例上挂属性, 会永久污染), 再挂 edge_idx。
        """
        if agent_action is PolarAction.stop or agent_action is PolarAction.null:
            agent_action = PolarAction(0, 0)
        try:
            agent_action.edge_idx = int(idx)
        except Exception:
            agent_action.edge_idx = 0
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
        act = PolarAction(0, 0)
        act.edge_idx = idx if idx is not None else 0
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

        Dijkstra (按边距离), 供"目击强制回访"使用。
        """
        g = getattr(self.simWrapper, 'nav_graph', None)
        if not g or from_node not in g['graph'] or to_node not in g['graph']:
            return None
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
                nd = d + e['distance']
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    prev[v] = (u, e['edge_type'])
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

    def _re_scan(self, step_num, reset_memory=True):
        """Mid-navigation 360° re-scan when VLM is stuck. Same as warmup but resets state.
        Saves PNG frames + GIF with step number in filename for visibility.

        reset_memory=False (周期性重定向): 保留 visited 记忆, 防止"一切重新变新"
        导致 VLM 放弃已搜索区域记忆而漂移。
        """
        logging.info(f'[RE-SCAN] Starting mid-navigation re-scan at step {step_num}...')
        saved_node = self.simWrapper.current_node

        # Run the scan (reuse warmup logic)
        summary, target_priority = self._warmup_scan()

        # Save re-scan as step-like directories so they appear in the navigation flow
        ep_dir = f'logs/{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'
        scan_src = f'{ep_dir}/warmup_scan'
        if os.path.exists(scan_src):
            import shutil
            # Move each scan frame as a step directory with RE-SCAN label
            for fi in range(6):
                src_png = f'{scan_src}/scan_{fi:02d}.png'
                dst_dir = f'{ep_dir}/step{step_num}_rescan_{fi:02d}'
                os.makedirs(dst_dir, exist_ok=True)
                if os.path.exists(src_png):
                    from PIL import Image, ImageDraw
                    img = Image.open(src_png)
                    draw = ImageDraw.Draw(img)
                    # Add RE-SCAN banner
                    draw.rectangle([(0, 0), (img.width, 36)], fill=(200, 50, 50))
                    draw.text((5, 4), f"RE-SCAN at step {step_num}  |  direction {fi*60} deg",
                              fill=(255, 255, 255))
                    img.save(f'{dst_dir}/color_sensor.png')
                    shutil.copy(src_png, f'{dst_dir}/color_sensor.png')
            # Also keep the original warmup_scan folder
            rescan_dst = f'{ep_dir}/rescan_step{step_num}'
            if os.path.exists(rescan_dst):
                shutil.rmtree(rescan_dst)
            shutil.move(scan_src, rescan_dst)
            logging.info(f'[RE-SCAN] Saved {rescan_dst}/ + step frames for GIF visibility')

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

            if new_angle is not None:
                # 方案3: 锁定目标方位 (世界坐标系) → 进入全自动接近模式
                # 扫描方向约定: 顺时针 = 正角 (与 rotate_cw/右转一致)
                if yolo_conf is not None and yolo_conf >= 0.5:
                    yaw = self._yaw_world()
                    self._approach_bearing = yaw + np.radians(new_angle)
                    self._approach_miss = 0
                    logging.info(f'[APPROACH-LOCK] bearing locked: scan angle '
                                 f'{new_angle}°, conf {yolo_conf:.2f}, '
                                 f'world bearing {np.degrees(self._approach_bearing):.0f}°')
                rot_steps = (360 - new_angle) // 30 if new_angle > 180 else new_angle // 30
                target_et = 'rotate_ccw' if new_angle > 180 else 'rotate_cw'
                edge_opts = obs.get('edge_options', [])
                for i, opt in enumerate(edge_opts):
                    if opt.get('chain_type') == target_et and opt.get('chain_count') == 1:
                        self._warmup_auto_action = i
                        self._warmup_auto_steps = rot_steps
                        logging.info(f'[RE-SCAN] New auto-action: {i} toward {new_angle}°')
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
            logging.info('[RE-SCAN] Complete. Fresh perspective injected.')
        else:
            self._scan_obs = None

    def _step_env(self, obs: dict):
        """Override: warmup auto-action + re-scan + 目击强制回访 + 全自动接近."""
        # 暂存当前观测: _calculate_metrics 阶段的停止仲裁需要当前视野
        self._last_obs = obs

        # === Check if re-scan obs is pending ===
        if hasattr(self, '_scan_obs') and self._scan_obs is not None:
            obs = self._scan_obs
            self._scan_obs = None

        # === 周期重扫描 (决策前!): 扫描结果当步立即生效, 不再浪费一步
        #     (Bug修复: 之前扫描在 VLM 决策之后触发, 结果要下一步才用上) ===
        if not hasattr(self, '_periodic_scanned_at'):
            self._periodic_scanned_at = set()
        if self.step > 0 and self.step % 10 == 0 \
                and self.step not in self._periodic_scanned_at:
            sightings = getattr(self.simWrapper, 'target_sighting_nodes', [])
            recent_sighting = bool(sightings) and (self.step - sightings[-1][0]) <= 8
            if not recent_sighting:
                logging.info(f'[RE-SCAN] Triggered: periodic re-orientation '
                             f'at step {self.step} (pre-action)')
                self._periodic_scanned_at.add(self.step)
                self._re_scan(self.step, reset_memory=False)
                if hasattr(self, '_scan_obs') and self._scan_obs is not None:
                    obs = self._scan_obs
                    self._scan_obs = None

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
        if yolo_now.get('target_found') and yolo_now.get('confidence', 0) >= 0.5 \
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
                # 及时纠偏: 走进一看不是目标 → 放弃锁定/目击, 转去探索其他区域
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
                elif self.step - getattr(self, '_last_identity_step', -99) >= 3:
                    self._last_identity_step = self.step
                    _id_verdict = self._vlm_identity_check(obs, yolo_now)
                    if _id_verdict == 'no':
                        self._identity_reject = {
                            **getattr(self, '_identity_reject', {}), cur: self.step}
                        sw.target_sighting_nodes = [
                            s for s in (getattr(sw, 'target_sighting_nodes', []) or [])
                            if s[1] != cur]
                        logging.info(
                            f'[IDENTITY] unarmed sighting at {cur} rejected — '
                            f'drop forced-return obligation, keep exploring')
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

        # === 停止仲裁 (替代旧 STOP-BLOCK 一刀切拦截): VLM 要停 + 有新鲜
        #     目击时, 用 VLM 视觉判断"该不该停"。
        #     旧逻辑无条件拦掉一切停止 — 实测把 1.98m 处的正确停止拦掉,
        #     机器人随后转身走远导致失败。仲裁规则:
        #     'close' → 尊重停止 (走下面 fp-救援的放宽判定)
        #     'wrong' → 拉黑误报, 继续找
        #     'far'   → 继续找 (原拦截行为) ===
        if (agent_action is PolarAction.stop and metrics.get('done')
                and distance >= self.cfg['success_threshold']
                and not getattr(self, '_force_stop', False)
                and not getattr(self, '_exhausted_stop', False)):
            # === D2 第一关: 有 YOLO 框 → 自家深度传感器米制判 (零 VLM 调用)。
            #     近 (面积≥2% 或深度≤2.2m) → 尊重停止; 远 → 拒停转接近。
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

        return metrics
