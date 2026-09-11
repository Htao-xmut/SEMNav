"""AVDB Agent — clean side-panel + YOLO injection + edge_idx mapping."""
import logging, numpy as np
from PIL import Image, ImageDraw
from agent import ObjectNavAgent, PolarAction


class AVDBAgent(ObjectNavAgent):

    def _preprocessing_module(self, obs: dict):
        """Draw a clean numbered direction panel on the right side of the image."""
        edge_options = obs.get('edge_options', [])
        if not edge_options:
            return super()._preprocessing_module(obs)

        rgb = obs['color_sensor']
        img = Image.fromarray(rgb[:, :, :3] if rgb.ndim == 3 and rgb.shape[2] >= 3 else rgb)
        w, h = img.size
        draw = ImageDraw.Draw(img)

        # Right-side panel
        panel_w = 220
        px = w - panel_w
        draw.rectangle([(px, 0), (w, h)], fill=(0, 0, 0))
        draw.rectangle([(px-2, 0), (px, h)], fill=(80, 80, 80))  # border

        # Direction colors (same as nav_demo)
        colors = {
            'turn_around': (100, 180, 255), 'rotate_left': (255, 100, 180),
            'rotate_right': (200, 100, 255), 'forward': (0, 220, 0),
            'backward': (255, 140, 0), 'left': (0, 200, 255),
            'right': (255, 200, 0),
            'rotate_left_then_forward': (255, 160, 100),
            'rotate_right_then_forward': (180, 140, 255),
        }

        row_h = min(36, (h - 20) // max(len(edge_options), 1))
        for i, opt in enumerate(edge_options):
            y = 8 + i * row_h
            direction = opt.get('direction', 'forward')
            color = colors.get(direction, (200, 200, 200))

            # Colored circle with number
            r = 10
            cx, cy = px + 18, y + row_h // 2
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color, outline=(255,255,255), width=1)
            draw.text((cx-4, cy-6), str(i), fill=(255, 255, 255))

            # Short description text
            if opt.get('composite'):
                steps = opt['composite']
                parts = []
                for et, cnt in steps:
                    if 'rotate_ccw' in et: parts.append(f'L{cnt}')
                    elif 'rotate_cw' in et: parts.append(f'R{cnt}')
                    elif 'forward' in et: parts.append(f'F{cnt}')
                desc = '+'.join(parts)
            else:
                et = opt.get('chain_type', '')
                cnt = opt.get('chain_count', 1)
                if 'rotate_ccw' in (et or ''): desc = f'L{cnt}' if cnt > 1 else 'L'
                elif 'rotate_cw' in (et or ''): desc = f'R{cnt}' if cnt > 1 else 'R'
                elif et == 'forward': desc = f'FWD{cnt}' if cnt > 1 else 'FWD'
                elif et == 'backward': desc = 'BACK'
                elif et == 'left': desc = 'LEFT'
                elif et == 'right': desc = 'RGHT'
                else: desc = et[:4]

            draw.text((cx + 15, cy - 6), desc, fill=color)

        # Top bar
        name = getattr(self, '_last_target_name', 'target')
        draw.rectangle([(0, 0), (w, 28)], fill=(0, 0, 0))
        draw.text((4, 4), f'Target: {name}  |  step {getattr(self, "step_ndx", 0)}',
                  fill=(255, 255, 255))
        draw.text((4, 16), f'{len(edge_options)} moves. Be BOLD: long FWD if path is clear!',
                  fill=(200, 210, 255))

        images = {'color_sensor': np.array(img)}
        # 深度扫描可视化图 (存进 step 日志文件夹, 供事后查看)
        if obs.get('depth_scan_img') is not None:
            images['color_sensor_depth_scan'] = obs['depth_scan_img']
        if obs.get('depth_map_img') is not None:
            images['topdown_depth_map'] = obs['depth_map_img']
        a_final = [(1.0, 0.0)] * len(edge_options)
        return a_final, images

    @staticmethod
    def _option_bearing(opt):
        """edge_option → 近似方位角 (相对相机前向, 左负右正)"""
        if opt.get('composite'):
            et, cnt = opt['composite'][0]
        else:
            et, cnt = opt.get('chain_type'), opt.get('chain_count', 1)
        base = {'rotate_cw': 30.0, 'rotate_ccw': -30.0, 'forward': 0.0,
                'backward': 180.0, 'left': -90.0, 'right': 90.0}.get(et, 0.0)
        return base * (cnt or 1)

    def _map_depth_rec_to_action(self, obs):
        """深度 Decision Trace 的推荐方向 → 附加最接近的动作编号行"""
        trace = obs.get('depth_trace', '')
        if not trace:
            return trace
        import re
        rec_m = re.search(r'Recommended:\s*([+-]?\d+)deg', trace)
        if not rec_m:
            return trace
        rec_deg = float(rec_m.group(1))
        edge_options = obs.get('edge_options', [])
        avail_lines = obs.get('avail_actions', '').split('\n')
        best, best_d = None, 181.0
        for i, opt in enumerate(edge_options):
            b = self._option_bearing(opt)
            ang_dist = abs(((rec_deg - b + 180.0) % 360.0) - 180.0)
            if ang_dist < best_d:
                best_d, best = ang_dist, i
        if best is None:
            return trace
        desc = ''
        for line in avail_lines:
            if re.match(rf'^\s*\[{best}\]', line):
                desc = re.sub(r'^\s*\[\d+\]\s*', '', line).strip()
                break
        if not desc:
            desc = str(edge_options[best].get('direction', '?'))
        return trace + f"\n  → Matching action for recommendation: [{best}] {desc}"


    def _construct_prompt(self, goal, prompt_type, num_actions=0, memory_section="", avail_actions=""):
        if not avail_actions:
            avail_actions = getattr(self, '_cached_edge_text', '')
        result = super()._construct_prompt(goal, prompt_type, num_actions, memory_section, avail_actions)
        if prompt_type == 'stopping':
            return result
        import re
        result = re.sub(r"Actions: \{0: .*?, 1: .*?, 2: .*?, 3: .*?\}\n\n", "", result)
        result = re.sub(r"You are on a camera graph.*?\n",
            "You have move options listed on the right panel.\n"
            "BE BOLD: pick LONG forward when the path is OPEN.\n"
            "Composite actions (L+FWD, R+FWD) combine turn+walk in ONE step.\n", result)
        return result

    def _choose_action(self, obs: dict):
        self._cached_edge_text = obs.get('avail_actions', '')
        edge_options = obs.get('edge_options', [])
        goal_name = obs.get('goal', '')
        if isinstance(goal_name, dict): goal_name = goal_name.get('name', '')
        self._last_target_name = goal_name

        # 把深度推荐方向映射到具体动作编号 (消除 VLM 角度↔编号歧义)
        depth_trace = self._map_depth_rec_to_action(obs)

        orig_construct = self._construct_prompt
        def avdb_construct(goal, prompt_type, num_actions=0, memory_section="", avail_actions=""):
            if prompt_type != 'stopping' and depth_trace:
                # 深度决策证据注入 memory 段 (CoT 的 OBSERVE 步骤会读取它)
                memory_section = f"{memory_section}\n\n{depth_trace}".strip()
            r = orig_construct(goal, prompt_type, num_actions, memory_section, avail_actions)
            if prompt_type == 'stopping':
                yt = obs.get('yolo_text', '')
                if yt: r = f"{yt}\n\n---\n\n{r}"
            return r
        self._construct_prompt = avdb_construct
        try:
            agent_action, metadata = super()._choose_action(obs)
        finally:
            self._construct_prompt = orig_construct

        action_number = metadata['step_metadata'].get('action_number', -1)
        if 0 <= action_number < len(edge_options):
            agent_action.edge_idx = action_number
            opt = edge_options[action_number]
            # Parse reasoning for VLM-specified angle/distance
            reasoning = metadata['step_metadata'].get('reasoning', '')
            import re

            if opt.get('chain_type') in ('rotate_cw', 'rotate_ccw'):
                angle_match = re.search(r'(\d+)\s*[°deg]', reasoning)
                if angle_match:
                    desired_deg = int(angle_match.group(1))
                    # AVDB graph: each rotate edge is ~30°. Round to nearest 30° multiple.
                    # REMOVE this rounding on real robot — just execute the exact angle.
                    steps = max(1, min(6, round(desired_deg / 30)))
                    agent_action.rotate_steps = steps
                    logging.info(f'VLM wants ~{desired_deg}° → {steps} edges = {steps*30}°')

            elif opt.get('chain_type') == 'forward':
                dist_match = re.search(r'(\d+\.?\d*)\s*m', reasoning)
                if dist_match:
                    desired_m = float(dist_match.group(1))
                    # AVDB graph: each forward edge is ~0.65m. Round to nearest 0.65m multiple.
                    # REMOVE this rounding on real robot — just execute the exact distance.
                    steps = max(1, min(5, round(desired_m / 0.65)))
                    agent_action.forward_steps = steps
                    logging.info(f'VLM wants ~{desired_m}m → {steps} forward edges = {steps*0.65:.1f}m')
            logging.info(f'Action {action_number} -> {opt}')
        else:
            agent_action.edge_idx = 0
        return agent_action, metadata
