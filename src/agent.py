import logging
import math
import random
import habitat_sim
import numpy as np
import cv2
import ast
import concurrent.futures

from simWrapper import PolarAction
from utils import *
from vlm import *
from pivot import PIVOT


class Agent:
    def __init__(self, cfg: dict):
        pass

    def step(self, obs: dict):
        """Primary agent loop to map observations to the agent's action and returns metadata."""
        raise NotImplementedError

    def get_spend(self):
        """Returns the dollar amount spent by the agent on API calls."""
        return 0

    def reset(self):
        """To be called after each episode."""
        pass


class RandomAgent(Agent):
    """Example implementation of a random agent."""
    
    def step(self, obs):
        rotate = random.uniform(-0.2, 0.2)
        forward = random.uniform(0, 1)

        agent_action = PolarAction(forward, rotate)
        metadata = {
            'step_metadata': {'success': 1}, # indicating the VLM succesfully selected an action
            'logging_data': {}, # to be logged in the txt file
            'images': {'color_sensor': obs['color_sensor']} # to be visualized in the GIF
        }
        return agent_action, metadata


class VLMNavAgent(Agent):
    """
    Primary class for the VLMNav agent. Four primary components: navigability, action proposer, projection, and prompting. Runs seperate threads for stopping and preprocessing. This class steps by taking in an observation and returning a PolarAction, along with metadata for logging and visulization.
    """
    explored_color = GREY
    unexplored_color = GREEN
    map_size = 5000
    explore_threshold = 3
    voxel_ray_size = 60
    e_i_scaling = 0.8

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.fov = cfg['sensor_cfg']['fov']
        self.resolution = (
            1080 // cfg['sensor_cfg']['res_factor'],
            1920 // cfg['sensor_cfg']['res_factor']
        )

        self.focal_length = calculate_focal_length(self.fov, self.resolution[1])
        self.scale = cfg['map_scale']
        self._initialize_vlms(cfg['vlm_cfg'])       
        self.pivot = PIVOT(self.actionVLM, self.fov, self.resolution, max_action_length=cfg['max_action_dist']) if cfg['pivot'] else None

        assert cfg['navigability_mode'] in ['none', 'depth_estimate', 'segmentation', 'depth_sensor']
        self.depth_estimator = DepthEstimator() if cfg['navigability_mode'] == 'depth_estimate' else None
        self.segmentor = Segmentor() if cfg['navigability_mode'] == 'segmentation' else None
        self.reset()

    def step(self, obs: dict):
        agent_state: habitat_sim.AgentState = obs['agent_state']
        self.memory_context = obs.get('memory_context', '')  # AVDB memory
        self.yolo_text = obs.get('yolo_text', '')  # YOLO detection result
        self.avail_actions = obs.get('edge_text', '')  # Dynamic edge list
        self.edge_map = obs.get('edge_map', {0: 'turn_around'})  # idx→edge_type
        self.edge_list = obs.get('edge_list', [])  # [(type, dist, angle), ...]

        # 方案A+B: 执行验证 — 检测位移是否异常（滑移/穿墙/方向偏航）
        current_pos = np.array(agent_state.position)
        if self.prev_pos is not None and self.step_ndx > 0:
            actual_displacement = np.linalg.norm(current_pos - self.prev_pos)
            # 方案B: 异常位移检测
            self.abnormal_movement = False
            if hasattr(self, 'last_action_r') and self.last_action_r is not None:
                expected_r = self.last_action_r
                if actual_displacement > expected_r * 2.0 and expected_r > 0.1:
                    # 滑移/穿墙：实际移动远超预期
                    logging.warning(f"Abnormal movement: expected {expected_r:.2f}m, got {actual_displacement:.2f}m (slide/glitch)")
                    self.abnormal_movement = True
                elif actual_displacement > expected_r * 1.5 and expected_r > 0.1:
                    logging.info(f"Slight overshoot: expected {expected_r:.2f}m, got {actual_displacement:.2f}m")

            # AVDB: stuck detection disabled — graph edges are pre-validated walkable paths
            self.stuck_count = 0
            self.last_displacement = actual_displacement
        self.prev_pos = current_pos

        if self.step_ndx == 0:
            self.init_pos = agent_state.position

        agent_action, metadata = self._choose_action(obs)
        metadata['step_metadata'].update(self.cfg)

        if metadata['step_metadata']['action_number'] == 0:
            self.turned = self.step_ndx

        # Visualize the chosen action (nav_demo 风格: 只画选中的动作)
        chosen_action_image = obs['color_sensor'].copy()
        if 'avail_actions' in obs:
            # AVDB: 中心大箭头 + 编号圈 + 顶部信息条 (全画备选太乱, 弃用)
            self._draw_chosen_action_nav_demo(
                chosen_action_image,
                metadata['step_metadata']['action_number'],
                obs['avail_actions'],
                self.step_ndx
            )
        else:
            self._project_onto_image(
                metadata['a_final'], chosen_action_image, agent_state,
                agent_state.sensor_states['color_sensor'],
                chosen_action=metadata['step_metadata']['action_number']
            )
        metadata['images']['color_sensor_chosen'] = chosen_action_image

        self.step_ndx += 1
        return agent_action, metadata
    
    def get_spend(self):
        return self.actionVLM.get_spend() + self.stoppingVLM.get_spend()

    def reset(self):
        self.voxel_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.explored_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.stopping_calls = [-2]
        self.step_ndx = 0
        self.init_pos = None
        self.turned = -self.cfg['turn_around_cooldown']
        self.actionVLM.reset()
        # 初始化动作历史记录（短时记忆）
        self.action_history = []
        self._extended_forward = False  # A1 init
        # 停止决策历史记录
        self.stop_history = []  # list of bool (done values)
        # 方案A: 执行验证变量
        self.prev_pos = None
        self.stuck_count = 0
        self.last_displacement = 0.0
        # 方案1: 目标追踪记忆
        self.target_memory = None  # (step_ndx, description, action)
        # 方案3: 已探索区域
        self.visited_areas = set()
        self.room_history = []  # 最近每步所在的房间类型
        self.steps_without_progress = 0  # 连续无进展步数
        self.last_action_theta = None  # 方案B: 上一步的方向
        self.last_action_r = None
        self.abnormal_movement = False
        # 方案4: 死角检测
        self.dead_end_count = 0

    def _construct_prompt(self, goal: dict, prompt_type:str, num_actions: int=0, memory_section: str="", avail_actions: str=""):
        """Constructs the prompt, depending on the goal modality. """
        # Extract goal name from dictionary
        goal_name = goal['name'] if isinstance(goal, dict) else goal
        
        if prompt_type == 'stopping':
            stopping_prompt = (f"The agent is navigating to a {goal_name.upper()}. Image attached.\n\n"
            f"### STEP 1 — Visual match: List ALL items on tables/counters/shelves with their [color, shape, size, material, label].\n\n"
            f"### STEP 2 — Any item matching {goal_name}? Even without exact label — does it LOOK right?\n\n"
            f"### STEP 3 — Is it clearly visible, NOT blurry/hidden, close enough to describe details?\n\n"
            f"Return a JSON with TWO fields:\n"
            f"  'done': 1 if ALL 3 steps pass, 0 otherwise\n"
            f"  'reasoning': What you see and why you decided done=1 or done=0\n"
            f"Format: {{\"done\": <1 or 0>, \"reasoning\": \"...\"}}")
            return stopping_prompt
        if prompt_type == 'no_project':
            baseline_prompt = (f"TASK: NAVIGATE TO THE NEAREST {goal_name.upper()} and get as close to it as possible. "
                        f"Use your prior knowledge about where items are typically located within a home.\n\n"
                        f"Below are your ONLY possible directions from this position. Each jumps to a nearby camera.\n"
                        f"[0] is ALWAYS turn 180°. Analyze the IMAGE to decide which direction is best.\n\n"
                        f"{avail_actions}\n\n"
                        f"{memory_section}"
                        f"### STEP 1 — OBSERVE\n"
                        f"What room am I in? What objects do I see on tables, counters, shelves?\n"
                        f"Any object that MIGHT be a {goal_name}? Describe it briefly.\n"
                        f"READ the DEPTH SCAN block (depth-camera evidence): each candidate direction "
                        f"has a measured safe distance + confidence. Graph-verified candidates are "
                        f"guaranteed walkable; depth sectors tell how far each way stays clear.\n\n"
                        f"### STEP 2 — THINK\n"
                        f"Where would a {goal_name} most likely be? Which room, which surface?\n"
                        f"Check MEMORY: have I already searched that area? Is there a fresh direction?\n"
                        f"PRIORITY RULES (highest first):\n"
                        f"  1. MEMORY hint (CONDIMENT HOT-ZONE / target sighting / warmup direction / "
                        f"glimpsed object) → go CHECK that area FIRST, before exploring elsewhere.\n"
                        f"  2. If the scene itself looks like the target's habitat (kitchen counters, "
                        f"shelves with bottles) → search LOCALLY: rotate to face surfaces, sidestep "
                        f"along counters. Do NOT walk away down a long corridor!\n"
                        f"  3. Only with NO hint: follow the DEPTH SCAN recommendation or explore "
                        f"the longest clear direction toward unvisited areas.\n\n"
                        f"### STEP 3 — CHOOSE\n"
                        f"Pick ONE direction number from the AVAILABLE DIRECTIONS list above.\n"
                        f"Apply the PRIORITY RULES from STEP 2. Prefer directions with long safe "
                        f"distance + high confidence from DEPTH SCAN only when no higher-priority "
                        f"hint exists. Don't repeat the same action many times.\n"
                        f"Rotate to scan; forward/sidestep to move closer to target's likely location.\n\n"
                        f"OUTPUT: JSON only.\n"
                        f"{{\"reasoning\": \"<2-3 sentences>\", \"action\": <0-{num_actions-1}>}}"
            )
            return baseline_prompt
        if prompt_type == 'pivot':
            pivot_prompt = f"NAVIGATE TO THE NEAREST {goal_name.upper()} and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
            return pivot_prompt
        if prompt_type == 'action':
            action_prompt = (
            f"TASK: NAVIGATE TO THE NEAREST {goal_name.upper()}, and get as close to it as possible. "
            f"Use your prior knowledge about where items are typically located within a home.\n\n"
            f"There are {num_actions} red arrow(s) on your observation, labeled 0 to {num_actions-1}. "
            f"Each arrow = a walkable direction. You MUST pick a number 0 to {num_actions-1}.\n\n"
            f"{memory_section}"
            f"### STEP 1 — OBSERVE: What do I see? Describe objects, furniture, room. Be SPECIFIC.\n\n"
            f"### STEP 2 — LOCALIZE: What room? Where would {goal_name} likely be?\n\n"
            f"### STEP 3 — EVALUATE: Which arrow points toward target's area? Prefer UNEXPLORED. "
            f"+5 FOLLOW_UP beats -2 BACKTRACK. 3+ rotates → pick FORWARD arrow!\n\n"
            f"### STEP 4 — DECIDE: Choose BEST arrow number.\n\n"
            f"⚠️ STUCK 3+ steps → pick forward or behind arrow\n"
            f"⚠️ OBSTACLE → go AROUND\n"
            f"⚠️ NO closed doors/stairs\n\n"
            f"Return ONLY JSON: "
            f'{{"reasoning": "<analysis>", "action": <0-{num_actions-1}>}}'
            )
            return action_prompt

        raise ValueError('Prompt type must be stopping, pivot, no_project, or action')

    def _choose_action(self, obs):
        raise NotImplementedError

    def _initialize_vlms(self, cfg: dict):
        vlm_cls = globals()[cfg['model_cls']]
        
        # 检查 model_kwargs 中是否已经包含 system_instruction
        if 'system_instruction' in cfg['model_kwargs']:
            # 如果配置中已指定，直接使用配置的值
            self.actionVLM: VLM = vlm_cls(**cfg['model_kwargs'])
        else:
            # 否则使用默认的 system_instruction
            system_instruction = (
                "You are an embodied robotic assistant, with an RGB image sensor. You observe the image and instructions "
                "given to you and output a textual response, which is converted into actions that physically move you "
                "within the environment. You cannot move through closed doors. "
            )
            self.actionVLM: VLM = vlm_cls(**cfg['model_kwargs'], system_instruction=system_instruction)
        
        # 为 stopping VLM 使用独立的 system_instruction
        stopping_kwargs = dict(cfg['model_kwargs'])
        stopping_kwargs.pop('system_instruction', None)  # 移除 action 的 system_instruction
        stopping_kwargs['system_instruction'] = (
            "You are an embodied robotic assistant. Your job is to look at an image and determine "
            "whether the agent has reached its navigation goal. Respond ONLY with a JSON object "
            "in the format {'done': <1 or 0>}."
        )
        self.stoppingVLM: VLM = vlm_cls(**stopping_kwargs)

    def _run_threads(self, obs: dict, stopping_images: list[np.array], goal):
        """Concurrently runs the stopping thread to determine if the agent should stop, and the preprocessing thread to calculate potential actions."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            preprocessing_thread = executor.submit(self._preprocessing_module, obs)
            stopping_thread = executor.submit(self._stopping_module, stopping_images, goal)

            a_final, images = preprocessing_thread.result()
            called_stop, stopping_response = stopping_thread.result()
        
        if called_stop:
            logging.info('Model called stop')
            self.stopping_calls.append(self.step_ndx)
            # If the model calls stop, turn off navigability and explore bias tricks
            if self.cfg['navigability_mode'] != 'none' and self.cfg['project']:
                new_image = obs['color_sensor'].copy()
                a_final = self._project_onto_image(
                    self._get_default_arrows(), new_image, obs['agent_state'],
                    obs['agent_state'].sensor_states['color_sensor']
                )
                images['color_sensor'] = new_image

        step_metadata = {
            'action_number': -10,
            'success': 1,
            'pivot': 1 if self.pivot is not None else 0,
            'model': self.actionVLM.name,
            'agent_location': obs['agent_state'].position,
            'called_stopping': called_stop
        }
        return a_final, images, step_metadata, stopping_response

    def _preprocessing_module(self, obs: dict):
        """Excutes the navigability, action_proposer and projection submodules."""
        agent_state = obs['agent_state']
        images = {'color_sensor': obs['color_sensor'].copy()}
        if not self.cfg['project']:
            # Actions for the w/o proj baseline
            a_final = {
                (self.cfg['max_action_dist'], -0.28 * np.pi): 1,
                (self.cfg['max_action_dist'], 0): 2,
                (self.cfg['max_action_dist'], 0.28 * np.pi): 3,
            }
            return a_final, images

        if self.cfg['navigability_mode'] == 'none':
            a_final = [
                # Actions for the w/o nav baseline
                (self.cfg['max_action_dist'], -0.36 * np.pi),
                (self.cfg['max_action_dist'], -0.28 * np.pi),
                (self.cfg['max_action_dist'], 0),
                (self.cfg['max_action_dist'], 0.28 * np.pi),
                (self.cfg['max_action_dist'], 0.36 * np.pi)
            ]
        else:
            a_initial = self._navigability(obs)
            a_final = self._action_proposer(a_initial, agent_state)

        a_final_projected = self._projection(a_final, images, agent_state)
        images['voxel_map'] = self._generate_voxel(a_final_projected, agent_state=agent_state)
        return a_final_projected, images

    def _stopping_module(self, stopping_images: list[np.array], goal):
        """Determines if the agent should stop."""
        stopping_prompt = self._construct_prompt(goal, 'stopping')
        # Use call_chat for vision models to properly handle images
        stopping_response = self.stoppingVLM.call_chat(0, stopping_images, stopping_prompt)
        dct = self._eval_response(stopping_response)
        if 'done' in dct and int(dct['done']) == 1:
            return True, stopping_response
        
        return False, stopping_response

    def _navigability(self, obs: dict):
        """Generates the set of navigability actions. Uses graph edges if available."""
        agent_state: habitat_sim.AgentState = obs['agent_state']

        # AVDB: graph-based navigability — use graph edges instead of depth sensor
        if obs.get('graph_directions'):
            return self._graph_navigability(obs)

        sensor_state = agent_state.sensor_states['color_sensor']
        rgb_image = obs['color_sensor']
        depth_image = obs[f'depth_sensor']
        if self.cfg['navigability_mode'] == 'depth_estimate':
            depth_image = self.depth_estimator.call(rgb_image)
        if self.cfg['navigability_mode'] == 'segmentation':
            depth_image = None

        navigability_mask = self._get_navigability_mask(
            rgb_image, depth_image, agent_state, sensor_state
        )

        sensor_range =  np.deg2rad(self.fov / 2) * 1.5

        all_thetas = np.linspace(-sensor_range, sensor_range, self.cfg['num_theta'])
        start = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )

        a_initial = []
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state, depth_image)
            if r_i is not None:
                self._update_voxel(
                    r_i, theta_i, agent_state,
                    clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling
                )
                a_initial.append((r_i, theta_i))

        return a_initial

    def _graph_navigability(self, obs):
        """AVDB: Use graph edges as navigable directions (instead of depth sensor)."""
        graph_dirs = obs.get('graph_directions', [])
        if not graph_dirs:
            return []
        # Return graph edges directly as walkable (r, theta) actions
        return [(float(r), float(theta)) for r, theta in graph_dirs]

    def _action_proposer(self, a_initial: list, agent_state: habitat_sim.AgentState):
        """Refines the initial set of actions, ensuring spacing and adding a bias towards exploration."""
        min_angle = self.fov/self.cfg['spacing_ratio']
        explore_bias = self.cfg['explore_bias']
        clip_frac = self.cfg['clip_frac']
        clip_mag = self.cfg['max_action_dist']

        explore = explore_bias > 0
        unique = {}
        for mag, theta in a_initial:
            if theta in unique:
                unique[theta].append(mag)
            else:
                unique[theta] = [mag]
        arrowData = []

        topdown_map = self.voxel_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color
        for theta, mags in unique.items():
            # Reference the map to classify actions as explored or unexplored
            mag = min(mags)
            cart = [self.e_i_scaling*mag*np.sin(theta), 0, -self.e_i_scaling*mag*np.cos(theta)]
            global_coords = local_to_global(agent_state.position, agent_state.rotation, cart)
            grid_coords = self._global_to_grid(global_coords)
            score = (sum(np.all((topdown_map[grid_coords[1]-2:grid_coords[1]+2, grid_coords[0]] == self.explored_color), axis=-1)) + 
                    sum(np.all(topdown_map[grid_coords[1], grid_coords[0]-2:grid_coords[0]+2] == self.explored_color, axis=-1)))
            arrowData.append([clip_frac*mag, theta, score<3])

        arrowData.sort(key=lambda x: x[1])
        thetas = set()
        out = []
        filter_thresh = 0.75
        filtered = list(filter(lambda x: x[0] > filter_thresh, arrowData))

        filtered.sort(key=lambda x: x[1])
        if filtered == []:
            return []
        if explore:
            # Add unexplored actions with spacing, starting with the longest one
            f = list(filter(lambda x: x[2], filtered))
            if len(f) > 0:
                longest = max(f, key=lambda x: x[0])
                longest_theta = longest[1]
                smallest_theta = longest[1]
                longest_ndx = f.index(longest)
            
                out.append([min(longest[0], clip_mag), longest[1], longest[2]])
                thetas.add(longest[1])
                for i in range(longest_ndx+1, len(f)):
                    if f[i][1] - longest_theta > (min_angle*0.9):
                        out.append([min(f[i][0], clip_mag), f[i][1], f[i][2]])
                        thetas.add(f[i][1])
                        longest_theta = f[i][1]
                for i in range(longest_ndx-1, -1, -1):
                    if smallest_theta - f[i][1] > (min_angle*0.9):
                        
                        out.append([min(f[i][0], clip_mag), f[i][1], f[i][2]])
                        thetas.add(f[i][1])
                        smallest_theta = f[i][1]

                for r_i, theta_i, e_i in filtered:
                    if theta_i not in thetas and min([abs(theta_i - t) for t in thetas]) > min_angle*explore_bias:
                        out.append((min(r_i, clip_mag), theta_i, e_i))
                        thetas.add(theta)

        if len(out) == 0:
            # if no explored actions or no explore bias
            longest = max(filtered, key=lambda x: x[0])
            longest_theta = longest[1]
            smallest_theta = longest[1]
            longest_ndx = filtered.index(longest)
            out.append([min(longest[0], clip_mag), longest[1], longest[2]])
            
            for i in range(longest_ndx+1, len(filtered)):
                if filtered[i][1] - longest_theta > min_angle:
                    out.append([min(filtered[i][0], clip_mag), filtered[i][1], filtered[i][2]])
                    longest_theta = filtered[i][1]
            for i in range(longest_ndx-1, -1, -1):
                if smallest_theta - filtered[i][1] > min_angle:
                    out.append([min(filtered[i][0], clip_mag), filtered[i][1], filtered[i][2]])
                    smallest_theta = filtered[i][1]


        if (out == [] or max(out, key=lambda x: x[0])[0] < self.cfg['min_action_dist']) and (self.step_ndx - self.turned) < self.cfg['turn_around_cooldown']:
            return self._get_default_arrows()
        
        out.sort(key=lambda x: x[1])
        return [(mag, theta) for mag, theta, _ in out]

    def _projection(self, a_final: list, images: dict, agent_state: habitat_sim.AgentState):
        """
        Projection component of VLMnav. Projects the arrows onto the image, annotating them with action numbers.
        Note actions that are too close together or too close to the boundaries of the image will not get projected.
        """
        a_final_projected = self._project_onto_image(
            a_final, images['color_sensor'], agent_state,
            agent_state.sensor_states['color_sensor']
        )

        if not a_final_projected and (self.step_ndx - self.turned < self.cfg['turn_around_cooldown']):
            logging.info('No actions projected and cannot turn around')
            a_final = self._get_default_arrows()
            a_final_projected = self._project_onto_image(
                a_final, images['color_sensor'], agent_state,
                agent_state.sensor_states['color_sensor']
            )

        return a_final_projected

    def _prompting(self, goal, a_final: list, images: dict, step_metadata: dict):
        """
        Prompting component of VLMNav. Constructs the textual prompt and calls the action model.
        Parses the response for the chosen action number.
        """
        logging_data = {}
        response = ""
        action_prompt = ""
        goal_name = goal['name'] if isinstance(goal, dict) else goal
        
        try:
            prompt_type = 'action' if self.cfg['project'] else 'no_project'
            yolo_section = getattr(self, 'yolo_text', '')
            mem = getattr(self, 'memory_context', '')
            avail = getattr(self, 'avail_actions', '')
            combined = (yolo_section + '\n\n' + mem) if yolo_section else mem
            n_actions = 1 + len(getattr(self, 'edge_list', []))  # action 0 + edges
            action_prompt = self._construct_prompt(goal, prompt_type, num_actions=n_actions, memory_section=combined, avail_actions=avail)

            # 构建输入图像列表（仅RGB，不使用俯视图）
            prompt_images = [images['color_sensor']]

            if 'goal_image' in images:
                prompt_images.append(images['goal_image'])

            # 注入动作历史（短时记忆）
            if hasattr(self, 'action_history') and self.action_history:
                recent_actions = self.action_history[-5:] if len(self.action_history) >= 5 else self.action_history

                # 统计最近动作的频率
                from collections import Counter
                action_counts = Counter(recent_actions)
                most_common_action = action_counts.most_common(1)[0][0] if action_counts else None
                most_common_count = action_counts.most_common(1)[0][1] if action_counts else 0

                # 方案5: 软化反循环 — 只在位移≈0时触发警告
                history_warning = ""
                if most_common_count >= 2 and hasattr(self, 'last_displacement') and self.last_displacement < 0.05:
                    history_warning = (
                        f"\n\n💡 TIP: Action {most_common_action} repeated {most_common_count} times without moving. "
                        f"Try a different direction to make progress.\n"
                    )
                elif most_common_count >= 4:
                    history_warning = (
                        f"\n\n💡 TIP: Action {most_common_action} repeated {most_common_count} times. "
                        f"Consider a DIFFERENT action to explore new areas.\n"
                    )

                action_prompt += (
                    f"\n\nRECENT ACTIONS: {recent_actions}\n"
                    f"{history_warning}"
                )

            # 方案1: 目标追踪记忆 (强化版)
            if hasattr(self, 'target_memory') and self.target_memory:
                mem_step, mem_desc, mem_action = self.target_memory
                steps_ago = self.step_ndx - mem_step
                if steps_ago <= 2:
                    # 刚看到过 → 最强提醒
                    action_prompt += (
                        f"\n\n🎯 CRITICAL: You JUST saw the {goal_name.upper()} {steps_ago} step(s) ago! "
                        f"It was {mem_desc}. It is VERY CLOSE.\n"
                        f"→ If you can still see it: pick the arrow pointing TOWARD it. Do NOT turn away.\n"
                        f"→ If you turned away: pick an arrow to go BACK toward where it was.\n"
                        f"→ The target is NEARBY. This is NOT the time to explore — it's time to APPROACH.\n"
                    )
                else:
                    action_prompt += (
                        f"\n\n🎯 TARGET MEMORY: You saw the {goal_name.upper()} {steps_ago} steps ago "
                        f"({mem_desc}, via action {mem_action}). "
                        f"If you've lost it, try going back toward that direction.\n"
                    )

            # 方案3: 探索状态 + 区域耗尽检测
            if hasattr(self, 'visited_areas') and self.visited_areas:
                areas_str = ", ".join(list(self.visited_areas)[-6:])
                action_prompt += (
                    f"\n\nEXPLORED AREAS: {areas_str}\n"
                    f"PRIORITIZE arrows leading to doorways, hallways, or rooms you haven't fully explored yet.\n"
                )

            # 短暂看到目标的提醒 (called_stop 来自 step_metadata)
            if step_metadata.get('called_stopping') and hasattr(self, 'stop_history') and sum(1 for v in self.stop_history[-4:] if v) < 2:
                action_prompt += (
                    f"\n\n📍 NOTE: Stopping module says the {goal_name.upper()} IS visible now. "
                    f"You are close! Pick the arrow going TOWARD it. Do NOT turn away.\n"
                )

            # 方案B: 异常位移恢复
            if hasattr(self, 'abnormal_movement') and self.abnormal_movement:
                action_prompt += (
                    f"\n\n🔄 RECOVERY: Last step caused unexpected movement (possible slide/glitch). "
                    f"Your position may have shifted unpredictably.\n"
                    f"→ RE-ORIENT: Look around carefully. Identify where you are NOW.\n"
                    f"→ If the target is visible, head TOWARD it. If not, use doorways/hallways to explore.\n"
                )

            # 区域耗尽检测: 基于 done=0 的连续步数
            if hasattr(self, 'stop_history') and len(self.stop_history) >= 6:
                # 计算最近连续多少个 done=0
                consecutive_zeros = 0
                for v in reversed(self.stop_history):
                    if not v:
                        consecutive_zeros += 1
                    else:
                        break
                if consecutive_zeros >= 8:
                    action_prompt += (
                        f"\n\n💡 TIP: {consecutive_zeros} steps without spotting the {goal_name.upper()}. "
                        f"It may be in a different area. Try exploring new directions — doorways, hallways, or different rooms.\n"
                    )
                elif consecutive_zeros >= 5:
                    action_prompt += (
                        f"\n\n💡 {consecutive_zeros} steps without seeing {goal_name.upper()}. "
                        f"Consider trying a different direction or checking nearby rooms.\n"
                    )

            response = self.actionVLM.call_chat(self.cfg['context_history'], prompt_images, action_prompt)

            # Debug: log response immediately after receiving it
            logging.info(f"Received response from VLM (length: {len(response)})")
            if len(response) > 0:
                logging.info(f"Response preview: {response[:200]}...")
            
            response_dict = self._eval_response(response)
            
            if 'action' not in response_dict:
                raise ValueError(f"Parsed response missing 'action' key: {response_dict}")

            # AVDB: action can be string (edge name) or int (legacy)
            raw_action = response_dict['action']
            if isinstance(raw_action, str):
                # Map edge name to action number via avail_edges
                avail_edges = getattr(self, 'avail_edges', [])
                if raw_action in avail_edges:
                    step_metadata['action_number'] = avail_edges.index(raw_action)
                else:
                    step_metadata['action_number'] = 2  # fallback to forward
                step_metadata['edge_name'] = raw_action
                logging.info(f"Edge action: {raw_action} → action_number={step_metadata['action_number']}")
            else:
                step_metadata['action_number'] = int(raw_action)
            step_metadata['success'] = 1
            
            # 提取 reasoning（如果存在）
            if 'reasoning' in response_dict:
                step_metadata['reasoning'] = response_dict['reasoning']
                # A1: Extended forward detection
                reasoning_lower = response_dict['reasoning'].lower()
                if 'extended forward' in reasoning_lower or 'long forward' in reasoning_lower:
                    self._extended_forward = True
                else:
                    self._extended_forward = False
            
        except Exception as e:
            logging.error(f"CRITICAL ERROR in _prompting: {e}", exc_info=True)
            step_metadata['success'] = 0
            step_metadata['action_number'] = -1
            step_metadata['error'] = str(e)

        finally:
            # 无论成功与否，都记录 Prompt、Response 和 Reasoning，方便调试
            logging_data['ACTION_NUMBER'] = step_metadata.get('action_number')
            logging_data['PROMPT'] = action_prompt
            logging_data['RESPONSE'] = response
            # 添加 reasoning 字段
            if 'reasoning' in step_metadata:
                logging_data['REASONING'] = step_metadata['reasoning']

        return step_metadata, logging_data, response

    def _get_navigability_mask(self, rgb_image: np.array, depth_image: np.array, agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose):
        """
        Get the navigability mask for the current state, according to the configured navigability mode.
        """
        if self.cfg['navigability_mode'] == 'segmentation':
            navigability_mask = self.segmentor.get_navigability_mask(rgb_image)
        else:
            thresh = 1 if self.cfg['navigability_mode'] == 'depth_estimate' else self.cfg['navigability_height_threshold']
            height_map = depth_to_height(depth_image, self.fov, sensor_state.position, sensor_state.rotation)
            navigability_mask = abs(height_map - (agent_state.position[1] - 0.04)) < thresh

        return navigability_mask

    def _get_default_arrows(self):
        """
        Get the action options for when the agent calls stop the first time, or when no navigable actions are found.
        """
        angle = np.deg2rad(self.fov / 2) * 0.7
        
        default_actions = [
            (self.cfg['stopping_action_dist'], -angle),
            (self.cfg['stopping_action_dist'], -angle / 4),
            (self.cfg['stopping_action_dist'], angle / 4),
            (self.cfg['stopping_action_dist'], angle)
        ]
        
        default_actions.sort(key=lambda x: x[1])
        return default_actions

    def _get_radial_distance(self, start_pxl: tuple, theta_i: float, navigability_mask: np.ndarray, 
                             agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose, 
                             depth_image: np.ndarray):
        """
        Calculates the distance r_i that the agent can move in the direction theta_i, according to the navigability mask.
        """
        agent_point = [2 * np.sin(theta_i), 0, -2 * np.cos(theta_i)]
        end_pxl = agent_frame_to_image_coords(
            agent_point, agent_state, sensor_state, 
            resolution=self.resolution, focal_length=self.focal_length
        )
        if end_pxl is None or end_pxl[1] >= self.resolution[0]:
            return None, None

        H, W = navigability_mask.shape

        # Find intersections of the theoretical line with the image boundaries
        intersections = find_intersections(start_pxl[0], start_pxl[1], end_pxl[0], end_pxl[1], W, H)
        if intersections is None:
            return None, None

        (x1, y1), (x2, y2) = intersections
        num_points = max(abs(x2 - x1), abs(y2 - y1)) + 1
        x_coords = np.linspace(x1, x2, num_points)
        y_coords = np.linspace(y1, y2, num_points)

        out = (int(x_coords[-1]), int(y_coords[-1]))
        if not navigability_mask[int(y_coords[0]), int(x_coords[0])]:
            return 0, theta_i

        for i in range(num_points - 4):
            # Trace pixels until they are not navigable
            y = int(y_coords[i])
            x = int(x_coords[i])
            if sum([navigability_mask[int(y_coords[j]), int(x_coords[j])] for j in range(i, i + 4)]) <= 2:
                out = (x, y)
                break

        if i < 5:
            return 0, theta_i

        if self.cfg['navigability_mode'] == 'segmentation':
            #Simple estimation of distance based on number of pixels
            r_i = 0.0794 * np.exp(0.006590 * i) + 0.616

        else:
            #use depth to get distance
            out = (np.clip(out[0], 0, W - 1), np.clip(out[1], 0, H - 1))
            camera_coords = unproject_2d(
                *out, depth_image[out[1], out[0]], resolution=self.resolution, focal_length=self.focal_length
            )
            local_coords = global_to_local(
                agent_state.position, agent_state.rotation,
                local_to_global(sensor_state.position, sensor_state.rotation, camera_coords)
            )
            r_i = np.linalg.norm([local_coords[0], local_coords[2]])

        return r_i, theta_i

    def _can_project(self, r_i: float, theta_i: float, agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose):
        """
        Checks whether the specified polar action can be projected onto the image, i.e., not too close to the boundaries of the image.
        """
        agent_point = [r_i * np.sin(theta_i), 0, -r_i * np.cos(theta_i)]
        end_px = agent_frame_to_image_coords(
            agent_point, agent_state, sensor_state, 
            resolution=self.resolution, focal_length=self.focal_length
        )
        if end_px is None:
            return None

        if (
            self.cfg['image_edge_threshold'] * self.resolution[1] <= end_px[0] <= (1 - self.cfg['image_edge_threshold']) * self.resolution[1] and
            self.cfg['image_edge_threshold'] * self.resolution[0] <= end_px[1] <= (1 - self.cfg['image_edge_threshold']) * self.resolution[0]
        ):
            return end_px
        return None

    def _project_onto_image(self, a_final: list, rgb_image: np.ndarray, agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose, chosen_action: int=None):
        """
        Projects a set of actions onto a single image. Keeps track of action-to-number mapping.
        """
        scale_factor = rgb_image.shape[0] / 1080
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = BLACK
        circle_color = WHITE
        projected = {}
        if chosen_action == -1:
            put_text_on_image(
                rgb_image, 'TERMINATING EPISODE', text_color=GREEN, text_size=4 * scale_factor,
                location='center', text_thickness=math.ceil(3 * scale_factor), highlight=False
            )
            return projected

        start_px = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state, 
            resolution=self.resolution, focal_length=self.focal_length
        )
        for action_idx, (r_i, theta_i) in enumerate(a_final):
            text_size = 2.4 * scale_factor
            text_thickness = math.ceil(3 * scale_factor)

            end_px = self._can_project(r_i, theta_i, agent_state, sensor_state)
            if end_px is not None:
                action_name = action_idx  # 从 0 开始，与动作编号一致
                projected[(r_i, theta_i)] = action_name

                cv2.arrowedLine(rgb_image, tuple(start_px), tuple(end_px), RED, math.ceil(5 * scale_factor), tipLength=0.0)
                text = str(action_name)
                (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
                circle_center = (end_px[0], end_px[1])
                circle_radius = max(text_width, text_height) // 2 + math.ceil(15 * scale_factor)

                if chosen_action is not None and action_name == chosen_action:
                    cv2.circle(rgb_image, circle_center, circle_radius, GREEN, -1)
                else:
                    cv2.circle(rgb_image, circle_center, circle_radius, circle_color, -1)
                cv2.circle(rgb_image, circle_center, circle_radius, RED, math.ceil(2 * scale_factor))
                text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
                cv2.putText(rgb_image, text, text_position, font, text_size, text_color, text_thickness)

        if (self.step_ndx - self.turned) >= self.cfg['turn_around_cooldown'] or self.step_ndx == self.turned or (chosen_action == 0):
            text = '0'
            text_size = 3.1 * scale_factor
            text_thickness = math.ceil(3 * scale_factor)
            (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
            circle_center = (math.ceil(0.05 * rgb_image.shape[1]), math.ceil(rgb_image.shape[0] / 2))
            circle_radius = max(text_width, text_height) // 2 + math.ceil(15 * scale_factor)

            if chosen_action is not None and chosen_action == 0:
                cv2.circle(rgb_image, circle_center, circle_radius, GREEN, -1)
            else:
                cv2.circle(rgb_image, circle_center, circle_radius, circle_color, -1)
            cv2.circle(rgb_image, circle_center, circle_radius, RED, math.ceil(2 * scale_factor))
            text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
            cv2.putText(rgb_image, text, text_position, font, text_size, text_color, text_thickness)
            cv2.putText(rgb_image, 'TURN AROUND', (text_position[0] // 2, text_position[1] + math.ceil(80 * scale_factor)), font, text_size * 0.75, RED, text_thickness)

        return projected


    def _draw_chosen_action_nav_demo(self, rgb_image: np.ndarray,
                                     chosen_num: int, avail_text: str,
                                     step_idx: int):
        """nav_demo 风格可视化: 只画"选中的下一步动作"。

        画面中心一个方向大箭头 (前/后/左侧移/右侧移/原地转/U形掉头,
        转再走=前进箭头+小旋转弧) + 白底编号圈 (编号=提示词里的 [N]) +
        顶部黑色信息条 (步数 + 动作原文)。备选方向不再全画 (太乱),
        完整备选列表见 details.txt 的 PROMPT。
        风格与 avdb_habitat_converter/scripts/nav_demo.py 保持一致。
        """
        H, W = rgb_image.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        s = H / 480.0                    # nav_demo 基准 640x480, sz=60
        u = s                            # 箭头坐标缩放
        sz = int(60 * s)
        cx, cy = W // 2, H // 2

        if chosen_num == -1:             # 终止步
            put_text_on_image(
                rgb_image, 'TERMINATING EPISODE', text_color=GREEN,
                text_size=2 * s, location='center',
                text_thickness=max(2, round(3 * s)), highlight=False
            )
            return

        # --- 选中项原文: "[N] turn LEFT ~30deg ..." (VLM 看到的同一行) ---
        desc = ''
        tag = f'[{chosen_num}]'
        for line in avail_text.split('\n'):
            line = line.strip()
            if line.startswith(tag):
                desc = line[len(tag):].strip()
                break
        desc = ''.join(c for c in desc if ord(c) < 128)  # ★ 等非 ASCII 去掉
        d = desc.upper()

        # --- 动作类型 (关键词判定, 顺序敏感: 掉头 > 复合 > 侧移/后退 > 原地转) ---
        if 'TURN 180' in d:
            kind = 'turn_around'
        elif 'THEN FWD' in d:
            kind = 'composite'
        elif 'SIDESTEP LEFT' in d:
            kind = 'left'
        elif 'SIDESTEP RIGHT' in d:
            kind = 'right'
        elif 'BACKWARD' in d:
            kind = 'backward'
        elif 'TURN LEFT' in d:
            kind = 'rotate_ccw'
        elif 'TURN RIGHT' in d:
            kind = 'rotate_cw'
        elif 'FWD' in d or 'FORWARD' in d:
            kind = 'forward'
        elif chosen_num == 0:
            kind = 'turn_around'
        else:
            kind = 'forward'

        ARROW_TEXT = {'forward': 'FORWARD', 'backward': 'BACKWARD',
                      'left': 'SIDESTEP LEFT', 'right': 'SIDESTEP RIGHT',
                      'rotate_cw': 'ROTATE CW', 'rotate_ccw': 'ROTATE CCW',
                      'turn_around': 'TURN 180', 'composite': 'TURN+FWD'}
        ARROW_COLOR = {                                    # BGR
            'forward': (0, 255, 0), 'backward': (0, 165, 255),
            'left': (255, 200, 0), 'right': (0, 220, 255),
            'rotate_cw': (255, 120, 200), 'rotate_ccw': (200, 130, 255),
            'turn_around': (255, 255, 255), 'composite': (0, 255, 0)}
        color = ARROW_COLOR.get(kind, (0, 255, 0))
        line_w = max(2, round(2 * s))

        # --- 中心大箭头 (坐标同 nav_demo, 按 u 缩放) ---
        if kind == 'composite':
            # 样式3 接力箭头: 旋转色弧段(先转) 平滑接 绿色直行段(再走),
            # 底部起点朝上出发, 末端三角箭头 (用户选定样式)
            ccw = 'TURN L' in d
            sign = -1 if ccw else 1
            turn = math.radians(60) * sign      # 显示 60° (真实 30° 太平难读)
            R, L = 150 * u, 190 * u
            sx0, sy0 = W // 2, int(0.82 * H)
            h0 = -math.pi / 2                   # 起点朝上 = 当前直前
            sv = (math.cos(h0 + sign * math.pi / 2),
                  math.sin(h0 + sign * math.pi / 2))
            ccx, ccy = sx0 + R * sv[0], sy0 + R * sv[1]
            arc = []
            for i in range(25):
                h = h0 + turn * i / 24.0
                rx = math.cos(h + sign * math.pi / 2)
                ry = math.sin(h + sign * math.pi / 2)
                arc.append((ccx - R * rx, ccy - R * ry))
            h_end = h0 + turn
            ux, uy = math.cos(h_end), math.sin(h_end)
            ex, ey = arc[-1]
            straight = [(ex + ux * L * (i + 1) / 8.0, ey + uy * L * (i + 1) / 8.0)
                        for i in range(8)]
            Pa = np.array([(int(x), int(y)) for x, y in arc], np.int32)
            Pb = np.array([(int(x), int(y)) for x, y in straight], np.int32)
            rot_col = (200, 130, 255) if ccw else (255, 120, 200)
            tk = max(4, round(7 * s))
            cv2.polylines(rgb_image, [Pa], False, rot_col, tk, cv2.LINE_AA)
            cv2.polylines(rgb_image, [Pb], False, GREEN, tk, cv2.LINE_AA)
            # 末端三角箭头 (沿直行方向)
            tipx, tipy = straight[-1]
            bx, by = straight[-2]
            vx, vy = tipx - bx, tipy - by
            vn = math.hypot(vx, vy)
            vx, vy = vx / vn, vy / vn
            nx, ny = -vy, vx
            hl = 26 * u
            head = np.array([(tipx + vx * hl * 0.6, tipy + vy * hl * 0.6),
                             (bx - vx * hl * 0.4 + nx * hl * 0.45,
                              by - vy * hl * 0.4 + ny * hl * 0.45),
                             (bx - vx * hl * 0.4 - nx * hl * 0.45,
                              by - vy * hl * 0.4 - ny * hl * 0.45)],
                            dtype=np.int32)
            cv2.fillPoly(rgb_image, [head], GREEN)
            lx, ly = sx0 - int(45 * u), sy0
        elif kind == 'forward':
            pts = np.array([(cx, cy - sz), (cx - 25 * u, cy - sz + 40 * u),
                            (cx - 10 * u, cy - sz + 40 * u),
                            (cx - 10 * u, cy + 25 * u),
                            (cx + 10 * u, cy + 25 * u),
                            (cx + 10 * u, cy - sz + 40 * u),
                            (cx + 25 * u, cy - sz + 40 * u)], dtype=np.int32)
            cv2.fillPoly(rgb_image, [pts], color)
            cv2.polylines(rgb_image, [pts], True, BLACK, 1)
            lx, ly = cx, int(cy - sz - 20 * u)
        elif kind == 'backward':
            pts = np.array([(cx, cy + sz), (cx - 25 * u, cy + sz - 40 * u),
                            (cx - 10 * u, cy + sz - 40 * u),
                            (cx - 10 * u, cy - 25 * u),
                            (cx + 10 * u, cy - 25 * u),
                            (cx + 10 * u, cy + sz - 40 * u),
                            (cx + 25 * u, cy + sz - 40 * u)], dtype=np.int32)
            cv2.fillPoly(rgb_image, [pts], color)
            cv2.polylines(rgb_image, [pts], True, BLACK, 1)
            lx, ly = cx, int(cy + sz + 20 * u)
        elif kind == 'left':
            pts = np.array([(cx - sz, cy), (cx - sz + 40 * u, cy - 25 * u),
                            (cx - sz + 40 * u, cy - 10 * u),
                            (cx + 25 * u, cy - 10 * u),
                            (cx + 25 * u, cy + 10 * u),
                            (cx - sz + 40 * u, cy + 10 * u),
                            (cx - sz + 40 * u, cy + 25 * u)], dtype=np.int32)
            cv2.fillPoly(rgb_image, [pts], color)
            cv2.polylines(rgb_image, [pts], True, BLACK, 1)
            lx, ly = int(cx - sz - 20 * u), cy
        elif kind == 'right':
            pts = np.array([(cx + sz, cy), (cx + sz - 40 * u, cy - 25 * u),
                            (cx + sz - 40 * u, cy - 10 * u),
                            (cx - 25 * u, cy - 10 * u),
                            (cx - 25 * u, cy + 10 * u),
                            (cx + sz - 40 * u, cy + 10 * u),
                            (cx + sz - 40 * u, cy + 25 * u)], dtype=np.int32)
            cv2.fillPoly(rgb_image, [pts], color)
            cv2.polylines(rgb_image, [pts], True, BLACK, 1)
            lx, ly = int(cx + sz + 20 * u), cy
        else:
            # 原地转 / 掉头: 圆弧箭头 (角度增大=画面顺时针, y 向下)
            r = 0.7 * sz
            if kind == 'rotate_cw':
                a0, a1 = -90.0, 190.0      # 12 点起顺时针扫 280°
            elif kind == 'rotate_ccw':
                a0, a1 = -90.0, -370.0     # 12 点起逆时针扫 280°
            else:                          # U 形掉头 180°
                a0, a1 = -90.0, 90.0
            step = 5.0 * (1 if a1 > a0 else -1)
            a = a0
            while abs(a1 - a) > abs(step):
                a2 = a + step
                cv2.line(rgb_image,
                         (int(cx + r * math.cos(math.radians(a))),
                          int(cy + r * math.sin(math.radians(a)))),
                         (int(cx + r * math.cos(math.radians(a2))),
                          int(cy + r * math.sin(math.radians(a2)))),
                         color, line_w)
                a = a2
            # 箭头头部: 弧终点沿运动方向切线的小三角
            sign = 1 if a1 > a0 else -1
            te = math.radians(a1 - step * sign)
            tvx, tvy = -math.sin(te) * sign, math.cos(te) * sign
            ex = cx + r * math.cos(math.radians(a1))
            ey = cy + r * math.sin(math.radians(a1))
            tipL, baseL = 14 * u, 9 * u
            nx, ny = -tvy, tvx
            head = np.array([
                (ex + tvx * tipL, ey + tvy * tipL),
                (ex - tvx * baseL + nx * baseL * 0.8,
                 ey - tvy * baseL + ny * baseL * 0.8),
                (ex - tvx * baseL - nx * baseL * 0.8,
                 ey - tvy * baseL - ny * baseL * 0.8)], dtype=np.int32)
            cv2.fillPoly(rgb_image, [head], color)
            if kind == 'rotate_cw':
                lx, ly = int(cx + sz + 25 * u), int(cy - 0.7 * sz)
            elif kind == 'rotate_ccw':
                lx, ly = int(cx - sz - 45 * u), int(cy - 0.7 * sz)
            else:                          # 掉头: 编号圈放弧右侧
                lx, ly = int(cx + r + 32 * u), cy

        # --- 编号圈 (白底黑字, 编号 = 提示词 [N]) ---
        num_txt = str(chosen_num)
        fs, ft = 0.9 * s, max(2, round(2 * s))
        (tw2, th2), _ = cv2.getTextSize(num_txt, font, fs, ft)
        cr = int(15 * u) + max(tw2, th2) // 2
        lx = int(np.clip(lx, cr + 2, W - cr - 2))
        ly = int(np.clip(ly, cr + 2, H - cr - 2))
        cv2.circle(rgb_image, (lx, ly), cr, WHITE, -1)
        cv2.circle(rgb_image, (lx, ly), cr, BLACK, max(2, round(2 * s)))
        cv2.putText(rgb_image, num_txt, (lx - tw2 // 2, ly + th2 // 2),
                    font, fs, BLACK, ft)

        # --- 顶部信息条 (半透明黑底: 步数 + 动作原文) ---
        bar_h = int(55 * s)
        roi = rgb_image[0:bar_h, 0:W]
        blk = np.zeros_like(roi)
        cv2.addWeighted(blk, 0.78, roi, 0.22, 0, roi)
        fs2, ft2 = 0.58 * s, max(1, round(1 * s))
        cv2.putText(rgb_image, f'Step {step_idx + 1} | VLMnav ObjectNav (AVDB graph)',
                    (int(8 * s), int(20 * s)), font, fs2, WHITE, ft2)
        line2 = f'> {ARROW_TEXT.get(kind, "?")} (option #{chosen_num}) | {desc}'
        if len(line2) > 100:
            line2 = line2[:97] + '...'
        cv2.putText(rgb_image, line2,
                    (int(8 * s), int(42 * s)), font, fs2, color, ft2)

    def _update_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, clip_dist: float, clip_frac: float):
        """Update the voxel map to mark actions as explored or unexplored"""
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark unexplored regions
        unclipped = max(r - 0.5, 0)
        local_coords = np.array([unclipped * np.sin(theta), 0, -unclipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.voxel_map, agent_coords, point, self.unexplored_color, self.voxel_ray_size)

        # Mark explored regions
        clipped = min(clip_frac * r, clip_dist)
        local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.explored_map, agent_coords, point, self.explored_color, self.voxel_ray_size)

    def _global_to_grid(self, position: np.ndarray, rotation=None):
        """Convert global coordinates to grid coordinates in the agent's voxel map"""
        dx = position[0] - self.init_pos[0]
        dz = position[2] - self.init_pos[2]
        resolution = self.voxel_map.shape
        x = int(resolution[1] // 2 + dx * self.scale)
        y = int(resolution[0] // 2 + dz * self.scale)

        if rotation is not None:
            original_coords = np.array([x, y, 1])
            new_coords = np.dot(rotation, original_coords)
            new_x, new_y = new_coords[0], new_coords[1]
            return (int(new_x), int(new_y))

        return (x, y)

    def _generate_voxel(self, a_final: dict, zoom: int=9, agent_state: habitat_sim.AgentState=None, chosen_action: int=None):
        """For visualization purposes, add the agent's position and actions onto the voxel map"""
        agent_coords = self._global_to_grid(agent_state.position)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        right_coords = self._global_to_grid(right)
        delta = abs(agent_coords[0] - right_coords[0])

        topdown_map = self.voxel_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color

        text_size = 1.25
        text_thickness = 1
        rotation_matrix = None
        agent_coords = self._global_to_grid(agent_state.position, rotation=rotation_matrix)
        x, y = agent_coords
        font = cv2.FONT_HERSHEY_SIMPLEX

        if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown']:
            a_final[(0.75, np.pi)] = 0

        for (r, theta), action in a_final.items():
            local_pt = np.array([r * np.sin(theta), 0, -r * np.cos(theta)])
            global_pt = local_to_global(agent_state.position, agent_state.rotation, local_pt)
            act_coords = self._global_to_grid(global_pt, rotation=rotation_matrix)

            # Draw action arrows and labels
            cv2.arrowedLine(topdown_map, tuple(agent_coords), tuple(act_coords), RED, 5, tipLength=0.05)
            text = str(action)
            (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
            circle_center = (act_coords[0], act_coords[1])
            circle_radius = max(text_width, text_height) // 2 + 15

            if chosen_action is not None and action == chosen_action:
                cv2.circle(topdown_map, circle_center, circle_radius, GREEN, -1)
            else:
                cv2.circle(topdown_map, circle_center, circle_radius, WHITE, -1)

            text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
            cv2.circle(topdown_map, circle_center, circle_radius, RED, 1)
            cv2.putText(topdown_map, text, text_position, font, text_size, RED, text_thickness + 1)

        # Draw agent's current position
        cv2.circle(topdown_map, agent_coords, radius=15, color=RED, thickness=-1)

        # Zoom the map
        max_x, max_y = topdown_map.shape[1], topdown_map.shape[0]
        x1 = max(0, x - delta)
        x2 = min(max_x, x + delta)
        y1 = max(0, y - delta)
        y2 = min(max_y, y + delta)

        zoomed_map = topdown_map[y1:y2, x1:x2]
        return zoomed_map

    def _action_number_to_polar(self, action_number: int, a_final: list):
        """Converts the chosen action number to its PolarAction instance"""
        try:
            action_number = int(action_number)
            if action_number <= len(a_final) and action_number > 0:
                r, theta = a_final[action_number - 1]
                return PolarAction(r, -theta)
            if action_number == 0:
                return PolarAction(0, np.pi)
        except ValueError:
            pass

        logging.info("Bad action number: " + str(action_number))
        return PolarAction.default

    def _eval_response(self, response: str):
        """Converts the VLM response string into a dictionary, if possible"""
        if not isinstance(response, str) or len(response) == 0:
            return {}
        
        # Preprocessing: Remove markdown code blocks if present
        cleaned_response = response
        if '```' in response:
            # Extract content between ``` markers
            import re
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                cleaned_response = json_match.group(1)
        
        # 1. Try JSON first
        if '{' in cleaned_response and '}' in cleaned_response:
            try:
                eval_resp = ast.literal_eval(cleaned_response[cleaned_response.rindex('{'):cleaned_response.rindex('}') + 1])
                if isinstance(eval_resp, dict) and ('action' in eval_resp or 'done' in eval_resp):
                    return eval_resp
            except:
                pass

        # Also try on original response in case cleaning removed something important
        if cleaned_response != response and '{' in response and '}' in response:
            try:
                eval_resp = ast.literal_eval(response[response.rindex('{'):response.rindex('}') + 1])
                if isinstance(eval_resp, dict) and ('action' in eval_resp or 'done' in eval_resp):
                    return eval_resp
            except:
                pass
        
        # 2. Robust Regex Extraction for Natural Language
        import re
        # Look for patterns like "action 3", "action number 3", "choose 3", "option 3"
        patterns = [
            r"(?:action|option|number|choose)\s*(\d+)",
            r"'action'\s*[:=]\s*(\d+)",
            r"\"action\"\s*[:=]\s*(\d+)"
        ]
        
        # Search in both cleaned and original response
        for text in [cleaned_response, response]:
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Take the last match as it's usually the final decision
                    try:
                        action_num = int(matches[-1])
                        return {'action': action_num}
                    except ValueError:
                        continue

        return {}


class GOATAgent(VLMNavAgent):
 
    def _choose_action(self, obs: dict):
        agent_state = obs['agent_state']
        goal = obs['goal']

        if goal['mode'] == 'image':
            stopping_images = [obs['color_sensor'], goal['goal_image']]
        else:
            stopping_images = [obs['color_sensor']]

        a_final, images, step_metadata, stopping_response = self._run_threads(obs, stopping_images, goal)
        if goal['mode'] == 'image':
            images['goal_image'] = goal['goal_image']

        step_metadata.update({
            'goal': goal['name'],
            'goal_mode': goal['mode']
        })

        # If model calls stop two times in a row, we return the stop action and terminate the episode
        if len(self.stopping_calls) >= 2 and self.stopping_calls[-2] == self.step_ndx - 1:
            step_metadata['action_number'] = -1
            agent_action = PolarAction.stop
            logging_data = {}
        else:
            if self.pivot is not None:
                pivot_instruction = self._construct_prompt(goal, 'pivot')
                agent_action, pivot_images = self.pivot.run(
                    obs['color_sensor'], pivot_instruction,
                    agent_state, agent_state.sensor_states['color_sensor'],
                    goal_image=goal['goal_image'] if goal['mode'] == 'image' else None
                )
                images.update(pivot_images)
                logging_data = {}
                step_metadata['action_number'] = -100
            else:
                step_metadata, logging_data, _ = self._prompting(goal, a_final, images, step_metadata)
                agent_action = self._action_number_to_polar(step_metadata['action_number'], list(a_final))

                # AVDB: attach edge_idx to PolarAction for dynamic edge selection
                agent_action.edge_idx = step_metadata['action_number']

                # A1: Extended forward — VLM requested longer forward in reasoning
                if getattr(self, '_extended_forward', False) and step_metadata['action_number'] == 2:
                    agent_action.r = agent_action.r * 2.5  # ~2m instead of ~0.8m
                    logging.info(f"Extended forward: r={agent_action.r:.2f}m")

                # 更新动作历史记录（短时记忆）
                if hasattr(self, 'action_history') and step_metadata.get('action_number', -1) >= 0:
                    self.action_history.append(step_metadata['action_number'])
                    # 限制历史记录长度为最近 20 步
                    if len(self.action_history) > 20:
                        self.action_history = self.action_history[-20:]

        logging_data['STOPPING RESPONSE'] = stopping_response
        metadata = {
            'step_metadata': step_metadata,
            'logging_data': logging_data,
            'a_final': a_final,
            'images': images
        }
        return agent_action, metadata
    
    def _construct_prompt(self, goal: dict, prompt_type: str, num_actions=0):
        """Constructs the prompt, depending on the goal modality. """
        if goal['mode'] == 'object':
            task = f'Navigate to the nearest {goal["name"]}'
            first_instruction = f'Find the nearest {goal["name"]} and navigate as close as you can to it. '
        if goal['mode'] == 'description':
            first_instruction = f"Find and navigate to the {goal['lang_desc']}. Navigate as close as you can to it. "
            task = first_instruction
        if goal['mode'] == 'image':
            task = f'Navigate to the specific {goal["name"]} shown in the image labeled GOAL IMAGE. Pay close attention to the details, and note you may see the object from a different angle than in the goal image. Navigate as close as you can to it '
            first_instruction = f"Observe the image labeled GOAL IMAGE. Find this specific {goal['name']} shown in the image and navigate as close as you can to it. "

        if prompt_type == 'stopping':        
            stopping_prompt = (f"The agent has the following navigation task: \n{task}\n. The agent has sent you an image taken from its current location{' as well as the goal image. ' if goal['mode'] == 'image' else '. '} "
                                f'Your job is to determine whether the agent is close to the specified {goal["name"].upper()}'
                                f"First, tell me what you see in the image, and tell me if there is a {goal['name']} that matches the description. Then, return 1 if the agent is close to the {goal['name']}, and 0 if it isn't. Format your answer in the json {{'done': <1 or 0>}}")
            return stopping_prompt

        if prompt_type == 'pivot':
            return f'{first_instruction} Use your prior knowledge about where items are typically located within a home. '
        
        if prompt_type == 'no_project':
            baseline_prompt = (f"TASK: {first_instruction} use your prior knowledge about where items are typically located within a home. "
                        "You have four possible actions: {0: Turn completely around, 1: Turn left, 2: Move straight ahead, 3: Turn right}. "
                        f"First, tell me what you see, and if you have any leads on finding the {goal['name']}. Second, tell me which general direction you should go in. "
                        f"Lastly, explain which action acheives that best, and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"             
            )
            return baseline_prompt
        
        if prompt_type == 'action':
            action_prompt = (f"TASK: {first_instruction} use your prior knowledge about where items are typically located within a home. "
            f"There are {num_actions} red arrow(s) superimposed onto your observation, which represent potential actions. " 
            f"These are labeled with numbers from 0 to {num_actions-1} in white circles. The number on each arrow IS the action number you should return. "
            f"IMPORTANT: You MUST choose a number between 0 and {num_actions-1}. Choosing a number outside this range will cause an error. "
            f"{'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
            f"You MUST respond with ONLY a JSON object containing TWO fields:\n"
            f"  1. 'reasoning': A brief explanation of what you see and why you chose that action (2-3 sentences).\n"
            f"  2. 'action': The action number you want to take (integer from 0 to {num_actions-1}).\n\n"
            f"Example response format:\n"
            f'{{"reasoning": "I see a hallway ahead with an open door on the left. Bedrooms are usually off hallways, so I should move forward.", "action": 2}}\n\n'
            f"Do NOT include any text before or after the JSON. Do NOT use markdown code blocks. Return ONLY the raw JSON object."
            )
            return action_prompt

        raise ValueError('Prompt type must be stopping, pivot, no_project, or action')

    def reset_goal(self):
        """Called after every subtask of GOAT. Notably does not reset the voxel map, only resets all areas to be unexplored"""
        self.stopping_calls = [self.step_ndx-2]
        self.explored_map = np.zeros_like(self.explored_map)
        self.turned = self.step_ndx - self.cfg['turn_around_cooldown']


class ObjectNavAgent(VLMNavAgent):

    def _choose_action(self, obs: dict):
        agent_state = obs['agent_state']
        goal = obs['goal']

        # 先跑 preprocessing 获取动作候选
        a_final, images = self._preprocessing_module(obs)
        step_metadata = {
            'action_number': -10,
            'success': 1,
            'called_stopping': False
        }

        # === Stopping: 纯视觉判断，不加任何历史提示 ===
        stopping_images = [obs['color_sensor']]
        stopping_prompt = self._construct_prompt(goal, 'stopping')

        stopping_response = self.stoppingVLM.call_chat(0, stopping_images, stopping_prompt)
        dct = self._eval_response(stopping_response)
        called_stop = dct.get('done') == 1
        if called_stop:
            logging.info('Model called stop')

        # 记录 stopping 决策历史（只记done值，不记action VLM的reasoning）
        if hasattr(self, 'stop_history'):
            self.stop_history.append(called_stop)
            if len(self.stop_history) > 20:
                self.stop_history = self.stop_history[-20:]

        # === 停止条件: 连续2次 done=1（当前 + 上一步）===
        # done=0 时自动重置计数，防止基于旧票误停
        consecutive_stops = 0
        for v in reversed(self.stop_history):
            if v:
                consecutive_stops += 1
            else:
                break
        # 仅当前帧 done=1 计入（called_stop 是当前帧的投票）
        if called_stop:
            consecutive_stops += 0  # already counted in stop_history

        should_stop = False
        stop_reason = ""

        # 条件1: 连续2次 done=1（可靠信号：一直在看到目标）
        if self.step_ndx >= 3 and consecutive_stops >= 2 and called_stop:
            should_stop = True
            stop_reason = f"consecutive stops: {consecutive_stops} in a row"
        # 条件2: 连续3次 done=1（即使当前没看到，但如果刚连续3次看到过也停）
        elif self.step_ndx >= 3 and consecutive_stops >= 3:
            should_stop = True
            stop_reason = f"3 consecutive recent stops"
        # 条件3: 卡住了 + 累计≥1票
        elif self.step_ndx >= 5 and self.stuck_count >= 3 and sum(1 for v in self.stop_history if v) >= 1:
            should_stop = True
            stop_reason = f"stuck+seen: stuck {self.stuck_count} steps"

        if should_stop:
            logging.info(f'Stop by {stop_reason}')
            self.stopping_calls.append(self.step_ndx)
            if self.cfg['navigability_mode'] != 'none' and self.cfg['project']:
                new_image = obs['color_sensor'].copy()
                a_final = self._project_onto_image(
                    self._get_default_arrows(), new_image, obs['agent_state'],
                    obs['agent_state'].sensor_states['color_sensor']
                )
                images['color_sensor'] = new_image

            step_metadata['action_number'] = -1
            agent_action = PolarAction.stop
            logging_data = {}
            step_metadata.update({
                'action_number': -1,
                'success': 1,
                'called_stopping': True
            })
        else:
            if called_stop:
                logging.info(f'Stop vote recorded (consecutive={consecutive_stops}, need 2 to stop)')

            if self.pivot is not None:
                pivot_instruction = self._construct_prompt(goal, 'pivot')
                agent_action, pivot_images = self.pivot.run(
                    obs['color_sensor'], pivot_instruction,
                    agent_state, agent_state.sensor_states['color_sensor']
                )
                images.update(pivot_images)
                logging_data = {}
                step_metadata['action_number'] = -100
            else:
                step_metadata, logging_data, _ = self._prompting(goal, a_final, images, step_metadata)
                chosen_action = step_metadata.get('action_number', -1)

                # 方案1: 记录目标追踪记忆
                reasoning = step_metadata.get('reasoning', '').lower()
                target_name = goal['name'].lower() if isinstance(goal, dict) else goal.lower()
                see_patterns = [f'i see {target_name}', f'i saw {target_name}',
                                f'can see {target_name}', f'{target_name} is visible',
                                f'{target_name} in the', f'{target_name} near',
                                f'{target_name} on the', f'found {target_name}']
                # 否定句排除 ("was last seen"/"not found" 等回忆表述不是目击,
                # 否则会形成假目击 → VLM 折返漂移)
                neg_words = ['not', 'no ', "wasn't", 'was not', 'without',
                             'last seen', 'was seen', 'unlocated', 'out of view']
                saw_target = False
                for p in see_patterns:
                    idx = reasoning.find(p)
                    if idx < 0:
                        continue
                    window = reasoning[max(0, idx - 40):idx]
                    if any(n in window for n in neg_words):
                        continue
                    saw_target = True
                    break
                if saw_target:
                    # 提取位置描述
                    desc = "somewhere nearby"
                    for phrase in ['near the window', 'behind the couch', 'on the cabinet',
                                   'in the background', 'to the right', 'to the left',
                                   'in the living room', 'near the center']:
                        if phrase in reasoning:
                            desc = phrase
                            break
                    self.target_memory = (self.step_ndx, desc, chosen_action)
                    logging.info(f'Target memory updated: saw {target_name} {desc} via action {chosen_action}')

                # 更新"无进展"计数器: 看到目标或进入新房间→重置, 否则+1
                entered_new_room = False
                for room in ['doorway', 'hallway', 'staircase', 'stairs', 'new room',
                              'unexplored', 'different room', 'another room']:
                    if room in reasoning and room not in (self.room_history[-1] if self.room_history else ''):
                        entered_new_room = True
                        break
                if hasattr(self, 'steps_without_progress'):
                    if saw_target or entered_new_room or (hasattr(self, 'last_displacement') and self.last_displacement > 0.5):
                        self.steps_without_progress = 0
                    else:
                        self.steps_without_progress += 1

                # 方案3: 记录已探索区域 + 房间历史
                room_type = 'unknown'
                for room in ['living room', 'kitchen', 'hallway', 'bedroom', 'bathroom',
                              'dining room', 'dining', 'corridor', 'staircase', 'stairs',
                              'office', 'closet', 'laundry']:
                    if room in reasoning:
                        room_type = room
                        break
                if hasattr(self, 'room_history'):
                    self.room_history.append(room_type)
                    if len(self.room_history) > 30:
                        self.room_history = self.room_history[-30:]

                if hasattr(self, 'visited_areas'):
                    if hasattr(self, 'last_displacement') and self.last_displacement > 0.3:
                        self.visited_areas.add(room_type if room_type != 'unknown' else f'area{self.step_ndx}')

                # 方案4: 死角逃脱
                num_available = len(list(a_final))
                if num_available <= 1:
                    self.dead_end_count += 1
                else:
                    self.dead_end_count = 0

                if self.dead_end_count >= 2 and num_available <= 1:
                    # 卡在死角，强制转身
                    logging.warning(f"Dead-end escape: only {num_available} arrow(s) for {self.dead_end_count} steps, forcing turn")
                    step_metadata['action_number'] = 0  # force turn around
                    step_metadata['forced_exploration'] = True
                    chosen_action = 0

                # 强制探索: action 0 循环 OR agent 物理卡住 OR done=0太久
                force_explore = False
                force_reason = ""
                if chosen_action == 0 and hasattr(self, 'action_history'):
                    recent = self.action_history[-4:] if len(self.action_history) >= 4 else []
                    if len(recent) >= 4 and all(a == 0 for a in recent):
                        force_explore = True
                        force_reason = "0-loop"
                if hasattr(self, 'stuck_count') and self.stuck_count >= 3:
                    force_explore = True
                    force_reason = f"physically stuck ({self.stuck_count} steps)"
                # 连续20步done=0 → 强制随机探索 (AVDB: 放宽避免误触发)
                if hasattr(self, 'stop_history') and len(self.stop_history) >= 20:
                    recent_dones = self.stop_history[-20:]
                    if all(not v for v in recent_dones):
                        force_explore = True
                        force_reason = "20 steps done=0"

                if force_explore:
                    valid_nonzero = [i for i in range(len(list(a_final))) if i != 0]
                    if valid_nonzero:
                        import random
                        forced = random.choice(valid_nonzero)
                        # bm15 实弹: 20 步 done=0 后每步 random 抢走选择权,
                        # 无视深度图 FRESH-AREA 推荐 → 掉头回已探索区游荡到
                        # max_steps。修: 跟随 wrapper 结构化推荐方位
                        # obs['depth_rec_deg'] (含区域新鲜度调整 + 滞回);
                        # 文本正则仅兜底 (bm18 实弹: depth_trace 里无
                        # "Recommended: ±Ndeg" 字样, 27 次 forced 0 次跟随)
                        try:
                            rec_deg = obs.get('depth_rec_deg')
                            if rec_deg is None:
                                import re as _re
                                m = _re.search(r'Recommended: ([+-]\d+)deg',
                                               obs.get('depth_trace') or '')
                                if m:
                                    rec_deg = float(m.group(1))
                            if rec_deg is not None:
                                rec_deg = float(rec_deg)

                                def _opt_bearing(o):
                                    if o.get('composite'):
                                        et, cnt = o['composite'][0]
                                    else:
                                        et, cnt = (o.get('chain_type'),
                                                   o.get('chain_count', 1))
                                    base = {'rotate_cw': 30.0, 'rotate_ccw': -30.0,
                                            'forward': 0.0, 'backward': 180.0,
                                            'left': -90.0, 'right': 90.0}.get(et, None)
                                    if base is None:
                                        return None
                                    return base * (cnt or 1)

                                best, best_d = None, 1e9
                                for i in valid_nonzero:
                                    b = _opt_bearing(list(a_final)[i])
                                    if b is None:
                                        continue
                                    d = abs((b - rec_deg + 540) % 360 - 180)
                                    if d < best_d:
                                        best_d, best = d, i
                                if best is not None and best_d <= 100:
                                    forced = best
                                    logging.info(
                                        f'Forced exploration follows DEPTH rec '
                                        f'{rec_deg:+.0f}deg (fresh-area aware)')
                        except Exception:
                            pass
                        logging.warning(f"Forced exploration ({force_reason}): forcing action {forced}")
                        step_metadata['action_number'] = forced
                        step_metadata['forced_exploration'] = True
                        chosen_action = forced

                agent_action = self._action_number_to_polar(chosen_action, list(a_final))

                # 方案E: 深度障碍检测 — 如果前方有障碍，缩短步长
                if hasattr(self, 'focal_length') and 'depth_sensor' in obs:
                    depth_img = np.array(obs['depth_sensor'])
                    h, w = depth_img.shape[:2]
                    theta_deg = agent_action.theta * 180 / np.pi
                    # 将 theta 映射到像素列
                    hfov = self.fov
                    px_col = int(w/2 + (theta_deg / (hfov/2)) * (w/2))
                    px_col = max(0, min(w-1, px_col))
                    # 取该列中间偏下区域的深度（agent前方地面区域）
                    row_start, row_end = int(h*0.5), int(h*0.85)
                    region = depth_img[row_start:row_end, px_col]
                    if region.size > 0:
                        min_depth = float(np.min(region[region > 0])) if np.any(region > 0) else 99.0
                        # 如果前方障碍物比计划步长更近，缩短步长
                        if min_depth < agent_action.r * 1.2 and min_depth > 0.1:
                            new_r = max(0.2, min_depth * 0.7)  # 走到障碍物70%距离处
                            if new_r < agent_action.r:
                                logging.info(f'Depth check: obstacle at {min_depth:.2f}m, reducing step {agent_action.r:.2f}→{new_r:.2f}m')
                                agent_action = PolarAction(new_r, agent_action.theta, agent_action.type)

                # 方案B: 异常位移标记 (在step()中检测，这里只存储action方向供下步比较)
                self.last_action_theta = agent_action.theta
                self.last_action_r = agent_action.r

                # 更新动作历史
                if hasattr(self, 'action_history') and chosen_action >= 0:
                    self.action_history.append(chosen_action)
                    if len(self.action_history) > 20:
                        self.action_history = self.action_history[-20:]

        step_metadata['object'] = goal
        logging_data['STOPPING RESPONSE'] = stopping_response
        metadata = {
            'step_metadata': step_metadata,
            'logging_data': logging_data,
            'a_final': a_final,
            'images': images
        }
        return agent_action, metadata

    def _construct_prompt(self, goal: dict, prompt_type:str, num_actions: int=0, memory_section: str="", avail_actions: str=""):
        """Constructs the prompt, depending on the goal modality. """
        # Extract goal name from dictionary
        goal_name = goal['name'] if isinstance(goal, dict) else goal
        
        if prompt_type == 'stopping':
            stopping_prompt = (f"The agent is navigating to a {goal_name.upper()}. Image attached.\n\n"
            f"### STEP 1 — Visual match: List ALL items on tables/counters/shelves with their [color, shape, size, material, label].\n\n"
            f"### STEP 2 — Any item matching {goal_name}? Even without exact label — does it LOOK right?\n\n"
            f"### STEP 3 — Is it clearly visible, NOT blurry/hidden, close enough to describe details?\n\n"
            f"Return a JSON with TWO fields:\n"
            f"  'done': 1 if ALL 3 steps pass, 0 otherwise\n"
            f"  'reasoning': What you see and why you decided done=1 or done=0\n"
            f"Format: {{\"done\": <1 or 0>, \"reasoning\": \"...\"}}")
            return stopping_prompt
        if prompt_type == 'no_project':
            baseline_prompt = (f"TASK: NAVIGATE TO THE NEAREST {goal_name.upper()} and get as close to it as possible. "
                        f"Use your prior knowledge about where items are typically located within a home.\n\n"
                        f"Below are your ONLY possible directions from this position. Each jumps to a nearby camera.\n"
                        f"[0] is ALWAYS turn 180°. Analyze the IMAGE to decide which direction is best.\n\n"
                        f"{avail_actions}\n\n"
                        f"{memory_section}"
                        f"### STEP 1 — OBSERVE\n"
                        f"What room am I in? What objects do I see on tables, counters, shelves?\n"
                        f"Any object that MIGHT be a {goal_name}? Describe it briefly.\n"
                        f"READ the DEPTH SCAN block (depth-camera evidence): each candidate direction "
                        f"has a measured safe distance + confidence. Graph-verified candidates are "
                        f"guaranteed walkable; depth sectors tell how far each way stays clear.\n\n"
                        f"### STEP 2 — THINK\n"
                        f"Where would a {goal_name} most likely be? Which room, which surface?\n"
                        f"Check MEMORY: have I already searched that area? Is there a fresh direction?\n"
                        f"PRIORITY RULES (highest first):\n"
                        f"  1. MEMORY hint (CONDIMENT HOT-ZONE / target sighting / warmup direction / "
                        f"glimpsed object) → go CHECK that area FIRST, before exploring elsewhere.\n"
                        f"  2. If the scene itself looks like the target's habitat (kitchen counters, "
                        f"shelves with bottles) → search LOCALLY: rotate to face surfaces, sidestep "
                        f"along counters. Do NOT walk away down a long corridor!\n"
                        f"  3. Only with NO hint: follow the DEPTH SCAN recommendation or explore "
                        f"the longest clear direction toward unvisited areas.\n\n"
                        f"### STEP 3 — CHOOSE\n"
                        f"Pick ONE direction number from the AVAILABLE DIRECTIONS list above.\n"
                        f"Apply the PRIORITY RULES from STEP 2. Prefer directions with long safe "
                        f"distance + high confidence from DEPTH SCAN only when no higher-priority "
                        f"hint exists. Don't repeat the same action many times.\n"
                        f"Rotate to scan; forward/sidestep to move closer to target's likely location.\n\n"
                        f"OUTPUT: JSON only.\n"
                        f"{{\"reasoning\": \"<2-3 sentences>\", \"action\": <0-{num_actions-1}>}}"
            )
            return baseline_prompt
        if prompt_type == 'pivot':
            pivot_prompt = f"NAVIGATE TO THE NEAREST {goal_name.upper()} and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
            return pivot_prompt
        if prompt_type == 'action':
            action_prompt = (
            f"TASK: NAVIGATE TO THE NEAREST {goal_name.upper()}, and get as close to it as possible. "
            f"Use your prior knowledge about where items are typically located within a home.\n\n"
            f"There are {num_actions} red arrow(s) on your observation, labeled 0 to {num_actions-1}. "
            f"Each arrow = a walkable direction. You MUST pick a number 0 to {num_actions-1}.\n\n"
            f"{memory_section}"
            f"### STEP 1 — OBSERVE: What do I see? Describe objects, furniture, room. Be SPECIFIC.\n\n"
            f"### STEP 2 — LOCALIZE: What room? Where would {goal_name} likely be?\n\n"
            f"### STEP 3 — EVALUATE: Which arrow points toward target's area? Prefer UNEXPLORED. "
            f"+5 FOLLOW_UP beats -2 BACKTRACK. 3+ rotates → pick FORWARD arrow!\n\n"
            f"### STEP 4 — DECIDE: Choose BEST arrow number.\n\n"
            f"⚠️ STUCK 3+ steps → pick forward or behind arrow\n"
            f"⚠️ OBSTACLE → go AROUND\n"
            f"⚠️ NO closed doors/stairs\n\n"
            f"Return ONLY JSON: "
            f'{{"reasoning": "<analysis>", "action": <0-{num_actions-1}>}}'
            )
            return action_prompt

        raise ValueError('Prompt type must be stopping, pivot, no_project, or action')
