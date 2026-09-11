import gzip
import json
import logging
import math
import os
import random
import requests
import traceback
import habitat_sim

import pandas as pd
import numpy as np

from PIL import Image
from simWrapper import PolarAction, SimWrapper
from agent import *
from utils import *

class Env:
    """
    Base class for creating an environment for embodied navigation tasks.
    This class defines the setup, logging, running, and evaluation of episodes.
    """

    task = 'Not defined'

    def __init__(self, cfg: dict):
        """
        Initializes the environment with the provided configuration.

        Args:
            cfg (dict): Configuration dictionary containing environment, simulation, and agent settings.
        """
        self.cfg = cfg['env_cfg']
        self.sim_cfg = cfg['sim_cfg']
        if self.cfg['name'] == 'default':
            self.cfg['name'] = f'default_{random.randint(0, 1000)}'
        self._initialize_logging(cfg)
        self._initialize_agent(cfg)
        self.outer_run_name = self.task + '_' + self.cfg['name']
        self.inner_run_name = f'{self.cfg["instance"]}_of_{self.cfg["instances"]}'
        self.curr_run_name = "Not initialized"
        self.path_calculator = habitat_sim.MultiGoalShortestPath()
        self.simWrapper: SimWrapper = None
        self.num_episodes = 0
        self._initialize_experiment()

    def _initialize_agent(self, cfg: dict):
        """Initializes the agent for the environment."""
        PolarAction.default = PolarAction(cfg['agent_cfg']['default_action'], 0, 'default')
        cfg['agent_cfg']['sensor_cfg'] = cfg['sim_cfg']['sensor_cfg']
        agent_cls = globals()[cfg['agent_cls']]
        self.agent: Agent = agent_cls(cfg['agent_cfg']) 

    def _initialize_logging(self, cfg: dict):
        """
        Initializes logging for the environment.

        Args:
            cfg (dict): Configuration dictionary containing logging settings.
        """
        self.log_file = f'logs/{cfg["task"]}_{self.cfg["name"]}/{self.cfg["instance"]}_of_{self.cfg["instances"]}.txt'
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        if self.cfg['parallel']:
            logging.basicConfig(
                filename=self.log_file,
                level=logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s'
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s'
            )

    def _initialize_experiment(self):
        """
        Abstract method for setting up the environment and initializing all required variables.
        Should be implemented in derived classes.
        """
        raise NotImplementedError

    def run_experiment(self):
        """
        Runs the experiment by iterating over episodes.
        """
        instance_size = math.ceil(self.num_episodes / self.cfg['instances'])
        start_ndx = self.cfg['instance'] * instance_size

        for episode_ndx in range(start_ndx, min(start_ndx + self.cfg['num_episodes'], self.num_episodes)):

            self.wandb_log_data = {
                'episode_ndx': episode_ndx,
                'instance': self.inner_run_name,
                'total_episodes': self.cfg['instances'] * self.cfg['num_episodes'],
                'task': self.task,
                'task_data': {},
                'spl': 0,
                'goal_reached': False
            }

            try:
                self._run_episode(episode_ndx)
            except Exception as e:
                log_exception(e)
                self.simWrapper.reset()


    def _run_episode(self, episode_ndx: int):
        """
        Runs a single episode.

        Args:
            episode_ndx (int): The index of the episode to run.
        """
        obs = self._initialize_episode(episode_ndx)

        logging.info(f'\n===================STARTING RUN: {self.curr_run_name} ===================\n')
        for _ in range(self.cfg['max_steps']):
            try:
                agent_action = self._step_env(obs)
                if agent_action is None:
                    break
                obs = self.simWrapper.step(agent_action)

            except Exception as e:
                log_exception(e)

            finally:
                self.step += 1
        self._post_episode()

    def _initialize_episode(self, episode_ndx: int):
        """
        Initializes the episode. This method should be implemented in derived classes.

        Args:
            episode_ndx (int): The index of the episode to initialize.
        """
        self.step = 0
        self.init_pos = None
        self.df = pd.DataFrame({})
        self.agent_distance_traveled = 0
        self.prev_agent_position = None

    def _step_env(self, obs: dict):
        """
        Takes a step in the environment. This method should be implemented in derived classes.

        Args:
            obs (dict): The current observation. Contains agent state and sensor observations.

        Returns:
            PolarAction: The next action to be taken by the agent.
        """
        logging.info(f'Step {self.step}')
        agent_state = obs['agent_state']
        if self.prev_agent_position is not None:
            self.agent_distance_traveled += np.linalg.norm(agent_state.position - self.prev_agent_position)
        self.prev_agent_position = agent_state.position

        return None

    def _post_episode(self):
        """
        Called after the episode is complete, saves the dataframe log, and resets the environment.
        Sends a request to the aggregator server if parallel is set to True.
        """
        self.df.to_pickle(f'logs/{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}/df_results.pkl')
        self.simWrapper.reset()
        self.agent.reset()
        if self.cfg['parallel']:
            try:
                self.wandb_log_data['spend'] = self.agent.get_spend()
                self.wandb_log_data['default_rate'] = len(self.df[self.df['success'] == 0]) / len(self.df)
                response = requests.post(f'http://localhost:{self.cfg["port"]}/log', json=self.wandb_log_data)
                if response.status_code != 200:
                    logging.error(f"Failed to send metrics: {response.text}")
            except Exception as e:
                tb = traceback.extract_tb(e.__traceback__)
                for frame in tb:
                    logging.error(f"Frame {frame.filename} line {frame.lineno}")
                logging.error(e)

        logging.info('\n===================RUN COMPLETE===================\n')
        if self.cfg['log_freq'] == 1:
            create_gif(
                f'logs/{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}',
            )

    def _log(self, images: dict, step_metadata: dict, logging_data: dict):
        """
        Appends the step metadata to the dataframe, and saves the images and general metadata to disk.

        Args:
            images (dict): Images generated during the step.
            step_metadata (dict): Metadata for the current step.
            logging_data (dict): General logging data.
        """
        self.df = pd.concat([self.df, pd.DataFrame([step_metadata])], ignore_index=True)

        if self.step % self.cfg['log_freq'] == 0 or step_metadata['success'] == 0:
            path = f'logs/{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}/step{self.step}'
            if not step_metadata['success']:
                path += '_ERROR'
            os.makedirs(path, exist_ok=True)
            for name, im in images.items():
                im = Image.fromarray(im[:, :, 0:3], mode='RGB')
                im.save(f'{path}/{name}.png')
            with open(f'{path}/details.txt', 'w') as file:
                # Always log key metrics regardless of success/failure
                for key in ['finish_status', 'goal_reached', 'distance_to_goal', 'spl', 'action_number']:
                    if key in step_metadata:
                        file.write(f'{key.upper()}\n{step_metadata[key]}\n\n')
                
                if step_metadata['success']:
                    for k, v in logging_data.items():
                        file.write(f'{k}\n{v}\n\n')
                else:
                    # 记录错误步骤的详细信息，帮助调试
                    file.write("=== ERROR STEP DETAILS ===\n")
                    for k, v in logging_data.items():
                        file.write(f'{k}\n{v}\n\n')
                    if 'error' in step_metadata:
                        file.write(f"ERROR_REASON\n{step_metadata['error']}\n\n")
                    file.write(f"SUCCESS_STATUS\n{step_metadata['success']}\n")

    def _calculate_metrics(self, agent_state: habitat_sim.AgentState, agent_action: PolarAction, geodesic_path: int, max_steps: int):
        """
        Calculates the navigation metrics at a given step.

        Args:
            agent_state: The state of the agent.
            agent_action: The action taken by the agent.
            geodesic_path: The shortest path to the goal.
            max_steps (int): Maximum steps allowed for the episode.

        Returns:
            dict: A dictionary containing calculated metrics.
        """
        import numpy as np
        
        metrics = {}
        self.path_calculator.requested_start = agent_state.position
        
        # Try to get geodesic distance first
        geo_distance = self.simWrapper.get_path(self.path_calculator)
        
        # If path finding fails (returns 1000), estimate distance based on movement
        if geo_distance >= 999:
            logging.info(f"Path finding failed (distance={geo_distance}), using estimation...")
            
            # Method 1: Use initial euclidean distance minus distance traveled
            if hasattr(self, 'current_episode_initial_distance') and self.current_episode_initial_distance is not None:
                # Estimate: initial_distance - distance_traveled (simplified assumption)
                estimated_distance = max(0, self.current_episode_initial_distance - self.agent_distance_traveled)
                geo_distance = estimated_distance
                logging.info(f"✅ Estimated distance using movement: {geo_distance:.2f}m (initial={self.current_episode_initial_distance:.2f}m, traveled={self.agent_distance_traveled:.2f}m)")
            else:
                logging.warning("⚠️ No initial distance available for estimation")
        
        metrics['distance_to_goal'] = geo_distance
        metrics['spl'] = 0
        metrics['goal_reached'] = False
        metrics['done'] = False
        metrics['finish_status'] = 'running'

        if agent_action is PolarAction.stop or self.step + 1 == max_steps:
            metrics['done'] = True

            if metrics['distance_to_goal'] < self.cfg['success_threshold']:
                metrics['finish_status'] = 'success'
                metrics['goal_reached'] = True
                metrics['spl'] = geodesic_path / max(geodesic_path, self.agent_distance_traveled)
                self.wandb_log_data.update({
                    'spl': metrics['spl'],
                    'goal_reached': metrics['goal_reached']
                })
            else:
                if agent_action is PolarAction.stop:
                    metrics['finish_status'] = 'fp'
                else:
                    metrics['finish_status'] = 'max_steps'

        return metrics

class GOATEnv(Env):
    """
    Environment for the GOAT task, extending the base Env class.
    This class defines the setup, initialization, and running of GOAT episodes.
    """

    task = 'GOAT'

    def _initialize_experiment(self):
        """
        Initializes the experiment by setting up the dataset split, scene configuration, and goals.
        """
        self.split = 'val' if 'val' in self.cfg['split'] else 'train'
        self.sim_cfg['scene_config'] = "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"

        self.all_episodes = []
        self.goals = {}
        for f in sorted(os.listdir(f'data/datasets/goat_bench/hm3d/v1/{self.cfg["split"]}/content')):
            with gzip.open(f'data/datasets/goat_bench/hm3d/v1/{self.cfg["split"]}/content/{f}', 'rt') as gz:
                js = json.load(gz)
                hsh = f.split('.')[0]
                self.goals[hsh] = js['goals']
                self.all_episodes += js['episodes']
        self.num_episodes = len(self.all_episodes)

    def _initialize_episode(self, episode_ndx: int):
        """
        Initializes the episode for the GOAT task.

        Args:
            episode_ndx (int): The index of the episode to initialize.
        """
        super()._initialize_episode(episode_ndx)

        episode = self.all_episodes[episode_ndx]
        f, glb = episode['scene_id'].split('/')[-2:]
        hsh = f[6:]
        goals = self.goals[hsh]
        self.sim_cfg['scene_id'] = f[2:5]
        self.sim_cfg['scene_path'] = f'data/scene_datasets/hm3d/{self.split}/{f}/{glb}'
        self.simWrapper = SimWrapper(self.sim_cfg)
        self.current_episode = []

        for goal in episode['tasks']:
            name = goal[0]
            mode = goal[1]
            subgoal = {'name': name, 'mode': mode, 'id': goal[2], 'view_points': []}
            for obj in goals[f'{f[6:]}.basis.glb_{name}']:
                if mode == 'object':
                    subgoal['view_points'] += [a['agent_state']['position'] for a in obj['view_points']]
                else:
                    if obj['object_id'] == goal[2]:
                        subgoal['view_points'] = [a['agent_state']['position'] for a in obj['view_points']]
                        if mode == 'description':
                            subgoal['lang_desc'] = obj['lang_desc']
                        if mode == 'image':
                            image_ndx = goal[3]
                            subgoal['image_position'] = obj['image_goals'][image_ndx]['position']
                            subgoal['image_rotation'] = obj['image_goals'][image_ndx]['rotation']

            self.current_episode.append(subgoal)

        logging.info(f'\nRUNNING EPISODE {episode_ndx}, SCENE: {self.simWrapper.scene_id}')
        for i, subgoal in enumerate(self.current_episode):
            logging.info(f'Goal {i}: {subgoal["name"]}, {subgoal["mode"]}')

        self.init_pos = np.array(episode['start_position'])
        self.simWrapper.set_state(pos=self.init_pos, quat=episode['start_rotation'])
        self.curr_goal_ndx = 0
        self.curr_run_name = f"{episode_ndx}_{self.simWrapper.scene_id}"
        self.last_goal_reset = -1
        self.path_calculator.requested_ends = np.array(self.current_episode[self.curr_goal_ndx]['view_points'], dtype=np.float32)
        self.path_calculator.requested_start = self.init_pos
        self.curr_shortest_path = self.simWrapper.get_path(self.path_calculator)

        obs = self.simWrapper.step(PolarAction.null)
        return obs

    def _step_env(self, obs: dict):
        """
        Takes a step in the environment for the GOAT task.

        Args:
            obs (dict): The current observation.

        Returns:
            list: The next action to be taken by the agent.
        """
        super()._step_env(obs)

        goal = self.current_episode[self.curr_goal_ndx]
        obs['goal'] = goal
        if goal['mode'] == 'image':
            position = goal['image_position']
            rotation = goal['image_rotation']
            goal_im = self.simWrapper.get_goal_image(position, rotation)
            put_text_on_image(goal_im, f"GOAL IMAGE: {goal['name']}", bg_color=(255, 255, 255), location='top_center')
            obs['goal']['goal_image'] = goal_im

        agent_state = obs['agent_state']
        agent_action, metadata = self.agent.step(obs)
        step_metadata = metadata['step_metadata']
        logging_data = metadata['logging_data']
        images = metadata['images']

        metrics = self._calculate_metrics(agent_state, agent_action, self.curr_shortest_path, self.last_goal_reset + 1 + self.cfg['max_steps_per_subgoal'])
        step_metadata.update(metrics)

        if metrics['done']:
            self.wandb_log_data['task_data'].setdefault('goal_data', []).append({
                'goal_mode': goal['mode'],
                'goal_reached': metrics['goal_reached'],
                'spl': metrics['spl'],
            })

            if 'spl' in self.wandb_log_data:
                del self.wandb_log_data['spl']
            if 'goal_reached' in self.wandb_log_data:
                del self.wandb_log_data['goal_reached']

            if self.curr_goal_ndx + 1 == len(self.current_episode):
                agent_action = None
            else:
                self.agent.reset_goal()
                self.agent_distance_traveled = 0
                agent_action = PolarAction.null
                self.curr_goal_ndx += 1
                self.last_goal_reset = self.step
                goal = self.current_episode[self.curr_goal_ndx]
                self.path_calculator.requested_ends = np.array(goal['view_points'], dtype=np.float32)
                self.path_calculator.requested_start = obs['agent_state'].position
                self.curr_shortest_path = self.simWrapper.get_path(self.path_calculator)

                logging.info(f'New goal {goal["mode"]}: {goal["name"]}, GEODESIC: {self.curr_shortest_path}')

        self._log(images, step_metadata, logging_data)

        return agent_action

class ObjectNavEnv(Env):
    """
    Environment for the ObjectNav task using real HM3D dataset.
    """
    
    task = 'ObjectNav'

    def _initialize_experiment(self):
        """
        Initializes the experiment with real ObjectNav dataset.
        Loads episodes from val_mini or val split.
        """
        self.all_episodes = []
        
        # Determine which split to use
        self.split = self.cfg.get('split', 'val_mini')  # Default to val_mini for quick testing
        
        # Optional: filter by object category (e.g., 'sofa', 'chair', 'bed')
        target_category = self.cfg.get('target_category', None)
        
        # Optional: start from specific episode index
        start_episode_idx = self.cfg.get('start_episode_idx', 0)
        
        # Dataset paths
        dataset_base = "data/datasets/objectnav_hm3d_v2"
        self.scene_base = "data/scene_datasets/hm3d/minival"
        
        # Load episodes from content directory
        content_dir = os.path.join(dataset_base, self.split, "content")
        
        if not os.path.exists(content_dir):
            raise FileNotFoundError(f"Dataset content directory not found: {content_dir}")
        
        import gzip
        import json
        
        for scene_file in sorted(os.listdir(content_dir)):
            if scene_file.endswith('.json.gz'):
                scene_path_full = os.path.join(content_dir, scene_file)
                try:
                    with gzip.open(scene_path_full, 'rt') as f:
                        scene_data = json.load(f)
                        episodes = scene_data.get('episodes', [])
                        
                        # Filter by target category if specified
                        if target_category:
                            episodes = [ep for ep in episodes if ep['object_category'] == target_category]
                            if episodes:
                                logging.info(f"Loaded {len(episodes)} '{target_category}' episodes from {scene_file}")
                        else:
                            self.all_episodes.extend(episodes)
                            logging.info(f"Loaded {len(episodes)} episodes from {scene_file}")
                        
                        self.all_episodes.extend(episodes)
                except Exception as e:
                    logging.warning(f"Failed to load {scene_file}: {e}")
        
        # Apply start_episode_idx filter
        if start_episode_idx > 0:
            if start_episode_idx >= len(self.all_episodes):
                raise ValueError(f"start_episode_idx ({start_episode_idx}) is out of range (total episodes: {len(self.all_episodes)})")
            self.all_episodes = self.all_episodes[start_episode_idx:]
            logging.info(f"Skipped first {start_episode_idx} episodes, starting from index {start_episode_idx}")
        
        self.num_episodes = len(self.all_episodes)
        
        if self.num_episodes == 0:
            if target_category:
                raise ValueError(f"No episodes found for category '{target_category}' in {content_dir}")
            else:
                raise ValueError(f"No episodes loaded from {content_dir}")
        
        if target_category:
            logging.info(f'Initialized ObjectNav with {self.num_episodes} "{target_category}" episodes from {self.split}')
        else:
            logging.info(f'Initialized ObjectNav with {self.num_episodes} episodes from {self.split}')
        logging.info(f'Scene base: {self.scene_base}')

    def _initialize_episode(self, episode_ndx: int):
        """
        Initializes a real ObjectNav episode.

        Args:
            episode_ndx (int): The index of the episode to initialize.
        """
        super()._initialize_episode(episode_ndx)
        episode = self.all_episodes[episode_ndx]
        
        # Parse scene_id to get actual file path
        # Episode scene_id format: "hm3d_v0.2/minival/00802-wcojb4TFT35/wcojb4TFT35.basis.glb"
        scene_id_full = episode['scene_id']
        parts = scene_id_full.split('/')
        
        # Extract scene directory and filename
        # Expected: ['hm3d_v0.2', 'minival', '00802-wcojb4TFT35', 'wcojb4TFT35.basis.glb']
        if len(parts) >= 4:
            scene_dir_name = parts[2]  # e.g., "00802-wcojb4TFT35"
            scene_file_name = parts[3]  # e.g., "wcojb4TFT35.basis.glb"
        else:
            # Fallback for different format
            scene_dir_name = os.path.basename(os.path.dirname(scene_id_full))
            scene_file_name = os.path.basename(scene_id_full)
        
        # Construct full scene path
        scene_path = os.path.join(self.scene_base, scene_dir_name, scene_file_name)
        
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"Scene file not found: {scene_path}")
        
        # Update simulator config - MUST set scene_dataset_config_file for semantic scene access
        scene_dataset_config = "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
        
        self.sim_cfg['scene_config'] = scene_dataset_config
        self.sim_cfg['scene_id'] = scene_dir_name
        self.sim_cfg['scene_path'] = scene_path
        
        logging.info(f"Loading scene with dataset config: {scene_dataset_config}")
        
        # Initialize simulator
        self.simWrapper = SimWrapper(self.sim_cfg)
        
        # Set agent state
        self.init_pos = np.array(episode['start_position'])
        start_rot = episode['start_rotation']
        self.simWrapper.set_state(pos=self.init_pos, quat=start_rot)
        
        # Prepare current episode info
        object_category = episode['object_category']
        display_category = object_category.replace('_', ' ')
        if object_category == 'tv_monitor':
            display_category = 'tv screen'
        
        # Get goal positions - Note: val_mini dataset doesn't provide exact positions
        # We'll use euclidean_distance from info as initial reference
        object_positions = []
        initial_euclidean_dist = episode.get('info', {}).get('euclidean_distance', None)
        closest_obj_id = episode.get('info', {}).get('closest_goal_object_id', None)
        
        logging.info(f"Episode info: euclidean_distance={initial_euclidean_dist}, closest_goal_object_id={closest_obj_id}")
        
        if initial_euclidean_dist is not None:
            # Store initial distance for later estimation
            self.current_episode_initial_distance = initial_euclidean_dist
            logging.info(f"Initial Euclidean distance to target: {initial_euclidean_dist:.2f}m")
        else:
            self.current_episode_initial_distance = None
            logging.warning("⚠️ No euclidean_distance in episode info")
        
        self.current_episode = {
            'object': display_category,
            'shortest_path': episode['info']['geodesic_distance'],
            'object_positions': object_positions,
            'view_positions': [],
            'initial_euclidean_distance': initial_euclidean_dist
        }
        
        self.curr_run_name = f"{episode_ndx}_{scene_dir_name}"
        
        logging.info(f'RUNNING EPISODE {episode_ndx} - Target: {object_category}, Scene: {scene_dir_name}')
        logging.info(f'Geodesic distance: {episode["info"]["geodesic_distance"]:.2f}m')
        
        # Get initial observation
        obs = self.simWrapper.step(PolarAction.null)
        return obs

    def _step_env(self, obs: dict):
        """
        Takes a step in the environment for the ObjectNav task.

        Args:
            obs (dict): The current observation.

        Returns:
            list: The next action to be taken by the agent.
        """
        super()._step_env(obs)
        obs['goal'] = self.current_episode['object']
        agent_state = obs['agent_state']
        self.agent_distance_traveled += np.linalg.norm(agent_state.position - self.prev_agent_position)
        self.prev_agent_position = agent_state.position
        agent_action, metadata = self.agent.step(obs)
        step_metadata = metadata['step_metadata']
        logging_data = metadata['logging_data']
        images = metadata['images']

        metrics = self._calculate_metrics(agent_state, agent_action, self.current_episode['shortest_path'], self.cfg['max_steps'])
        step_metadata.update(metrics)

        self._log(images, step_metadata, logging_data)

        if metrics['done']:
            agent_action = None

        return agent_action

class QuickTestEnv(Env):
    """
    Simplified environment for quick testing using Habitat built-in scenes.
    This class defines the setup, initialization, and running of test episodes.
    """

    task = 'QuickTest'

    def _initialize_experiment(self):
        """
        Initializes the experiment with a simple test scene.
        """
        self.all_episodes = []
        
        # Use habitat-sim built-in test scene with proper dataset config
        scene_path = "/home/tao_h/VLMnav/habitat-sim/data/test_assets/dataset_tests/dataset_0/stages/dataset_test_stage.glb"
        config_file = "/home/tao_h/VLMnav/habitat-sim/data/test_assets/dataset_tests/dataset_0/test_dataset_0.scene_dataset_config.json"
        
        self.sim_cfg['scene_config'] = config_file
        self.sim_cfg['scene_id'] = scene_path
        self.sim_cfg['scene_path'] = scene_path
        
        # Create a simple episode with a random object category
        self.num_episodes = self.cfg.get('num_episodes', 1)
        
        logging.info(f'Initialized QuickTest with {self.num_episodes} episodes')
        logging.info(f'Scene: {scene_path}')
        logging.info(f'Dataset Config: {config_file}')

    def _initialize_episode(self, episode_ndx: int):
        """
        Initializes the episode for the QuickTest task.

        Args:
            episode_ndx (int): The index of the episode to initialize.
        """
        super()._initialize_episode(episode_ndx)
        
        # Initialize simulator with test scene
        self.simWrapper = SimWrapper(self.sim_cfg)
        
        # Set random starting position
        self.simWrapper.reset()
        
        obs = self.simWrapper.step(PolarAction.null)
        self.curr_run_name = f"episode_{episode_ndx}"
        
        logging.info(f'RUNNING EPISODE {episode_ndx}')
        return obs

    def _step_env(self, obs: dict):
        """
        Takes a step in the environment for the QuickTest task.

        Args:
            obs (dict): The current observation.

        Returns:
            list: The next action to be taken by the agent.
        """
        super()._step_env(obs)
        
        # Simple navigation goal: explore the room
        obs['goal'] = "explore the room"
        
        agent_state = obs['agent_state']
        if self.prev_agent_position is not None:
            self.agent_distance_traveled += np.linalg.norm(agent_state.position - self.prev_agent_position)
        self.prev_agent_position = agent_state.position
        
        agent_action, metadata = self.agent.step(obs)
        step_metadata = metadata['step_metadata']
        logging_data = metadata['logging_data']
        images = metadata['images']

        # Basic metrics
        metrics = {
            'done': False,
            'success': False,
            'spl': 0.0
        }
        step_metadata.update(metrics)

        self._log(images, step_metadata, logging_data)

        # Stop after max steps
        if self.step >= self.cfg['max_steps']:
            agent_action = None
            logging.info(f'Episode finished after {self.step} steps')

        return agent_action

    def _post_episode(self):
        """
        Post-episode processing for QuickTest.
        """
        logging.info(f'Episode completed. Steps: {self.step}, Distance traveled: {self.agent_distance_traveled:.2f}m')
