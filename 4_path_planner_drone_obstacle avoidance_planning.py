#!/usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import os
import sys # For sys.exit
import matplotlib
matplotlib.use('Agg')

# --- Simulation and RL Model Imports ---
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from DroneMPEnv import DroneMPEnv # Ensure this is your custom environment class
    from gym_pybullet_drones.utils.enums import DroneModel # Assuming you use this from the library
except ImportError as e:
    print(f"Import Error: {e}. Please ensure stable-baselines3, gym-pybullet-drones, and DroneMPEnv.py are correctly installed and accessible.")
    sys.exit(1)


# --- Global Configuration for PPO Models and Execution ---
RESULTS_BASE_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_results")
PPO_MODELS_BASE_DIR = os.environ.get("PPO_MODELS_PATH", RESULTS_BASE_DIR_DEFAULT)
ALGO = "PPO"
ENV_CONFIG_NAME = "model_0p1_pid_rl" # Or your specific config
CTRL_FREQ = 240 # Control frequency in Hz

class StandardDummyGP:
    def __init__(self, t_min=0.1, t_max=1.0, typical_dist_step=0.5):
        self.t_min = t_min
        self.t_max = t_max
        self.typical_dist_step = typical_dist_step if typical_dist_step > 1e-3 else 0.5
    def predict(self, X, return_std=False):
        displacement = X[0][0]
        assumed_speed = self.typical_dist_step / ((self.t_min + self.t_max) / 2.0) if (self.t_min + self.t_max) > 0 else 1.0
        time_taken = displacement / assumed_speed if assumed_speed > 1e-3 else self.t_min
        time_taken = np.clip(time_taken, self.t_min, self.t_max)
        return (np.array([time_taken]), np.array([0.1])) if return_std else np.array([time_taken])

class Node:
    def __init__(self, x, y, z, t):
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        self.cost = 0.0
        self.parent = None
        self.primitive_from_parent = None
        self.dt_from_parent = 0.0

class Obstacle:
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max, t_start, t_end):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.t_start = t_start
        self.t_end = t_end

    def collides(self, node):
        # Check if the node's spatial coordinates are within the obstacle's bounds
        spatial_collision = (self.x_min <= node.x <= self.x_max) and \
                            (self.y_min <= node.y <= self.y_max) and \
                            (self.z_min <= node.z <= self.z_max)
        # Check if the node's time is within the obstacle's active time window
        temporal_collision = (self.t_start <= node.t <= self.t_end)
        return spatial_collision and temporal_collision

def euclidean_distance_3d_points(p1_xyz, p2_xyz):
    return math.sqrt((p1_xyz[0] - p2_xyz[0])**2 + (p1_xyz[1] - p2_xyz[1])**2 + (p1_xyz[2] - p2_xyz[2])**2)

def euclidean_distance_3d_nodes(node1, node2):
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2 + (node1.z - node2.z)**2)

def steer(from_node, to_node_target_coords, min_dist, max_dist):
    dist = euclidean_distance_3d_points((from_node.x, from_node.y, from_node.z), to_node_target_coords)

    if dist == 0:
        ratio = 0.0
    elif dist > max_dist:
        ratio = max_dist / dist
    else:
        ratio = 1.0

    new_x = from_node.x + (to_node_target_coords[0] - from_node.x) * ratio
    new_y = from_node.y + (to_node_target_coords[1] - from_node.y) * ratio
    new_z = from_node.z + (to_node_target_coords[2] - from_node.z) * ratio
    return (new_x, new_y, new_z)

def custom_cost_function(from_node, to_node, dt=0.0):
    distance_cost = euclidean_distance_3d_nodes(from_node, to_node)
    time_cost = dt * 1.0  # Weight for time, can be tuned
    return distance_cost + time_cost

class RRTStar:
    PRIMITIVE_DEFINITIONS = {
        "Forward":  ([1,0,0], "GP_X_Pos_Translate.pkl"),
        "Backward": ([-1,0,0], "GP_X_Neg_Translate.pkl"),
        "Left":     ([0,1,0], "GP_Y_Pos_Translate.pkl"),
        "Right":    ([0,-1,0], "GP_Y_Neg_Translate.pkl"),
        "Up":       ([0,0,1], "GP_Z_Pos_Translate.pkl"),
        "Down":     ([0,0,-1], "GP_Z_Neg_Translate.pkl"),
    }

    PRIMITIVE_STEP_CALIBRATION_FACTORS = {
        "Forward": 2.,
        "Backward": 2.5,
        "Left": 2.2,
        "Right": 2.2,
        "Up": 6.2,
        "Down": 3.7,
        "Hover": 1.0
    }

    def __init__(self, start, goal, x_bounds, y_bounds, z_bounds, t_bounds, obstacles,
                 min_dist, max_dist, t_min_primitive, t_max_primitive,
                 ctrl_freq=CTRL_FREQ, max_iter=20000,
                 gp_model_dir_override=None,
                 ppo_model_base_dir_override=None,
                 checkpoint_iter_override=None,
                 cost_time_weight=1.0,
                 cost_remaining_dist_weight=1.0,
                 dominant_axis_adherence_factor=0.5
                 ):

        self.start_node_def = Node(*start)
        self.goal_node_def = Node(*goal)
        self.min_dist_primitive = min_dist
        self.max_dist_primitive = max_dist
        self.t_min_clip = t_min_primitive
        self.t_max_clip = t_max_primitive

        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds
        self.t_bounds = t_bounds
        self.obstacles = obstacles
        self.max_iter = max_iter
        self.nodes = [self.start_node_def]

        self.CTRL_FREQ = ctrl_freq
        self.COST_TIME_WEIGHT = cost_time_weight
        self.COST_REMAINING_DIST_WEIGHT = cost_remaining_dist_weight
        self.DOMINANT_AXIS_ADHERENCE_FACTOR = dominant_axis_adherence_factor

        gp_base_dir = gp_model_dir_override if gp_model_dir_override else os.path.join(os.path.dirname(os.path.abspath(__file__)), "Ideal", "GP_models_3D")
        os.makedirs(gp_base_dir, exist_ok=True)
        self.GP_MODELS = {}

        print(f"DEBUG: RRTStar using PRIMITIVE_DEFINITIONS: {list(RRTStar.PRIMITIVE_DEFINITIONS.keys())}")

        for prim_name, (_, fname) in RRTStar.PRIMITIVE_DEFINITIONS.items():
            gp_model_path = os.path.join(gp_base_dir, fname)
            try:
                with open(gp_model_path, "rb") as f:
                    self.GP_MODELS[prim_name] = pickle.load(f)
                    print(f"✅ Successfully loaded ACTUAL GP Model for {prim_name} from {gp_model_path}")
            except FileNotFoundError:
                print(f"⚠️ GP Model for {prim_name} ({fname}) not found in {gp_base_dir}. Using DUMMY GP.")
                self.GP_MODELS[prim_name] = self._create_dummy_gp(prim_name)
            except AttributeError as e:
                print(f"Pickle AttributeError for {prim_name} ({fname}): {e}. Using DUMMY GP.")
                self.GP_MODELS[prim_name] = self._create_dummy_gp(prim_name)
            except Exception as e:
                print(f"Error loading GP Model for {prim_name} ({fname}): {type(e).__name__} - {e}. Using DUMMY GP.")
                self.GP_MODELS[prim_name] = self._create_dummy_gp(prim_name)

        for prim_name_def in RRTStar.PRIMITIVE_DEFINITIONS.keys():
            if prim_name_def not in self.GP_MODELS:
                 print(f"⚠️ Critical: Primitive {prim_name_def} from PRIMITIVE_DEFINITIONS has no GP model. Assigning DUMMY GP.")
                 self.GP_MODELS[prim_name_def] = self._create_dummy_gp(prim_name_def)

        print(f"✅ GP Models loading complete. Check messages above for dummy fallbacks.")
        self.PPO_MODELS_BASE_DIR = ppo_model_base_dir_override if ppo_model_base_dir_override else PPO_MODELS_BASE_DIR
        self.CHECKPOINT_ITERATION = checkpoint_iter_override

    def _create_dummy_gp(self, prim_name=""):
        return StandardDummyGP(self.t_min_clip, self.t_max_clip, self.max_dist_primitive)

    def sample(self):
        if np.random.rand() < 0.1: # 10% chance to sample goal
            return (self.goal_node_def.x, self.goal_node_def.y, self.goal_node_def.z)
        return (np.random.uniform(*self.x_bounds),
                np.random.uniform(*self.y_bounds),
                np.random.uniform(*self.z_bounds))

    def nearest_node(self, sample_xyz):
        return min(self.nodes, key=lambda node: euclidean_distance_3d_points((node.x,node.y,node.z), sample_xyz))

    def is_collision_free(self, node_to_check):
        if not (self.t_bounds[0] <= node_to_check.t <= self.t_bounds[1]): return False
        if not (self.x_bounds[0] <= node_to_check.x <= self.x_bounds[1] and \
                self.y_bounds[0] <= node_to_check.y <= self.y_bounds[1] and \
                self.z_bounds[0] <= node_to_check.z <= self.z_bounds[1]): return False
        return all(not obs.collides(node_to_check) for obs in self.obstacles)

    def _predict_primitive_and_time(self, parent_node_xyz, target_node_xyz):
        dx = target_node_xyz[0] - parent_node_xyz[0]
        dy = target_node_xyz[1] - parent_node_xyz[1]
        dz = target_node_xyz[2] - parent_node_xyz[2]

        target_disp_vec = np.array([dx, dy, dz])
        target_dist_mag = np.linalg.norm(target_disp_vec)

        if target_dist_mag < 1e-4:
            return "Hover", 0.0

        best_primitive_name = None
        min_heuristic_cost = float('inf')
        best_predicted_dt = self.t_max_clip

        abs_displacements = np.abs(target_disp_vec)
        dominant_axis_idx = np.argmax(abs_displacements)

        STRONGLY_AXIAL_THRESHOLD_FACTOR = 3.0
        is_strongly_axial_segment = abs_displacements[dominant_axis_idx] > \
                                   STRONGLY_AXIAL_THRESHOLD_FACTOR * \
                                   (sum(np.delete(abs_displacements, dominant_axis_idx)) + 1e-6)

        primitives_to_consider_loop = []
        if is_strongly_axial_segment:
            if dominant_axis_idx == 0:
                primitives_to_consider_loop = ["Forward", "Backward"]
            elif dominant_axis_idx == 1:
                primitives_to_consider_loop = ["Left", "Right"]
            elif dominant_axis_idx == 2:
                primitives_to_consider_loop = ["Up", "Down"]
            primitives_to_consider_loop = [p for p in primitives_to_consider_loop if p in RRTStar.PRIMITIVE_DEFINITIONS]
            if not primitives_to_consider_loop:
                 primitives_to_consider_loop = list(RRTStar.PRIMITIVE_DEFINITIONS.keys())
        else:
            primitives_to_consider_loop = list(RRTStar.PRIMITIVE_DEFINITIONS.keys())

        for prim_name in primitives_to_consider_loop:
            char_dir_vec_np, _ = RRTStar.PRIMITIVE_DEFINITIONS[prim_name]
            char_dir_vec = np.array(char_dir_vec_np)

            if prim_name not in self.GP_MODELS: continue
            gp_model = self.GP_MODELS[prim_name]

            effective_displacement_for_gp = 0
            if prim_name == "Up" or prim_name == "Down": effective_displacement_for_gp = abs(dz)
            elif prim_name == "Forward" or prim_name == "Backward": effective_displacement_for_gp = abs(dx)
            elif prim_name == "Left" or prim_name == "Right": effective_displacement_for_gp = abs(dy)
            else: continue

            if effective_displacement_for_gp < 1e-4:
                 predicted_dt_for_prim = self.t_min_clip
            else:
                try:
                    mean_dt, _ = gp_model.predict(np.array([[effective_displacement_for_gp]]), return_std=True)
                    predicted_dt_for_prim = np.clip(mean_dt.item(), self.t_min_clip, self.t_max_clip)
                except Exception:
                    predicted_dt_for_prim = self.t_max_clip

            alignment_factor = 0.0
            if target_dist_mag > 1e-4:
                alignment_factor = np.dot(target_disp_vec / target_dist_mag, char_dir_vec)

            current_heuristic_cost = self.COST_TIME_WEIGHT * predicted_dt_for_prim + \
                                     self.COST_REMAINING_DIST_WEIGHT * (target_dist_mag * (1.0 - max(0, alignment_factor)))

            if is_strongly_axial_segment:
                if char_dir_vec[dominant_axis_idx] * target_disp_vec[dominant_axis_idx] < 0:
                    current_heuristic_cost += self.DOMINANT_AXIS_ADHERENCE_FACTOR * target_dist_mag * 10.0
            else:
                off_axis_displacement_sum = 0
                if prim_name in ["Up", "Down"] and dominant_axis_idx != 2: off_axis_displacement_sum = abs(dx) + abs(dy)
                elif prim_name in ["Left", "Right"] and dominant_axis_idx != 1: off_axis_displacement_sum = abs(dx) + abs(dz)
                elif prim_name in ["Forward", "Backward"] and dominant_axis_idx != 0: off_axis_displacement_sum = abs(dy) + abs(dz)
                current_heuristic_cost += self.DOMINANT_AXIS_ADHERENCE_FACTOR * off_axis_displacement_sum * 2.0

            if current_heuristic_cost < min_heuristic_cost:
                min_heuristic_cost = current_heuristic_cost
                best_primitive_name = prim_name
                best_predicted_dt = predicted_dt_for_prim

        if best_primitive_name is None :
            if abs_displacements[0] >= abs_displacements[1] and abs_displacements[0] >= abs_displacements[2]:
                best_primitive_name = "Forward" if dx >= 0 else "Backward"
            elif abs_displacements[1] >= abs_displacements[0] and abs_displacements[1] >= abs_displacements[2]:
                best_primitive_name = "Left" if dy >= 0 else "Right"
            else:
                best_primitive_name = "Up" if dz >= 0 else "Down"

            if best_primitive_name in self.GP_MODELS:
                gp_model = self.GP_MODELS[best_primitive_name]
                eff_disp_fallback = 0
                if best_primitive_name in ["Up", "Down"]: eff_disp_fallback = abs(dz)
                elif best_primitive_name in ["Forward", "Backward"]: eff_disp_fallback = abs(dx)
                elif best_primitive_name in ["Left", "Right"]: eff_disp_fallback = abs(dy)

                if eff_disp_fallback < 1e-4: best_predicted_dt = self.t_min_clip
                else:
                    try:
                        mean_dt, _ = gp_model.predict(np.array([[eff_disp_fallback]]), return_std=True)
                        best_predicted_dt = np.clip(mean_dt.item(), self.t_min_clip, self.t_max_clip)
                    except Exception:
                         best_predicted_dt = self.t_max_clip
            else:
                best_predicted_dt = self.t_max_clip

        return best_primitive_name, best_predicted_dt

    def find_path(self):
        for i in range(self.max_iter):
            sampled_xyz_tuple = self.sample()
            nearest_node_obj = self.nearest_node(sampled_xyz_tuple)
            steered_xyz_tuple = steer(nearest_node_obj, sampled_xyz_tuple, self.min_dist_primitive, self.max_dist_primitive)

            dx_steer = steered_xyz_tuple[0] - nearest_node_obj.x
            dy_steer = steered_xyz_tuple[1] - nearest_node_obj.y
            dz_steer = steered_xyz_tuple[2] - nearest_node_obj.z

            if abs(dx_steer) < 1e-4 and abs(dy_steer) < 1e-4 and abs(dz_steer) < 1e-4:
                continue

            primitive_name, dt = self._predict_primitive_and_time(
                (nearest_node_obj.x, nearest_node_obj.y, nearest_node_obj.z),
                steered_xyz_tuple
            )

            if primitive_name is None or dt is None: continue

            new_node_potential_t = nearest_node_obj.t + dt
            new_node = Node(steered_xyz_tuple[0], steered_xyz_tuple[1], steered_xyz_tuple[2], new_node_potential_t)
            new_node.primitive_from_parent = primitive_name
            new_node.dt_from_parent = dt

            if self.is_collision_free(new_node):
                min_cost_parent = nearest_node_obj
                cost_through_nearest = nearest_node_obj.cost + custom_cost_function(nearest_node_obj, new_node, dt)

                new_node.parent = min_cost_parent
                new_node.cost = cost_through_nearest

                for potential_parent_candidate in self.nodes:
                    dist_cand_to_new_spatial = euclidean_distance_3d_points(
                        (potential_parent_candidate.x, potential_parent_candidate.y, potential_parent_candidate.z),
                        (new_node.x, new_node.y, new_node.z)
                    )

                    if dist_cand_to_new_spatial <= self.max_dist_primitive and dist_cand_to_new_spatial > 1e-4:
                        p_primitive_name, p_dt = self._predict_primitive_and_time(
                            (potential_parent_candidate.x, potential_parent_candidate.y, potential_parent_candidate.z),
                            (new_node.x, new_node.y, new_node.z)
                        )

                        if p_primitive_name is None or p_dt is None: continue

                        potential_t_via_candidate = potential_parent_candidate.t + p_dt
                        cost_via_candidate = potential_parent_candidate.cost + custom_cost_function(potential_parent_candidate, new_node, p_dt)

                        temp_node_for_check = Node(new_node.x, new_node.y, new_node.z, potential_t_via_candidate)

                        if cost_via_candidate < new_node.cost and self.is_collision_free(temp_node_for_check):
                            new_node.parent = potential_parent_candidate
                            new_node.cost = cost_via_candidate
                            new_node.t = potential_t_via_candidate
                            new_node.primitive_from_parent = p_primitive_name
                            new_node.dt_from_parent = p_dt

                if not self.is_collision_free(new_node):
                    continue

                self.nodes.append(new_node)
                self._rewire(new_node)

                dist_to_goal_spatial = euclidean_distance_3d_points(
                    (new_node.x, new_node.y, new_node.z),
                    (self.goal_node_def.x, self.goal_node_def.y, self.goal_node_def.z)
                )

                if dist_to_goal_spatial < self.max_dist_primitive:
                    g_primitive, g_dt = self._predict_primitive_and_time(
                        (new_node.x, new_node.y, new_node.z),
                        (self.goal_node_def.x, self.goal_node_def.y, self.goal_node_def.z)
                    )

                    if g_primitive is not None and g_dt is not None:
                        goal_final_t = new_node.t + g_dt
                        final_goal_state_node = Node(self.goal_node_def.x, self.goal_node_def.y, self.goal_node_def.z, goal_final_t)
                        final_goal_state_node.primitive_from_parent = g_primitive
                        final_goal_state_node.dt_from_parent = g_dt

                        if self.is_collision_free(final_goal_state_node):
                            final_goal_state_node.parent = new_node
                            final_goal_state_node.cost = new_node.cost + custom_cost_function(new_node, final_goal_state_node, g_dt)
                            self.nodes.append(final_goal_state_node)
                            print(f"✅ Goal connected at iter {i}. Path cost: {final_goal_state_node.cost:.2f}, Path time: {final_goal_state_node.t:.2f}")
                            return self.extract_path_for_execution(final_goal_state_node)

            if i % 1000 == 0: print(f"[Iter {i}/{self.max_iter}] Nodes: {len(self.nodes)}")

        print(f"❌ No valid path to goal found after {self.max_iter} iterations.")
        closest_node_to_goal = min(self.nodes, key=lambda node: euclidean_distance_3d_nodes(node, self.goal_node_def))
        dist = euclidean_distance_3d_nodes(closest_node_to_goal, self.goal_node_def)
        print(f"ℹ️ Closest node to goal found at spatial distance: {dist:.2f}")
        if dist < self.max_dist_primitive * 2 :
             return self.extract_path_for_execution(closest_node_to_goal)
        return []


    def _rewire(self, new_node_obj):
        for existing_node in self.nodes:
            if existing_node == new_node_obj.parent or existing_node == new_node_obj:
                continue

            dist_new_to_existing_spatial = euclidean_distance_3d_points(
                (new_node_obj.x, new_node_obj.y, new_node_obj.z),
                (existing_node.x, existing_node.y, existing_node.z)
            )
            if dist_new_to_existing_spatial <= self.max_dist_primitive and dist_new_to_existing_spatial > 1e-4:
                e_primitive, e_dt = self._predict_primitive_and_time(
                    (new_node_obj.x, new_node_obj.y, new_node_obj.z),
                    (existing_node.x, existing_node.y, existing_node.z)
                )

                if e_primitive is None or e_dt is None: continue

                potential_t_via_new = new_node_obj.t + e_dt
                cost_via_new = new_node_obj.cost + custom_cost_function(new_node_obj, existing_node, e_dt)

                temp_existing_node_state = Node(existing_node.x, existing_node.y, existing_node.z, potential_t_via_new)

                if cost_via_new < existing_node.cost and self.is_collision_free(temp_existing_node_state):
                    existing_node.parent = new_node_obj
                    existing_node.cost = cost_via_new
                    existing_node.t = potential_t_via_new
                    existing_node.primitive_from_parent = e_primitive
                    existing_node.dt_from_parent = e_dt

    def extract_path_for_execution(self, goal_node_in_tree):
        path_segments_for_execution = []
        current = goal_node_in_tree
        while current.parent:
            calibration_factor = RRTStar.PRIMITIVE_STEP_CALIBRATION_FACTORS.get(current.primitive_from_parent, 1.0)
            planned_dt = current.dt_from_parent
            calibrated_dt = planned_dt * calibration_factor

            num_steps = 0
            if calibrated_dt > 1e-3 :
                 num_steps = max(1, round(calibrated_dt * self.CTRL_FREQ))

            if current.primitive_from_parent is None:
                print(f"Warning: Node ({current.x:.2f},{current.y:.2f},{current.z:.2f}) reached at t={current.t:.2f} has no primitive from parent. Skipping segment.")
                current = current.parent
                continue

            path_segments_for_execution.append({
                "primitive_name": current.primitive_from_parent,
                "num_steps": num_steps,
                "target_abstract_xyz": (current.x, current.y, current.z),
                "target_abstract_t": current.t
            })
            current = current.parent
        return path_segments_for_execution[::-1]

    def execute_path(self, planned_segments, start_xyz, show_gui=False): # Corrected method name
        print("\n--- Path Execution Phase ---")
        if not planned_segments:
            print("  No segments to execute.")
            return []

        current_actual_pos = np.array(start_xyz)
        full_executed_trajectory = [np.copy(current_actual_pos)]

        for i, segment in enumerate(planned_segments):
            mp_name_exec = segment["primitive_name"]
            num_steps_exec = segment["num_steps"]

            if num_steps_exec <= 0:
                print(f"  Segment {i+1}/{len(planned_segments)}: Primitive '{mp_name_exec}' has {num_steps_exec} steps. Skipping.")
                continue

            print(f"  Segment {i+1}/{len(planned_segments)}: Cmd: '{mp_name_exec}' for {num_steps_exec} steps (calibrated). Start actual: {np.round(current_actual_pos,2)}")

            ppo_model_dir = os.path.join(self.PPO_MODELS_BASE_DIR, mp_name_exec, ALGO, ENV_CONFIG_NAME, "models")
            ppo_model_path = os.path.join(ppo_model_dir, f"model_iter_{self.CHECKPOINT_ITERATION}.zip")
            ppo_stats_path = os.path.join(ppo_model_dir, "vec_normalize_stats.pkl")

            if not (os.path.exists(ppo_model_path) and os.path.exists(ppo_stats_path)):
                print(f"      ❌ Model/stats missing for {mp_name_exec} (Iter {self.CHECKPOINT_ITERATION}) in {ppo_model_dir}. Skipping segment.")
                full_executed_trajectory.append(np.copy(current_actual_pos))
                continue

            initial_rpys_for_env = np.array([[0.0,0.0,0.0]])

            def make_exec_env_fn():
                env_inst = DroneMPEnv(
                    mp_name=mp_name_exec,
                    num_drones=1, gui=show_gui, # Pass show_gui to the environment
                    initial_xyzs=np.array([current_actual_pos]),
                    initial_rpys=initial_rpys_for_env,
                    episode_len_sec=(num_steps_exec / self.CTRL_FREQ) + 5, # Buffer
                    drone_model=DroneModel.CF2X # Make sure this matches your training
                )
                # Add obstacles to the PyBullet simulation if your DroneMPEnv supports it
                # This is a conceptual placement; your DroneMPEnv needs to handle loading them.
                # For example:
                # if show_gui: # Only add if GUI is shown, or always if env supports it
                #    # The RRTStar instance's obstacles are in self.obstacles
                #    # However, make_exec_env_fn is a local function and doesn't have direct access to self (RRTStar instance)
                #    # You would need to pass self.obstacles to execute_path and then to make_exec_env_fn
                #    # or make DroneMPEnv aware of the RRTStar instance's obstacles list.
                #    # For now, this part needs to be implemented in DroneMPEnv or by passing obstacles.
                #    # Example if obstacles were passed to make_exec_env_fn:
                #    # for obs_data in passed_obstacles:
                #    #    env_inst.add_obstacle_to_sim(obs_data.x_min, obs_data.x_max, ..., obs_data.z_max)

                return env_inst

            exec_env = DummyVecEnv([make_exec_env_fn])
            try:
                exec_env = VecNormalize.load(ppo_stats_path, exec_env)
                exec_env.training = False
                exec_env.norm_reward = False
            except Exception as e:
                print(f"      ❌ Error loading VecNormalize for {mp_name_exec}: {e}. Skipping segment."); exec_env.close(); continue

            try:
                ppo_model = PPO.load(ppo_model_path, env=exec_env)
            except Exception as e:
                print(f"      ❌ Error loading PPO model for {mp_name_exec}: {e}. Skipping segment."); exec_env.close(); continue

            obs_norm_exec = exec_env.reset()
            segment_trajectory_this_primitive = []

            for step_num in range(num_steps_exec):
                action, _ = ppo_model.predict(obs_norm_exec, deterministic=True)
                obs_norm_exec, _, done_array, info_array = exec_env.step(action)

                current_pos_in_step = exec_env.envs[0]._getDroneStateVector(0)[0:3]
                segment_trajectory_this_primitive.append(np.copy(current_pos_in_step))

                if done_array[0]:
                    print(f"      Primitive '{mp_name_exec}' terminated by env at step {step_num + 1}/{num_steps_exec}.")
                    if info_array and "termination_reason" in info_array[0]:
                         print(f"        Reason: {info_array[0]['termination_reason']}")
                    break

            if segment_trajectory_this_primitive:
                 current_actual_pos = segment_trajectory_this_primitive[-1]

            full_executed_trajectory.extend(segment_trajectory_this_primitive)

            print(f"      Primitive '{mp_name_exec}' ended. Actual end pos: {np.round(current_actual_pos,2)}")
            exec_env.close()
            if show_gui: import time; time.sleep(0.05) # Small delay if GUI is active

        print("--- Path Execution Complete ---")
        return np.array(full_executed_trajectory)


    def plot_3d_spatial(self, rrt_path_coords, executed_trajectory_coords, filename_suffix=""):
        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111, projection='3d')

        if rrt_path_coords:
            rrt_x = [p[0] for p in rrt_path_coords]
            rrt_y = [p[1] for p in rrt_path_coords]
            rrt_z = [p[2] for p in rrt_path_coords]
            ax.plot(rrt_x, rrt_y, rrt_z, 'g--', linewidth=1.5, label="RRT* Planned Path (Spatial)")
            ax.scatter(self.start_node_def.x, self.start_node_def.y, self.start_node_def.z, c='blue', s=100, marker='o', label='RRT* Start')
            ax.scatter(self.goal_node_def.x, self.goal_node_def.y, self.goal_node_def.z, c='red', s=100, marker='X', label='RRT* Goal Target')

        if executed_trajectory_coords is not None and len(executed_trajectory_coords) > 0:
            exec_x = executed_trajectory_coords[:, 0]
            exec_y = executed_trajectory_coords[:, 1]
            exec_z = executed_trajectory_coords[:, 2]
            ax.plot(exec_x, exec_y, exec_z, 'b-', linewidth=2, label="Actual Executed Path")
            if len(executed_trajectory_coords) > 0:
                 ax.scatter(executed_trajectory_coords[0,0], executed_trajectory_coords[0,1], executed_trajectory_coords[0,2], c='cyan', s=80, marker='o', edgecolors='k', label='Execution Start')
                 ax.scatter(executed_trajectory_coords[-1,0], executed_trajectory_coords[-1,1], executed_trajectory_coords[-1,2], c='magenta',s=80, marker='P', edgecolors='k', label='Execution End')

        for obs in self.obstacles:
            x_ = [obs.x_min, obs.x_max, obs.x_max, obs.x_min, obs.x_min]
            y_ = [obs.y_min, obs.y_min, obs.y_max, obs.y_max, obs.y_min]
            ax.plot(x_, y_, zs=obs.z_min, zdir='z', color='k', alpha=0.3)
            ax.plot(x_, y_, zs=obs.z_max, zdir='z', color='k', alpha=0.3)
            for i_edge in range(4):
                ax.plot([x_[i_edge], x_[i_edge]], [y_[i_edge], y_[i_edge]], [obs.z_min, obs.z_max], color='k', alpha=0.3)
            ax.text(obs.x_min, obs.y_min, obs.z_min, f"O (t:{obs.t_start}-{obs.t_end})", color='red', fontsize=8)


        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title(f'RRT* 3D Spatial Path Plan and Execution (PPO Iter: {self.CHECKPOINT_ITERATION})')

        all_x = [self.start_node_def.x, self.goal_node_def.x] + [obs.x_min for obs in self.obstacles] + [obs.x_max for obs in self.obstacles]
        all_y = [self.start_node_def.y, self.goal_node_def.y] + [obs.y_min for obs in self.obstacles] + [obs.y_max for obs in self.obstacles]
        all_z = [self.start_node_def.z, self.goal_node_def.z] + [obs.z_min for obs in self.obstacles] + [obs.z_max for obs in self.obstacles]
        if rrt_path_coords: all_x.extend([p[0] for p in rrt_path_coords]); all_y.extend([p[1] for p in rrt_path_coords]); all_z.extend([p[2] for p in rrt_path_coords])
        if executed_trajectory_coords is not None and len(executed_trajectory_coords) > 0: all_x.extend(executed_trajectory_coords[:,0]); all_y.extend(executed_trajectory_coords[:,1]); all_z.extend(executed_trajectory_coords[:,2])

        if all_x: ax.set_xlim(min(all_x)-0.5, max(all_x)+0.5)
        if all_y: ax.set_ylim(min(all_y)-0.5, max(all_y)+0.5)
        if all_z: ax.set_zlim(min(all_z)-0.5, max(all_z)+0.5)

        try:
            x_range = np.ptp(ax.get_xlim())
            y_range = np.ptp(ax.get_ylim())
            z_range = np.ptp(ax.get_zlim())
            ax.set_box_aspect((x_range if x_range > 1e-3 else 1,
                               y_range if y_range > 1e-3 else 1,
                               z_range if z_range > 1e-3 else 1))
        except AttributeError:
            pass

        plt.legend()
        output_path = os.path.join(os.path.abspath(os.getcwd()), f"rrt_3d_spatial_exec_path{filename_suffix}.png")
        plt.savefig(output_path)
        print(f"✅ Spatial Execution Plot saved to {output_path}")
        plt.close(fig)


if __name__ == '__main__':
    start_node_params = (8, 0, 0.1, 0)
    goal_node_params = (1, 7, 7.1, 0)

    x_dim_bounds = (-1, 8)
    y_dim_bounds = (-1, 8)
    z_dim_bounds = (-0.5, 8)
    time_overall_bounds = (0, 100)

    min_prim_dist_step = 0.1
    max_prim_dist_step = 0.5
    min_prim_time_pred = 0.1
    max_prim_time_pred = 1.5

    example_obstacles_list = [
        Obstacle(x_min=2, x_max=3, y_min=2, y_max=3, z_min=0, z_max=2, t_start=0, t_end=50),
        Obstacle(x_min=5, x_max=6, y_min=5, y_max=6, z_min=1, z_max=3, t_start=10, t_end=60),
        Obstacle(x_min=1, x_max=2, y_min=4, y_max=5, z_min=0.5, z_max=2.5, t_start=0, t_end=100)
    ]

    script_file_dir = os.path.dirname(os.path.abspath(__file__))
    gp_model_directory = os.path.join(script_file_dir, "Ideal", "GP_models_3D")
    os.makedirs(gp_model_directory, exist_ok=True)

    defined_primitive_files = [definition[1] for definition in RRTStar.PRIMITIVE_DEFINITIONS.values()]
    for gp_fname in defined_primitive_files:
        gp_fpath = os.path.join(gp_model_directory, gp_fname)
        if not os.path.exists(gp_fpath) and gp_fname:
            print(f"Creating dummy GP for missing file: {gp_fname}")
            with open(gp_fpath, "wb") as f:
                pickle.dump(StandardDummyGP(min_prim_time_pred, max_prim_time_pred, max_prim_dist_step), f)

    CHECKPOINT_ITERATION = 43 # Default

    user_checkpoint_iter_str = input(f"Enter PPO model checkpoint iteration to use (e.g., 50, default: {CHECKPOINT_ITERATION}): ").strip()
    if user_checkpoint_iter_str.isdigit():
        checkpoint_to_use = int(user_checkpoint_iter_str)
    else:
        print(f"Invalid input or no input, using default checkpoint iteration: {CHECKPOINT_ITERATION}")
        checkpoint_to_use = CHECKPOINT_ITERATION

    print(f"Using PPO models from iteration: {checkpoint_to_use}")

    rrt_planner_instance = RRTStar(
        start_node_params, goal_node_params,
        x_dim_bounds, y_dim_bounds, z_dim_bounds, time_overall_bounds,
        example_obstacles_list,
        min_prim_dist_step, max_prim_dist_step,
        min_prim_time_pred, max_prim_time_pred,
        ctrl_freq=CTRL_FREQ,
        max_iter=5000,
        gp_model_dir_override=gp_model_directory,
        ppo_model_base_dir_override=PPO_MODELS_BASE_DIR,
        checkpoint_iter_override=checkpoint_to_use,
        cost_time_weight=0.05,
        cost_remaining_dist_weight=2.5,
        dominant_axis_adherence_factor= 1.0
    )

    planned_execution_segments = rrt_planner_instance.find_path()

    rrt_ideal_path_coords = []
    if planned_execution_segments:
        rrt_ideal_path_coords.append((rrt_planner_instance.start_node_def.x,
                                     rrt_planner_instance.start_node_def.y,
                                     rrt_planner_instance.start_node_def.z,
                                     rrt_planner_instance.start_node_def.t))
        for seg in planned_execution_segments:
            rrt_ideal_path_coords.append((*seg["target_abstract_xyz"], seg["target_abstract_t"]))

    executed_trajectory = None
    if planned_execution_segments:
        print(f"\nPlanning phase complete. RRT* path has {len(planned_execution_segments)} segments.")
        # Corrected the method name from execute__path to execute_path
        executed_trajectory = rrt_planner_instance.execute_path(
            planned_segments=planned_execution_segments, 
            start_xyz=(start_node_params[0], start_node_params[1], start_node_params[2]),
            show_gui=True 
        )
    else:
        print("\nNo RRT* path found for execution.")

    rrt_planner_instance.plot_3d_spatial(rrt_ideal_path_coords, executed_trajectory, filename_suffix=f"_iter{checkpoint_to_use}_3DGoal_WithObs_GUI")

    print("\n--- RRT* 3D Planner and Executor Finished ---")
