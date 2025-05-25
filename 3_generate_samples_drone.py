#!/usr/bin/env python3
import os
import sys
import csv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import time # For potential delays if needed

# --- SETUP PATH ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from DroneMPEnv import DroneMPEnv
except ImportError:
    print("❌ Could not import DroneMPEnv. Ensure DroneMPEnv.py is in the Python path.")
    sys.exit(1)

from gym_pybullet_drones.utils.enums import DroneModel


def generate_drone_samples(algo="PPO",
                           model_checkpoint_base_name="model_iter",
                           sample_count=100,
                           max_steps_per_sample=150,
                           gui_for_debug_mp=None,
                           max_cumulative_reward_threshold=None,
                           target_cumulative_reward_for_success=None):
    """
    Generates samples for each motion primitive using a trained model.

    Args:
        algo (str): The RL algorithm used (e.g., "PPO").
        model_checkpoint_base_name (str): Base name for model checkpoints.
        sample_count (int): Number of samples (rollouts) to generate per motion primitive.
        max_steps_per_sample (int): Max control steps for the longest sample rollout.
                                    The j-th sample runs for a portion of these max_steps.
        gui_for_debug_mp (str, optional): Name of a single motion primitive to enable GUI for debugging.
        max_cumulative_reward_threshold (float, optional): If set, sample generation stops if cumulative reward exceeds this.
        target_cumulative_reward_for_success (float, optional): If set, sample generation stops if this cumulative reward is met or exceeded.
    """
    env_config_name = "model_0p1_pid_rl" # Should match the training config name

    results_base_dir = os.path.join(script_dir, "training_results")
    samples_output_dir = os.path.join(results_base_dir, "all_samples")
    os.makedirs(samples_output_dir, exist_ok=True)
    print(f"ℹ️ Sample CSV files will be saved in: {samples_output_dir}")

    motion_primitives = [
        "Hover", "Forward", "Backward", "Left", "Right",
        "Up", "Down", "Yaw_CW", "Yaw_CCW",
        "Diagonal_Forward_Left", "Diagonal_Forward_Right"
    ]

    checkpoint_iteration_str = input(f"Enter iteration number of PPO models for sample generation (e.g., 50, 100): ").strip()
    if not checkpoint_iteration_str.isdigit():
        print("❌ Invalid iteration number. Exiting.")
        sys.exit(1)
    checkpoint_iteration = int(checkpoint_iteration_str)

    # Get user input for target_cumulative_reward_for_success if not passed directly via function call
    if target_cumulative_reward_for_success is None:
        user_target_reward_str = input(f"Enter a target cumulative reward for successful sample termination (e.g., 150), or press Enter to disable: ").strip()
        if user_target_reward_str:
            try:
                target_cumulative_reward_for_success = float(user_target_reward_str)
                print(f"ℹ️ Using target cumulative reward for success: {target_cumulative_reward_for_success}")
            except ValueError:
                print(f"⚠️ Invalid number for target success reward. Disabling this feature.")
                target_cumulative_reward_for_success = None
        else:
            print(f"ℹ️ Target cumulative reward for successful sample termination disabled.")


    print("\n============== IMPORTANT PREREQUISITES ==============")
    print("1. Ensure your URDF file has the CORRECT physical 'kf' value.")
    print("2. Ensure DroneMPEnv.py uses a controller that correctly interprets this 'kf'.")
    print("3. If using a modified controller (like DSLPIDControl), ENSURE ITS PID GAINS ARE TUNED.")
    print("4. The PPO models loaded below MUST BE RETRAINED with the corrected environment.")
    print("   Using old PPO models with a changed environment will lead to poor/failed sample generation.")
    print("====================================================\n")


    for mp in motion_primitives:
        print(f"\n📌 Attempting to generate {sample_count} samples for '{mp}' using PPO models from iteration {checkpoint_iteration}")

        model_load_dir = os.path.join(results_base_dir, mp, algo, env_config_name, "models")
        model_path = os.path.join(model_load_dir, f"{model_checkpoint_base_name}_{checkpoint_iteration}.zip")
        stats_path = os.path.join(model_load_dir, "vec_normalize_stats.pkl")

        if not os.path.isfile(model_path):
            print(f"  ❌ Model file not found: {model_path}. Skipping '{mp}'.")
            continue
        if not os.path.isfile(stats_path):
            print(f"  ❌ VecNormalize stats file not found: {stats_path}. Skipping '{mp}'.")
            print(f"     Training must use VecNormalize and save stats for consistent sample generation.")
            continue

        print(f"  Found model: {model_path}")
        print(f"  Found stats: {stats_path}")

        enable_gui_this_mp = (mp == gui_for_debug_mp)
        if enable_gui_this_mp:
            print(f"  💡 GUI enabled for debugging '{mp}'.")

        def make_env_fn():
            env_instance = DroneMPEnv(
                mp_name=mp,
                drone_model=DroneModel.CF2X,
                num_drones=1,
                gui=enable_gui_this_mp,
                episode_len_sec=(max_steps_per_sample / 240) + 5 # Max episode length in seconds, give some buffer
                                                              # Assuming ctrl_freq=30Hz from DroneMPEnv
            )
            return env_instance

        try:
            env = DummyVecEnv([make_env_fn])
            env = VecNormalize.load(stats_path, env)
            env.training = False
            env.norm_reward = False
            print(f"  ✅ Environment for '{mp}' created and VecNormalize stats loaded.")
        except Exception as e:
            print(f"  ❌ Error creating or loading VecNormalize for '{mp}': {e}. Skipping.")
            if 'env' in locals() and env is not None: env.close()
            continue

        try:
            model = PPO.load(model_path, env=env)
            print(f"  🧠 PPO Model for '{mp}' loaded successfully. (REMINDER: Model must be trained with current env dynamics!)")
        except Exception as e:
            print(f"  ❌ Error loading PPO model for '{mp}' from {model_path}: {e}. Skipping.")
            env.close()
            continue

        csv_file_name = f"{mp}_iter{checkpoint_iteration}_samples.csv"
        csv_file_path = os.path.join(samples_output_dir, csv_file_name)

        generated_samples_for_mp = 0
        with open(csv_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric_value", "steps", "cumulative_reward",
                             "terminated_early_env", "initial_z", "final_z",
                             "reward_cap_met", "target_reward_met"])

            for j in range(sample_count):
                obs_norm = env.reset()
                initial_true_state = env.envs[0]._getDroneStateVector(0)
                initial_pos = initial_true_state[0:3]
                initial_rpy = initial_true_state[7:10]
                initial_z_for_log = round(float(initial_pos[2]), 4)

                done_env = False 
                terminated_early_env_flag = False
                reward_cap_met_flag = False
                target_reward_met_flag = False 
                current_rollout_cumulative_reward = 0.0
                num_control_steps_for_sample = max(1, int(((j + 1) / sample_count) * max_steps_per_sample))
                actual_steps_taken = 0

                for i in range(num_control_steps_for_sample):
                    actual_steps_taken += 1
                    action, _ = model.predict(obs_norm, deterministic=True)
                    obs_norm, reward_vec, done_vec, info_vec = env.step(action)
                    done_env = done_vec[0]
                    current_rollout_cumulative_reward += reward_vec[0]

                    if done_env:
                        terminated_early_env_flag = True
                        break
                    
                    if max_cumulative_reward_threshold is not None and \
                       current_rollout_cumulative_reward >= max_cumulative_reward_threshold:
                        reward_cap_met_flag = True
                        break
                    
                    if target_cumulative_reward_for_success is not None and \
                       current_rollout_cumulative_reward >= target_cumulative_reward_for_success:
                        target_reward_met_flag = True
                        break
                
                final_true_state = env.envs[0]._getDroneStateVector(0)
                final_pos = final_true_state[0:3]
                final_rpy = final_true_state[7:10]
                final_z_for_log = round(float(final_pos[2]), 4)

                metric_value = 0.0
                if mp in ["Yaw_CW", "Yaw_CCW"]:
                    yaw_change = final_rpy[2] - initial_rpy[2]
                    metric_value = (yaw_change + np.pi) % (2 * np.pi) - np.pi
                elif mp == "Hover":
                    target_hover_point = np.array([0.0, 0.0, env.envs[0].target_altitude])
                    metric_value = np.linalg.norm(final_pos - target_hover_point)
                else: 
                    displacement_vec = final_pos - initial_pos
                    if mp == "Forward": metric_value = displacement_vec[0]
                    elif mp == "Backward": metric_value = -displacement_vec[0]
                    elif mp == "Left": metric_value = displacement_vec[1]
                    elif mp == "Right": metric_value = -displacement_vec[1]
                    elif mp == "Up": metric_value = displacement_vec[2]
                    elif mp == "Down": metric_value = -displacement_vec[2]
                    elif mp == "Diagonal_Forward_Left": metric_value = np.linalg.norm(np.array([displacement_vec[0], displacement_vec[1]]))
                    elif mp == "Diagonal_Forward_Right": metric_value = np.linalg.norm(np.array([displacement_vec[0], displacement_vec[1]]))
                    else: metric_value = np.linalg.norm(displacement_vec)

                writer.writerow([
                    round(float(metric_value), 4),
                    actual_steps_taken,
                    round(float(current_rollout_cumulative_reward), 4),
                    1 if terminated_early_env_flag else 0,
                    initial_z_for_log,
                    final_z_for_log,
                    1 if reward_cap_met_flag else 0,
                    1 if target_reward_met_flag else 0
                ])
                generated_samples_for_mp += 1

                if (j + 1) % max(1, (sample_count // 10)) == 0:
                    print(f"    Generated {j+1}/{sample_count} samples for '{mp}' (last metric: {metric_value:.3f}, steps: {actual_steps_taken}, reward: {current_rollout_cumulative_reward:.2f})...")
        
        if generated_samples_for_mp > 0:
            print(f"  ✅ Samples for '{mp}' ({generated_samples_for_mp} lines) saved to {csv_file_path}")
        else:
            print(f"  ⚠️ No samples were effectively generated for '{mp}'.")
            print(f"     This often means PPO models are incompatible with current DroneMPEnv dynamics (check KF, PID tuning, and ensure PPO models are RETRAINED).")

        env.close()

    print("\n🎉 All sample generation attempts complete.")
    print(f"   Please check the directory: {samples_output_dir}")

if __name__ == "__main__":
    debug_mp_with_gui = None
    
    reward_cap = None
    user_max_reward_str = input(f"Enter a maximum cumulative reward CAP for sample termination (e.g., 2000), or press Enter to disable: ").strip()
    if user_max_reward_str:
        try:
            reward_cap = float(user_max_reward_str)
            print(f"ℹ️ Using max cumulative reward CAP: {reward_cap}")
        except ValueError:
            print(f"⚠️ Invalid number for reward CAP. Disabling reward CAP.")
            reward_cap = None
    else:
        print(f"ℹ️ Max cumulative reward CAP disabled.")

    # target_success_reward will be prompted for inside the function if this remains None
    target_success_reward_arg = None 

    generate_drone_samples(
        model_checkpoint_base_name="model_iter",
        sample_count=100,            
        max_steps_per_sample=100,    
        gui_for_debug_mp=debug_mp_with_gui,
        max_cumulative_reward_threshold=reward_cap,
        target_cumulative_reward_for_success=target_success_reward_arg
    )