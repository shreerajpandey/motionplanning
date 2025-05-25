#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for saving plots
import matplotlib.pyplot as plt
from stable_baselines3 import PPO # Assuming PPO was used for training
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize # Import VecNormalize

# --- SETUP PATH ---
# Ensure DroneMPEnv can be imported
base_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir) # Assuming DroneMPEnv.py is in the parent directory or base_dir
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
if project_root not in sys.path and project_root != base_dir : # If DroneMPEnv is one level up
    sys.path.insert(0, project_root)

from DroneMPEnv import DroneMPEnv # Import your custom environment
from gym_pybullet_drones.utils.enums import DroneModel # For DroneModel.CF2X if needed


def select_motion_primitive():
    mp_map = {
        "1": "Hover",
        "2": "Forward",
        "3": "Backward",
        "4": "Left",
        "5": "Right",
        "6": "Up",
        "7": "Down",
        "8": "Yaw_CW",
        "9": "Yaw_CCW",
        "10": "Diagonal_Forward_Left",
        "11": "Diagonal_Forward_Right"
    }
    print("\nSelect Motion Primitive for Testing:")
    for key, value in mp_map.items():
        print(f"  {key}: {value}")
    mp_sel = input("\nYour choice: ").strip()
    MP_name = mp_map.get(mp_sel)
    if MP_name is None:
        print("❌ Invalid motion primitive selected. Exiting.")
        sys.exit(1)
    return MP_name

def main():
    MP_name = select_motion_primitive()
    # Prompt for the specific iteration number of the model
    checkpoint_iter = input(f"Enter iteration number of the model for {MP_name} to load (e.g., 155): ").strip()
    if not checkpoint_iter.isdigit():
        print("❌ Invalid iteration number. Exiting.")
        sys.exit(1)

    algo = "PPO" # Assuming PPO was used for training
    env_config_name = "model_0p1_pid_rl" # Should match the training config name

    # Corrected base_dir for path construction relative to the script's location
    script_base_dir = os.path.dirname(os.path.abspath(__file__)) 

    model_dir = os.path.join(script_base_dir, "training_results", MP_name, algo, env_config_name, "models")
    model_path = os.path.join(model_dir, f"model_iter_{checkpoint_iter}.zip")
    stats_path = os.path.join(model_dir, "vec_normalize_stats.pkl") # Path to VecNormalize stats

    if not os.path.isfile(model_path):
        print(f"❌ Cannot find model checkpoint: {model_path}")
        sys.exit(1)
    # stats_path is optional for testing if you sometimes train without VecNormalize, but highly recommended
    # if not os.path.isfile(stats_path):
    #     print(f"❌ Cannot find VecNormalize stats: {stats_path}. Testing might be inaccurate if VecNormalize was used in training.")
    #     sys.exit(1) 


    print(f"🧪 Testing Motion Primitive: {MP_name} using model: {model_path}")

    # --- Environment Creation for Testing ---
    gui_enabled_for_test = True 

    def make_env_fn_test():
        env_instance = DroneMPEnv(
            mp_name=MP_name,
            num_drones=1,
            drone_model=DroneModel.CF2X,
            gui=gui_enabled_for_test, 
        )
        return env_instance

    env = DummyVecEnv([make_env_fn_test])
    
    if os.path.isfile(stats_path):
        print(f"Loading VecNormalize stats from: {stats_path}")
        env = VecNormalize.load(stats_path, env)
        env.training = False 
        env.norm_reward = False 
        print("✅ Test environment created and VecNormalize stats loaded.")
    else:
        print("⚠️ VecNormalize stats file not found. Proceeding without normalization for testing.")
        print("   If training used VecNormalize, results might be skewed or model loading might fail if expecting a VecNormalizedEnv.")

    # --- Load Model ---
    try:
        model = PPO.load(model_path, env=env) 
        print("🧠 Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        env.close()
        return

    # --- Run One Episode ---
    # obs is the normalized observation if VecNormalize is used
    obs_norm = env.reset() 
    
    positions = [] # To store true (unnormalized) positions
    rewards_collected = [] 
    total_unnormalized_reward = 0.0
    episode_steps = 0
    
    # Get initial true state for plotting start point accurately
    initial_true_state_vector = env.envs[0]._getDroneStateVector(0)
    initial_pos_true = initial_true_state_vector[0:3]
    positions.append(initial_pos_true) # Log initial true position

    max_episode_steps = int(env.envs[0].EPISODE_LEN_SEC * env.envs[0].PYB_FREQ)
    print(f"🚀 Starting test episode (max steps: {max_episode_steps})...")

    for step in range(max_episode_steps):
        action, _ = model.predict(obs_norm, deterministic=True) 
        obs_norm, reward, done_array, info_array = env.step(action)

        current_unnormalized_reward = reward[0] # This is unnormalized due to env.norm_reward = False
        # info = info_array[0] # info can be used if needed

        # Get TRUE (unnormalized) state for plotting and analysis
        true_state_vector = env.envs[0]._getDroneStateVector(0)
        pos_true = true_state_vector[0:3]

        positions.append(pos_true) # Append true position
        rewards_collected.append(current_unnormalized_reward)
        total_unnormalized_reward += current_unnormalized_reward
        episode_steps += 1

        if env.envs[0].GUI:
            time.sleep(1./env.envs[0].PYB_FREQ) 

        if done_array[0]:
            print(f"\n✅ Episode finished after {episode_steps} steps.")
            break
    
    if not done_array[0] and episode_steps == max_episode_steps:
        print(f"\nℹ️ Episode truncated after reaching max steps: {max_episode_steps}.")

    env.close()
    print("🚪 Environment closed.")

    # --- Plotting Results ---
    if len(positions) < 2 : # Need at least two points to plot lines
        print(f"⚠️ Only {len(positions)} position(s) recorded, insufficient for detailed plotting.")
        return

    positions_np = np.array(positions)
    # t should correspond to the number of states recorded, which is episode_steps + 1 (including initial state)
    t = np.arange(len(positions_np)) 

    plt.figure(figsize=(12, 10))

    plt.subplot(3, 2, 1)
    plt.plot(t, positions_np[:, 0], lw=2, label="X (True)")
    plt.title("X Position"); plt.xlabel("Step Count"); plt.ylabel("Meters"); plt.grid(True); plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(t, positions_np[:, 1], lw=2, label="Y (True)")
    plt.title("Y Position"); plt.xlabel("Step Count"); plt.ylabel("Meters"); plt.grid(True); plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(t, positions_np[:, 2], lw=2, label="Z (Altitude - True)")
    plt.axhline(y=env.envs[0].target_altitude, color='r', linestyle='--', label=f"Target Alt ({env.envs[0].target_altitude:.2f}m)")
    plt.title("Z Position (Altitude)"); plt.xlabel("Step Count"); plt.ylabel("Meters"); plt.grid(True); plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(positions_np[:, 0], positions_np[:, 1], marker='.', linestyle='-', markersize=3, label="Path")
    plt.plot(positions_np[0, 0], positions_np[0, 1], 'go', markersize=8, label="Start") 
    if MP_name == "Hover": 
        plt.plot(0,0,'rx', markersize=8, label="Origin/Target (Hover)") 
    plt.title("XY Path (True Positions)"); plt.xlabel("X (m)"); plt.ylabel("Y (m)"); plt.grid(True); plt.axis('equal'); plt.legend()

    # Ensure rewards_collected has data before plotting
    if rewards_collected:
        steps_rewards = np.arange(len(rewards_collected))
        plt.subplot(3, 2, 5)
        plt.plot(steps_rewards, rewards_collected, lw=1.5, label="Step Reward")
        plt.title("Step Rewards (Unnormalized)"); plt.xlabel("Step Index (0 to N-1)"); plt.ylabel("Reward"); plt.grid(True); plt.legend()

        cumulative_rewards = np.cumsum(rewards_collected)
        plt.subplot(3, 2, 6)
        plt.plot(steps_rewards, cumulative_rewards, lw=2, label="Cumulative Reward")
        plt.title("Cumulative Reward (Unnormalized)"); plt.xlabel("Step Index (0 to N-1)"); plt.ylabel("Total Reward"); plt.grid(True); plt.legend()
    else:
        print("⚠️ No rewards recorded to plot.")


    plt.suptitle(f"Test Results: {MP_name} - Checkpoint iter_{checkpoint_iter} - Total Reward: {total_unnormalized_reward:.2f}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 

    plot_save_dir = os.path.join(script_base_dir, "training_results", MP_name, algo, env_config_name, "plots")
    os.makedirs(plot_save_dir, exist_ok=True)
    plot_filename = f"test_{algo}_{MP_name}_iter_{checkpoint_iter}.png"
    out_png_path = os.path.join(plot_save_dir, plot_filename)

    plt.savefig(out_png_path)
    print(f"\n📈 Plot saved at: {out_png_path}")
    print(f"🎯 Total Unnormalized Reward Collected: {total_unnormalized_reward:.2f}")

if __name__ == "__main__":
    main()
