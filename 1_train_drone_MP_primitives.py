#!/usr/bin/env python3
import os
import sys
import numpy as np
# import matplotlib # Not strictly needed if not plotting directly in this script
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold # Optional

# --- SETUP PATH ---
# Ensure DroneMPEnv can be imported
# Assumes this script is in a directory, and DroneMPEnv.py is either
# in the same directory or in the parent directory.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) 

if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
if project_root not in sys.path and project_root != script_dir:
    sys.path.insert(0, project_root)

try:
    from DroneMPEnv import DroneMPEnv 
except ImportError:
    print("❌ Could not import DroneMPEnv. Ensure it's in the Python path (e.g., same directory or parent directory of this script).")
    sys.exit(1)

from gym_pybullet_drones.utils.enums import DroneModel


def select_motion_primitive():
    mp_map = {
        "1": "Hover", "2": "Forward", "3": "Backward", "4": "Left", "5": "Right",
        "6": "Up", "7": "Down", "8": "Yaw_CW", "9": "Yaw_CCW",
        "10": "Diagonal_Forward_Left", "11": "Diagonal_Forward_Right"
    }
    print("\nSelect Motion Primitive for Training:")
    for k, v in mp_map.items(): print(f"  {k}: {v}")
    sel = input("\nYour choice: ").strip()
    if sel not in mp_map:
        print("❌ Invalid motion primitive selected. Exiting.")
        sys.exit(1)
    return mp_map[sel]

def main():
    algo = "PPO" # Focusing on PPO
    ModelClass = PPO

    MP_name = select_motion_primitive()
    print(f"🚀 Starting training for Motion Primitive: {MP_name} using {algo}")

    # --- Directories ---
    env_config_name = "model_0p1_pid_rl" # Descriptive name for this env/training config
    # Centralized results directory, script_dir is where this script is.
    results_base_dir = os.path.join(script_dir, "training_results") 
    
    tb_log_dir    = os.path.join(results_base_dir, MP_name, algo, env_config_name, "tensorboard")
    model_save_dir = os.path.join(results_base_dir, MP_name, algo, env_config_name, "models")
    stats_path = os.path.join(model_save_dir, "vec_normalize_stats.pkl") # Path for VecNormalize stats
    # plot_save_dir = os.path.join(results_base_dir, MP_name, algo, env_config_name, "plots") # If plotting here

    os.makedirs(tb_log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    # os.makedirs(plot_save_dir, exist_ok=True) # Only if you add plotting back

    # --- Environment Creation ---
    def make_env_fn():
        env = DroneMPEnv(
            mp_name=MP_name,
            num_drones=1,
            drone_model=DroneModel.CF2X, 
            gui=False, 
            # episode_len_sec=10 # Using default from DroneMPEnv
        )
        return env

    env = DummyVecEnv([make_env_fn])
    # CRITICAL: Normalize observations and rewards.
    # Ensure gamma matches the PPO agent's gamma.
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.99) 
    print("✅ Environment created and wrapped with VecNormalize.")
    print(f"   VecNormalize stats will be saved to: {stats_path}")


    # --- PPO Agent Configuration ---
    policy_kwargs = dict(
        log_std_init=-1.5,  # Allow wider action exploration from the start
    )

    model = ModelClass(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
        tensorboard_log=tb_log_dir,
        policy_kwargs=policy_kwargs,
        n_steps=4096,               # Increased rollout length
        batch_size=128,             # Larger batch size for stability
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.01,              # Encourage exploration more aggressively
        max_grad_norm=0.5
    )

    timesteps_per_iteration = model.n_steps * 10  # 4096 * 10 = 40960

    print(f"🧠 {algo} agent built. Policy kwargs: {policy_kwargs}")
    #print(f"   Tensorboard log directory: {tb_log_dir}")
    #print(f"   Model save directory: {model_save_dir}")

    # --- Training Loop ---
    total_iterations = 1000 # Number of times model.learn() is called (each call saves a checkpoint)
    # Set timesteps_per_iteration to be a multiple of model.n_steps for clean logging and updates
    # This means each .learn() call will perform 'N' PPO updates (each update processes n_steps data).
    #timesteps_per_iteration = model.n_steps * 10 # e.g., 10 PPO updates per iteration/save
    
    #print(f"🏁 Starting training for {total_iterations} iterations.")
    #print(f"   Each iteration will run for {timesteps_per_iteration} environment steps.")
    #print(f"   Total timesteps to train: {total_iterations * timesteps_per_iteration}")
    #print(f"   Monitor progress with TensorBoard: tensorboard --logdir \"{os.path.abspath(tb_log_dir)}\"")


    try:
        for i in tqdm(range(1, total_iterations + 1), desc=f"Training {MP_name} with {algo}"):
            model.learn(
                total_timesteps=timesteps_per_iteration,
                reset_num_timesteps=False, 
                log_interval=1 # Log to TensorBoard after each rollout collection (n_steps)
            )

            model_checkpoint_path = os.path.join(model_save_dir, f"model_iter_{i}.zip")
            model.save(model_checkpoint_path)
            env.save(stats_path) # Save VecNormalize statistics with the model

            # The 'ep_info_buffer' is part of the Monitor wrapper, which VecNormalize handles internally for logging.
            # Accessing it directly on 'env' (which is VecNormalize) is not the standard way.
            # True mean reward should be monitored via TensorBoard ('rollout/ep_rew_mean').
            print(f"\n✅ Iteration {i}/{total_iterations} complete. Model saved to {model_checkpoint_path}")
            print(f"   VecNormalize stats saved to {stats_path}")

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user. Saving current model...")
        model.save(os.path.join(model_save_dir, "model_interrupted.zip"))
        env.save(stats_path) 
        print("   Model and stats saved.")
    finally:
        env.close() 

    print("\n✅ Training finished.")
    print(f"   Find final models in: {model_save_dir}")
    print(f"   Find TensorBoard logs in: {tb_log_dir}")

if __name__=="__main__":
    main()
