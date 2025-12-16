import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# ---------- Settings ----------
ENV_ID = "Hopper-v4"
TRAIN_TIMESTEPS = 100_000   # fast run; later you can increase to 200_000+
EVAL_STEPS = 1500           # steps to log in one rollout
SEED = 42
# -----------------------------

env = gym.make(ENV_ID)
env.reset(seed=SEED)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    gamma=0.99,
    n_steps=2048,
    batch_size=64,
    seed=SEED,
)

print("Training started...")
model.learn(total_timesteps=TRAIN_TIMESTEPS)
print("Training finished.")

# ----- Evaluate one episode + log -----
obs, _ = env.reset(seed=SEED)
vel_x, height_z, act_mag = [], [], []

for _ in range(EVAL_STEPS):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # For Hopper-v4, obs conventionally includes:
    # obs[0] = x position (not velocity), obs[1] = z height, obs[5] ~ x velocity (varies by version)
    # We'll compute a robust proxy using info if available; otherwise fall back to obs indexing.
    z = obs[1]
    height_z.append(z)

    act_mag.append(float(np.linalg.norm(action)))

    # try to get x-velocity from info (if present)
    vx = None
    if isinstance(info, dict):
        for k in ["x_velocity", "xvel", "vel_x", "forward_vel"]:
            if k in info:
                vx = float(info[k])
                break
    if vx is None:
        # fallback: many Hopper obs layouts have x-velocity around index 5
        vx = float(obs[5]) if len(obs) > 5 else 0.0
    vel_x.append(vx)

    if terminated or truncated:
        break

env.close()

vel_x = np.array(vel_x)
height_z = np.array(height_z)
act_mag = np.array(act_mag)

print("\n=== Quick metrics (1 rollout) ===")
print("Steps:", len(vel_x))
print("Average forward velocity (m/s):", float(vel_x.mean()))
print("Velocity std (m/s):", float(vel_x.std()))
print("Average action magnitude:", float(act_mag.mean()))
print("Average height z (m):", float(height_z.mean()))

# ----- Plots -----
plt.figure()
plt.plot(vel_x)
plt.title("Forward velocity over time (RL PPO)")
plt.xlabel("Timestep")
plt.ylabel("v_x (m/s)")
plt.grid(True)

plt.figure()
plt.plot(height_z)
plt.title("Body height over time (RL PPO)")
plt.xlabel("Timestep")
plt.ylabel("z (m)")
plt.grid(True)

plt.figure()
plt.plot(act_mag)
plt.title("Action magnitude over time (RL PPO)")
plt.xlabel("Timestep")
plt.ylabel("||action||")
plt.grid(True)

plt.show()
