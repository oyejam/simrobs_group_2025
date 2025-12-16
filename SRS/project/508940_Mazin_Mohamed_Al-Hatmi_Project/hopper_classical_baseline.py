import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

ENV_ID = "Hopper-v4"
STEPS = 1500
SEED = 42

# PD gains (simple baseline; we’ll tune if needed)
KP = np.array([25.0, 25.0, 15.0])   # thigh, leg, foot
KD = np.array([3.0,  3.0,  2.0])

# Targets
V_TARGET = 1.0  # smaller target = easier stability
TORSO_TARGET = 0.0

env = gym.make(ENV_ID)
obs, _ = env.reset(seed=SEED)

vel_x, z_trace, act_mag = [], [], []

for _ in range(STEPS):
    # Access MuJoCo state directly (more reliable than obs indexing)
    data = env.unwrapped.data
    qpos = data.qpos.copy()   # [x, z, torso_angle, thigh, leg, foot]
    qvel = data.qvel.copy()   # corresponding velocities

    xdot = float(qvel[0])
    z = float(qpos[1])
    torso = float(qpos[2])

    # Joint angles
    thigh, leg, foot = float(qpos[3]), float(qpos[4]), float(qpos[5])
    thigh_d, leg_d, foot_d = float(qvel[3]), float(qvel[4]), float(qvel[5])

    # --- Heuristic targets ---
    # 1) Try to keep torso upright
    torso_err = (TORSO_TARGET - torso)

    # 2) Velocity feedback: if too slow, swing leg forward more (thigh target)
    v_err = (V_TARGET - xdot)

    # Desired joint angles (very simple, just to get a non-RL baseline)
    thigh_des = 0.1 + 0.15 * v_err + 0.2 * torso_err
    leg_des   = -0.7
    foot_des  = 0.05 - 0.15 * torso_err

    q_des = np.array([thigh_des, leg_des, foot_des], dtype=np.float64)
    q = np.array([thigh, leg, foot], dtype=np.float64)
    qd = np.array([thigh_d, leg_d, foot_d], dtype=np.float64)

    # PD torque
    torque = KP * (q_des - q) - KD * qd
    torque = np.clip(torque, -0.6, 0.6)  # softer actions to avoid instant fall

    obs, reward, terminated, truncated, info = env.step(torque)

    vel_x.append(xdot)
    z_trace.append(z)
    act_mag.append(float(np.linalg.norm(torque)))

    if terminated or truncated:
        break

env.close()

vel_x = np.array(vel_x)
z_trace = np.array(z_trace)
act_mag = np.array(act_mag)

print("\n=== Classical baseline metrics (1 rollout) ===")
print("Steps:", len(vel_x))
print("Average forward velocity (m/s):", float(vel_x.mean()))
print("Velocity std (m/s):", float(vel_x.std()))
print("Average action magnitude:", float(act_mag.mean()))
print("Average height z (m):", float(z_trace.mean()))

plt.figure()
plt.plot(vel_x)
plt.title("Forward velocity over time (Classical PD baseline)")
plt.xlabel("Timestep")
plt.ylabel("v_x (m/s)")
plt.grid(True)

plt.figure()
plt.plot(z_trace)
plt.title("Body height over time (Classical PD baseline)")
plt.xlabel("Timestep")
plt.ylabel("z (m)")
plt.grid(True)

plt.figure()
plt.plot(act_mag)
plt.title("Action magnitude over time (Classical PD baseline)")
plt.xlabel("Timestep")
plt.ylabel("||action||")
plt.grid(True)

plt.show()
