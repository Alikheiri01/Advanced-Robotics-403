import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

# --- Simulation Parameters ---
np.random.seed(42)
dt = 1.0
total_time = 120
timesteps = int(total_time / dt)
n_dim = 6

# --- Motion Model (Constant Acceleration) ---
F = np.eye(n_dim)
F[0, 2] = dt; F[1, 3] = dt
F[2, 4] = dt; F[3, 5] = dt
F[0, 4] = 0.5 * dt**2; F[1, 5] = 0.5 * dt**2

# --- Process Noise Matrix (Q) ---
def create_Q_ca(q_val):
    Q_1d = np.array([[dt**5/20, dt**4/8, dt**3/6],
                     [dt**4/8,  dt**3/3, dt**2/2],
                     [dt**3/6,  dt**2/2, dt   ]])
    return q_val * block_diag(Q_1d, Q_1d)

q_val = 0.1
Q = create_Q_ca(q_val)

# --- Measurement Model ---
r = 20.0
R = r**2 * np.eye(2)
H = np.zeros((2, n_dim)); H[0, 0] = 1; H[1, 1] = 1

# --- True Trajectory Generation ---
true_states = np.zeros((timesteps, n_dim))
true_states[0] = [0, 0, 15, 0, 0, 0]
turn_rate = np.deg2rad(6)
acceleration_vector = np.array([2.0, 1.0])

def get_F_ct_true(omega): 
    sin_omega_dt, cos_omega_dt = np.sin(omega*dt), np.cos(omega*dt)
    F_4d = np.array([[1, 0, sin_omega_dt/omega, -(1-cos_omega_dt)/omega],
                     [0, 1, (1-cos_omega_dt)/omega, sin_omega_dt/omega],
                     [0, 0, cos_omega_dt, -sin_omega_dt],
                     [0, 0, sin_omega_dt, cos_omega_dt]])
    return block_diag(F_4d, np.eye(2))

for t in range(1, timesteps):
    x_prev = true_states[t-1].copy()
    if t == 30: x_prev[4:] = acceleration_vector
    elif t == 60: x_prev[4:] = 0.0
    F_true = get_F_ct_true(turn_rate) if 60 <= t < 90 else F
    true_states[t] = F_true @ x_prev

measurements = np.zeros((timesteps, 2))
for t in range(timesteps):
    measurements[t] = H @ true_states[t] + np.sqrt(R) @ np.random.randn(2)

# --- Kalman Filter Initialization & Main Loop ---
num_init_points = 5
if timesteps < num_init_points:
    raise ValueError("Not enough measurements to initialize.")

init_measurements = measurements[0:num_init_points]
init_time_axis = np.arange(num_init_points) * dt
vx, x0 = np.polyfit(init_time_axis, init_measurements[:, 0], 1)
vy, y0 = np.polyfit(init_time_axis, init_measurements[:, 1], 1)
initial_pos_x = vx * init_time_axis[-1] + x0
initial_pos_y = vy * init_time_axis[-1] + y0

x_kf = np.array([initial_pos_x, initial_pos_y, vx, vy, 0, 0])
P_kf = np.diag([r**2, r**2, 50.0, 50.0, 10.0, 10.0])

kf_estimates = np.zeros((timesteps, n_dim))
for t in range(num_init_points):
    kf_estimates[t, 0] = vx * (t*dt) + x0
    kf_estimates[t, 1] = vy * (t*dt) + y0
    kf_estimates[t, 2:] = x_kf[2:]

# Main Kalman Filter loop
for t in range(num_init_points, timesteps):
    # --- Prediction Step ---
    x_pred = F @ x_kf
    P_pred = F @ P_kf @ F.T + Q

    # --- Update Step ---
    z = measurements[t]
    y = z - H @ x_pred  
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    x_kf = x_pred + K @ y
    P_kf = (np.eye(n_dim) - K @ H) @ P_pred

    kf_estimates[t] = x_kf

# --- Error Calculation (RMSE) ---
start_idx = num_init_points
errors = true_states[start_idx:] - kf_estimates[start_idx:]

# Position RMSE
pos_errors = errors[:, 0:2]
pos_rmse = np.sqrt(np.mean(pos_errors**2))

# Velocity RMSE
vel_errors = errors[:, 2:4]
vel_rmse = np.sqrt(np.mean(vel_errors**2))

print("--- Kalman Filter (CA Model) Performance ---")
print(f"Position RMSE: {pos_rmse:.3f} m")
print(f"Velocity RMSE: {vel_rmse:.3f} m/s")
print("------------------------------------------")

# --- Visualization ---
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax1 = plt.subplots(figsize=(14, 8))
fig.suptitle(f'Simple Kalman Filter (CA Model) Tracking (r = {r})', fontsize=16)

ax1.plot(true_states[:, 0], true_states[:, 1], 'g-', label='True Trajectory', linewidth=3, alpha=0.8)
ax1.plot(measurements[:, 0], measurements[:, 1], 'x', color='gray', label='Measurements', markersize=5, alpha=0.6)
ax1.plot(kf_estimates[:, 0], kf_estimates[:, 1], 'b--', label='KF Estimate', linewidth=2)
ax1.set_xlabel('X Position (m)')
ax1.set_ylabel('Y Position (m)')
ax1.set_title('Object Trajectory Tracking')
ax1.legend()
ax1.axis('equal')
ax1.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()