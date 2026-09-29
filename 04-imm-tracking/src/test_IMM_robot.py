import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import pandas as pd
import io

# --- Load Data from CSV ---
# In a real-world scenario, you would load from a file path like:
# df = pd.read_csv('your_data.csv')
# For this example, we'll use the provided data in a string.
df = pd.read_csv('robot_position_log.csv')

# --- Simulation Parameters from Data ---
np.random.seed(42)
dt = np.mean(np.diff(df['timestamp'])) # Calculate average timestep
timesteps = len(df)
n_dim = 6

# Use Odom Data for measurements and true trajectory
measurements = df[['odom_x', 'odom_y']].values
true_positions = df[['true_x', 'true_y']].values

# Reconstruct full true state for error calculation
true_states = np.zeros((timesteps, n_dim))
true_states[:, 0:2] = true_positions
# Estimate true velocity and acceleration from position data
true_states[1:, 2:4] = np.diff(true_positions, axis=0) / dt
true_states[2:, 4:6] = np.diff(true_states[1:, 2:4], axis=0) / dt


# --- Motion Models ---
F_ca = np.eye(n_dim)
F_ca[0, 2] = dt; F_ca[1, 3] = dt
F_ca[2, 4] = dt; F_ca[3, 5] = dt
F_ca[0, 4] = 0.5 * dt**2; F_ca[1, 5] = 0.5 * dt**2

# --- Process Noise Matrices (Q) ---
def create_Q_ca(q_val):
    Q_1d = np.array([[dt**5/20, dt**4/8, dt**3/6],
                     [dt**4/8,  dt**3/3, dt**2/2],
                     [dt**3/6,  dt**2/2, dt   ]])
    return q_val * block_diag(Q_1d, Q_1d)

q_ca = 0.1
Q_ca = create_Q_ca(q_ca)

# The Coordinated Turn Model
def get_F_ct(omega):
    if np.abs(omega) < 1e-8:
        F_4d = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    else:
        sin_omega_dt, cos_omega_dt = np.sin(omega*dt), np.cos(omega*dt)
        F_4d = np.array([[1, 0, sin_omega_dt/omega, -(1-cos_omega_dt)/omega],
                         [0, 1, (1-cos_omega_dt)/omega, sin_omega_dt/omega],
                         [0, 0, cos_omega_dt, -sin_omega_dt],
                         [0, 0, sin_omega_dt, cos_omega_dt]])
    return block_diag(F_4d, np.eye(2))

q_ct_pos_vel = 0.005
q_ct_accel = 0.2
Q_ct_4d = q_ct_pos_vel * block_diag(np.eye(2)*dt**3/3, np.eye(2)*dt)
Q_ct_accel_part = q_ct_accel * np.eye(2)
Q_ct = block_diag(Q_ct_4d, Q_ct_accel_part)

# --- Measurement Model ---
r = 0.2 # Measurement noise standard deviation
R = r**2 * np.eye(2)
H = np.zeros((2, n_dim)); H[0, 0] = 1; H[1, 1] = 1

# --- IMM Filter Implementation (3-Model Version) ---
turn_rate_filter = np.deg2rad(5)
models = [{'F': F_ca, 'Q': Q_ca, 'name': 'CA'},
          {'F': get_F_ct(turn_rate_filter), 'Q': Q_ct, 'name': 'CT Left'},
          {'F': get_F_ct(-turn_rate_filter), 'Q': Q_ct, 'name': 'CT Right'}]
num_models = len(models)

T = np.array([[0.98, 0.01, 0.01],
              [0.05, 0.94, 0.01],
              [0.05, 0.01, 0.94]])

# --- IMM Initialization & Main Loop ---
num_init_points = 5
if timesteps < num_init_points:
    raise ValueError("Not enough measurements to initialize.")

init_measurements = measurements[0:num_init_points]
init_time_axis = np.arange(num_init_points) * dt
vx, x0 = np.polyfit(init_time_axis, init_measurements[:, 0], 1)
vy, y0 = np.polyfit(init_time_axis, init_measurements[:, 1], 1)

x_imm = np.array([x0, y0, vx, vy, 0, 0])
P_imm = np.diag([r**2, r**2, 50.0, 50.0, 10.0, 10.0])
mu = np.array([0.95, 0.025, 0.025]) # Initial model probabilities

x_models = np.tile(x_imm, (num_models, 1))
P_models = np.tile(P_imm, (num_models, 1, 1))
imm_estimates = np.zeros((timesteps, n_dim))
model_probabilities = np.zeros((timesteps, num_models))

# Populate initial estimates
for t in range(num_init_points):
    imm_estimates[t, 0] = vx * (t*dt) + x0
    imm_estimates[t, 1] = vy * (t*dt) + y0
    imm_estimates[t, 2:] = x_imm[2:]
    model_probabilities[t] = mu

# Main IMM Loop
for t in range(num_init_points, timesteps):
    # --- Mixing Step ---
    c_bar = T.T @ mu
    mixed_states = np.zeros((num_models, n_dim))
    mixed_covariances = np.zeros((num_models, n_dim, n_dim))
    for j in range(num_models):
        if c_bar[j] < 1e-9: c_bar[j] = 1e-9
        mu_ij = (T[:, j] * mu) / c_bar[j]
        x_mix_j = np.sum([mu_ij[i] * x_models[i] for i in range(num_models)], axis=0)
        mixed_states[j] = x_mix_j
        P_mix_j = np.sum([mu_ij[i] * (P_models[i] + np.outer(x_models[i] - x_mix_j, x_models[i] - x_mix_j)) for i in range(num_models)], axis=0)
        mixed_covariances[j] = P_mix_j

    # --- Filtering Step ---
    z = measurements[t]
    likelihoods = np.zeros(num_models)
    for j in range(num_models):
        F, Q = models[j]['F'], models[j]['Q']
        x_pred_j = F @ mixed_states[j]
        P_pred_j = F @ mixed_covariances[j] @ F.T + Q
        y_j = z - H @ x_pred_j
        S_j = H @ P_pred_j @ H.T + R
        try:
            S_j_inv = np.linalg.inv(S_j)
        except np.linalg.LinAlgError:
            S_j_inv = np.linalg.pinv(S_j) # Use pseudo-inverse if singular
        K_j = P_pred_j @ H.T @ S_j_inv
        x_models[j] = x_pred_j + K_j @ y_j
        P_models[j] = (np.eye(n_dim) - K_j @ H) @ P_pred_j
        
        det_S = np.linalg.det(2 * np.pi * S_j)
        if det_S <= 0:
            likelihoods[j] = 1e-9 # Assign a small likelihood if determinant is non-positive
            continue
        likelihoods[j] = (1.0/np.sqrt(det_S)) * np.exp(-0.5 * y_j.T @ S_j_inv @ y_j)

    # --- Update Step ---
    mu_new = likelihoods * c_bar
    if np.sum(mu_new) < 1e-9:
        mu = np.ones(num_models) / num_models # Re-initialize if probabilities collapse
    else:
        mu = mu_new / np.sum(mu_new)
    model_probabilities[t] = mu

    x_imm = np.sum([mu[j] * x_models[j] for j in range(num_models)], axis=0)
    imm_estimates[t] = x_imm

# --- Error Calculation (RMSE) ---
start_idx = num_init_points
errors = true_states[start_idx:] - imm_estimates[start_idx:]
pos_errors = errors[:, 0:2]
pos_rmse = np.sqrt(np.mean(pos_errors**2))
vel_errors = errors[:, 2:4]
vel_rmse = np.sqrt(np.mean(vel_errors**2))

print("--- Filter Performance ---")
print(f"Position RMSE: {pos_rmse:.3f} m")
print(f"Velocity RMSE: {vel_rmse:.3f} m/s")
print("------------------------")

# --- Visualization ---
plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle(f'3-Model IMM with Odom Data (r = {r})', fontsize=16)
ax1.plot(true_states[:, 0], true_states[:, 1], 'g-', label='True Trajectory', linewidth=3, alpha=0.8)
ax1.plot(measurements[:, 0], measurements[:, 1], 'x', color='gray', label='Odom Measurements', markersize=5, alpha=0.6)
ax1.plot(imm_estimates[:, 0], imm_estimates[:, 1], 'r--', label='IMM Estimate', linewidth=2)
ax1.set_xlabel('X Position (m)'); ax1.set_ylabel('Y Position (m)'); ax1.set_title('Object Trajectory Tracking from CSV Data')
ax1.legend(); ax1.axis('equal')
time_axis = df['timestamp'].values
for i in range(num_models): ax2.plot(time_axis, model_probabilities[:, i], label=f'P({models[i]["name"]})')
ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Model Probability'); ax2.set_title('Model Probabilities Over Time')
ax2.legend(loc='upper left', bbox_to_anchor=(1, 1)); ax2.set_ylim([0, 1]); ax2.grid(True)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()