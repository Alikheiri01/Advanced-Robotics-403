import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

# --- Simulation Parameters ---
np.random.seed(42)
dt = 1.0
total_time = 120
timesteps = int(total_time / dt)
n_dim = 6

# --- Motion Models ---
F_ca = np.eye(n_dim)
F_ca[0, 2] = dt; F_ca[1, 3] = dt
F_ca[2, 4] = dt; F_ca[3, 5] = dt
F_ca[0, 4] = 0.5 * dt**2; F_ca[1, 5] = 0.5 * dt**2
print(F_ca)
# --- Process Noise Matrices (Q) ---
# The "Generalist" CA Model
def create_Q_ca(q_val):
    Q_1d = np.array([[dt**5/20, dt**4/8, dt**3/6],
                     [dt**4/8,  dt**3/3, dt**2/2],
                     [dt**3/6,  dt**2/2, dt   ]])
    return q_val * block_diag(Q_1d, Q_1d)

q_ca = 0.2
Q_ca = create_Q_ca(q_ca)

# The "Specialist" CV Model - Redesigned to be hostile to acceleration
q_cv_vel = 0.01  
q_cv_accel = 1e-8 
Q_cv = np.zeros((n_dim, n_dim))
Q_cv[2:4, 2:4] = q_cv_vel * np.eye(2) * dt
Q_cv[4:6, 4:6] = q_cv_accel * np.eye(2) * dt

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
r = 20.0
R = r**2 * np.eye(2)

# --- True Trajectory Generation ---
true_states = np.zeros((timesteps, n_dim))
true_states[0] = [0, 0, 15, 0, 0, 0]
turn_rate = np.deg2rad(6)
acceleration_vector = np.array([2.0, 1.0])

for t in range(1, timesteps):
    x_prev = true_states[t-1].copy()
    if t == 30: x_prev[4:] = acceleration_vector
    elif t == 60: x_prev[4:] = 0.0
    F_matrix = get_F_ct(turn_rate) if 60 <= t < 90 else F_ca
    true_states[t] = F_matrix @ x_prev

measurements = np.zeros((timesteps, 2))
H = np.zeros((2, n_dim)); H[0, 0] = 1; H[1, 1] = 1
for t in range(timesteps):
    measurements[t] = H @ true_states[t] + np.sqrt(R) @ np.random.randn(2)

# --- IMM Filter Implementation (4-Model Version) ---
turn_rate_filter = np.deg2rad(6)
models = [{'F': F_ca, 'Q': Q_cv, 'name': 'CV'},
          {'F': F_ca, 'Q': Q_ca, 'name': 'CA'},
          {'F': get_F_ct(turn_rate_filter), 'Q': Q_ct, 'name': 'CT Left'},
          {'F': get_F_ct(-turn_rate_filter), 'Q': Q_ct, 'name': 'CT Right'}]
num_models = len(models)

# Moderately "stubborn" T matrix
T = np.array([[0.97, 0.01, 0.01, 0.01],
              [0.02, 0.96, 0.01, 0.01],
              [0.02, 0.01, 0.96, 0.01],
              [0.02, 0.01, 0.01, 0.96]])

# --- IMM Initialization & Main Loop ---
x_imm = np.zeros(n_dim)
x_imm[0] = measurements[0, 0]
x_imm[1] = measurements[0, 1]
P_imm = np.eye(n_dim) * 10
mu = np.array([0.7, 0.1, 0.1, 0.1])
x_models = np.tile(x_imm, (num_models, 1))
P_models = np.tile(P_imm, (num_models, 1, 1))
imm_estimates = np.zeros((timesteps, n_dim))
model_probabilities = np.zeros((timesteps, num_models))
model_probabilities[0] = mu

for t in range(timesteps):
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

    z = measurements[t]
    likelihoods = np.zeros(num_models)
    for j in range(num_models):
        F, Q = models[j]['F'], models[j]['Q']
        x_pred_j = F @ mixed_states[j]
        P_pred_j = F @ mixed_covariances[j] @ F.T + Q
        y_j = z - H @ x_pred_j
        S_j = H @ P_pred_j @ H.T + R
        try: S_j_inv = np.linalg.inv(S_j)
        except np.linalg.LinAlgError: S_j_inv = np.linalg.pinv(S_j)
        K_j = P_pred_j @ H.T @ S_j_inv
        x_models[j] = x_pred_j + K_j @ y_j
        P_models[j] = (np.eye(n_dim) - K_j @ H) @ P_pred_j
        det_S = np.linalg.det(2 * np.pi * S_j)
        if det_S <= 0: likelihoods[j] = 1e-9; continue
        likelihoods[j] = (1.0/np.sqrt(det_S)) * np.exp(-0.5*y_j.T@S_j_inv@y_j)

    mu_new = likelihoods * c_bar
    if np.sum(mu_new) < 1e-9: mu = np.ones(num_models) / num_models
    else: mu = mu_new / np.sum(mu_new)
    model_probabilities[t] = mu

    x_imm = np.sum([mu[j] * x_models[j] for j in range(num_models)], axis=0)
    imm_estimates[t] = x_imm

# --- Visualization ---
plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle(f'4-Model IMM with Specialized CV Model (r = {r})', fontsize=16)
ax1.plot(true_states[:, 0], true_states[:, 1], 'g-', label='True Trajectory', linewidth=3, alpha=0.8)
ax1.plot(measurements[:, 0], measurements[:, 1], 'x', color='gray', label='Measurements', markersize=5, alpha=0.6)
ax1.plot(imm_estimates[:, 0], imm_estimates[:, 1], 'r--', label='IMM Estimate', linewidth=2)
ax1.set_xlabel('X Position (m)'); ax1.set_ylabel('Y Position (m)'); ax1.set_title('Object Trajectory Tracking')
ax1.legend(); ax1.axis('equal')
time_axis = np.arange(timesteps) * dt
for i in range(num_models): ax2.plot(time_axis, model_probabilities[:, i], label=f'P({models[i]["name"]})')
ax2.axvspan(30, 60, color='blue', alpha=0.1, label='CA Phase')
ax2.axvspan(60, 90, color='green', alpha=0.1, label='CT Phase')
ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Model Probability'); ax2.set_title('Model Probabilities Over Time')
ax2.legend(loc='upper left', bbox_to_anchor=(1, 1)); ax2.set_ylim([0, 1]); ax2.grid(True)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()