#!/usr/bin/env python3
import rospy
import numpy as np
import tf.transformations as tf_trans
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range
from geometry_msgs.msg import PoseArray, Pose, Quaternion, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
import os
import math
import random
import time

# === Map parameters (from yaml) ===
MAP_RESOLUTION = 0.001        # m / pixel
MAP_ORIGIN     = np.array([-1.25, -1.25])   # lower‐left (x,y) in metres
MAP_PATH       = os.path.join(os.path.dirname(__file__), '../../maps/binary_map.npy')
binary_map = np.load(MAP_PATH)
max_laser_range = 0.4

# Map dimensions
origin_x = -1.25
origin_y = -1.25
map_height = 2500
map_width = 2500

# Particle filter parameters
INITIAL_PARTICLES = 1500
MIN_PARTICLES = 5
CONVERGED_PARTICLES = 100
FINAL_PARTICLES = 5
LINEAR_STD = 0.02
ANGULAR_STD = 0.05
LASER_STD = 0.05
MIN_TRANS_THRESH = 0.01
MIN_ROT_THRESH = 0.02
CONVERGENCE_THRESH = 0.05  # 5cm position, 5deg orientation
CONVERGENCE_TIME = 2.0     # seconds of sustained convergence
STOP_THRESH_POS = 0.02     # 2cm position error
STOP_THRESH_ANG = np.radians(2)  # 2 degrees orientation error
DIVERSITY_FACTOR_START = 0.5  # Higher for more exploration
DIVERSITY_FACTOR_END = 0.05
MAX_LOCALIZATION_TIME = 60.0  # 60 seconds timeout
EXPLORATION_THRESH = 0.3    # 30cm position error threshold for exploration

# Global state
particles = None
weights = None
last_odom = None
current_odom = None
latest_laser = None
new_odom = False
new_laser = False
convergence_start_time = None
is_converged = False
localization_start_time = None
localization_complete = False
diversity_factor = DIVERSITY_FACTOR_START
best_particle_index = -1
last_exploration_time = 0.0

def world_to_map(x, y):
    map_x = int((x - origin_x) / MAP_RESOLUTION)
    map_y = map_height - 1 - int((y - origin_y) / MAP_RESOLUTION)
    return map_x, map_y

def is_free(x, y):
    map_x, map_y = world_to_map(x, y)
    if 0 <= map_x < map_width and 0 <= map_y < map_height:
        return binary_map[map_y, map_x] == 1
    return False

def publish_particles(pub, particles, weights=None):
    marker_array = MarkerArray()
    clear_marker = Marker()
    clear_marker.header.stamp = rospy.Time.now()
    clear_marker.header.frame_id = "map"
    clear_marker.action = Marker.DELETEALL
    marker_array.markers.append(clear_marker)
    
    for i, p in enumerate(particles):
        x, y, theta = p
        marker = Marker()
        marker.header = Header(stamp=rospy.Time.now(), frame_id='map')
        marker.ns = "particles"
        marker.id = i
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        quat = tf_trans.quaternion_from_euler(0, 0, theta)
        marker.pose.orientation = Quaternion(*quat)
        marker.scale.x = 0.05
        marker.scale.y = 0.01
        marker.scale.z = 0.01
        
        color = ColorRGBA()
        if localization_complete and i == best_particle_index:
            # Gold color for best particle
            color.r = 1.0
            color.g = 0.84
            color.b = 0.0
            color.a = 1.0
            marker.scale.x = 0.08  # Make it larger
        elif weights is not None and i < len(weights):
            norm_weight = weights[i] / np.max(weights) if np.max(weights) > 0 else 0.5
            color.r = 1.0 - norm_weight
            color.g = norm_weight
            color.b = 0.0
            color.a = 0.8
        else:
            color.r = 0.0
            color.g = 0.0
            color.b = 1.0
            color.a = 0.5
        marker.color = color
        marker_array.markers.append(marker)
    
    pub.publish(marker_array)

def publish_pose_estimate(pub, x, y, theta):
    pose_msg = PoseWithCovarianceStamped()
    pose_msg.header.stamp = rospy.Time.now()
    pose_msg.header.frame_id = "map"
    pose_msg.pose.pose.position.x = x
    pose_msg.pose.pose.position.y = y
    quat = tf_trans.quaternion_from_euler(0, 0, theta)
    pose_msg.pose.pose.orientation = Quaternion(*quat)
    pose_msg.pose.covariance = [0.1]*36
    pub.publish(pose_msg)

def odom_cb(msg):
    global current_odom, new_odom
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    _, _, yaw = tf_trans.euler_from_quaternion([q.x, q.y, q.z, q.w])
    current_odom = (p.x, p.y, yaw)
    new_odom = True

def laser_cb(msg):
    global latest_laser, new_laser
    latest_laser = msg.range
    new_laser = True

def raycast(x, y, theta):
    step_size = MAP_RESOLUTION * 5
    distance = 0.0

    while distance < max_laser_range:
        test_x = x + distance * np.cos(theta)
        test_y = y + distance * np.sin(theta)
        map_x, map_y = world_to_map(test_x, test_y)

        if 0 <= map_x < map_width and 0 <= map_y < map_height:
            if binary_map[map_y, map_x] == 0:
                return distance
        else:
            return max_laser_range
            
        distance += step_size
    return max_laser_range

def apply_motion_model(particles, v, w, dt):
    # Add Gaussian noise to commands
    v_noisy = v + np.random.normal(0, LINEAR_STD, len(particles))
    w_noisy = w + np.random.normal(0, ANGULAR_STD, len(particles))
    
    particles_np = np.array(particles)
    x = particles_np[:, 0]
    y = particles_np[:, 1]
    theta = particles_np[:, 2]
    
    # For small angular velocities, use straight-line approximation
    straight_mask = np.abs(w_noisy) < 1e-5
    curved_mask = ~straight_mask
    
    x_new = np.zeros_like(x)
    y_new = np.zeros_like(y)
    theta_new = np.zeros_like(theta)
    
    # Handle straight motion
    if np.any(straight_mask):
        x_new[straight_mask] = x[straight_mask] + v_noisy[straight_mask] * np.cos(theta[straight_mask]) * dt
        y_new[straight_mask] = y[straight_mask] + v_noisy[straight_mask] * np.sin(theta[straight_mask]) * dt
        theta_new[straight_mask] = theta[straight_mask]
    
    # Handle curved motion
    if np.any(curved_mask):
        w_nonzero = w_noisy[curved_mask]
        radius = v_noisy[curved_mask] / w_nonzero
        center_x = x[curved_mask] - radius * np.sin(theta[curved_mask])
        center_y = y[curved_mask] + radius * np.cos(theta[curved_mask])
        theta_new[curved_mask] = theta[curved_mask] + w_nonzero * dt
        x_new[curved_mask] = center_x + radius * np.sin(theta_new[curved_mask])
        y_new[curved_mask] = center_y - radius * np.cos(theta_new[curved_mask])
    
    new_particles_np = np.column_stack((x_new, y_new, theta_new))
    return [tuple(p) for p in new_particles_np]

def compute_weights(robot_r):
    global best_particle_index
    
    weights = np.zeros(len(particles))
    
    # Preprocess real laser reading
    if robot_r < 0.39 and robot_r > 0.043:
        robot_r -= 0.043
    elif robot_r <= 0.043:
        robot_r = 0
    
    for i, (x, y, theta) in enumerate(particles):
        if not is_free(x, y):
            weights[i] = 0.0
            continue
            
        sim_r = raycast(x, y, theta)
        
        if sim_r < 0.39 and sim_r > 0.043:
            sim_r -= 0.043
        elif sim_r <= 0.043:
            sim_r = 0
        
        error = abs(robot_r - sim_r)
        weights[i] = np.exp(-error**2 / (2 * LASER_STD**2))
    
    total_weight = np.sum(weights)
    if total_weight > 0:
        weights /= total_weight
        best_particle_index = np.argmax(weights)
    else:
        weights = np.ones(len(particles)) / len(particles)
        best_particle_index = -1
    
    return weights

def enhanced_diversity_resample(particles, weights, diversity_factor, particle_count, pos_error):
    num_particles = len(particles)
    
    # Systematic resampling
    cumulative_sum = np.cumsum(weights)
    step = 1.0 / num_particles
    start = random.uniform(0, step)
    selection_points = [start + i*step for i in range(num_particles)]
    
    new_particles = []
    index = 0
    for point in selection_points:
        while point > cumulative_sum[index]:
            index += 1
        new_particles.append(particles[index])
    
    # Enhanced diversity preservation for large errors
    num_to_perturb = int(num_particles * diversity_factor)
    indices_to_perturb = random.sample(range(num_particles), num_to_perturb)
    
    for idx in indices_to_perturb:
        x, y, theta = new_particles[idx]
        
        # Scale perturbation based on position error
        if pos_error > EXPLORATION_THRESH and particle_count > 500:
            pos_noise = 0.25  # 25cm - very large for exploration
            ang_noise = 1.0   # ~57 degrees
        elif particle_count > 500:  # Early stage with many particles
            pos_noise = 0.15  # 15cm
            ang_noise = 0.5   # ~30 degrees
        elif particle_count > 200:
            pos_noise = 0.1   # 10cm
            ang_noise = 0.3   # ~17 degrees
        else:
            pos_noise = 0.02  # 2cm
            ang_noise = 0.05  # ~3 degrees
            
        # Apply perturbation
        new_x = x + np.random.normal(0, pos_noise)
        new_y = y + np.random.normal(0, pos_noise)
        new_theta = theta + np.random.normal(0, ang_noise)
        new_theta = np.arctan2(np.sin(new_theta), np.cos(new_theta))
        
        # Only keep if in free space
        if is_free(new_x, new_y):
            new_particles[idx] = (new_x, new_y, new_theta)
        else:
            # If in wall, try a different perturbation
            new_x = x + np.random.normal(0, pos_noise/2)
            new_y = y + np.random.normal(0, pos_noise/2)
            new_theta = theta + np.random.normal(0, ang_noise/2)
            new_theta = np.arctan2(np.sin(new_theta), np.cos(new_theta))
            if is_free(new_x, new_y):
                new_particles[idx] = (new_x, new_y, new_theta)
    
    return new_particles

def calculate_pose_estimate(particles, weights):
    particles_np = np.array(particles)
    weights_np = np.array(weights)
    
    # Weighted mean position
    mean_x = np.average(particles_np[:, 0], weights=weights_np)
    mean_y = np.average(particles_np[:, 1], weights=weights_np)
    
    # Circular mean for orientation
    sin_sum = np.sum(weights_np * np.sin(particles_np[:, 2]))
    cos_sum = np.sum(weights_np * np.cos(particles_np[:, 2]))
    mean_theta = np.arctan2(sin_sum, cos_sum)
    
    return mean_x, mean_y, mean_theta

def initialize_particles(num_particles):
    particles = []
    free_cells = np.argwhere(binary_map == 1)
    
    # If we have enough free cells, use them directly
    if len(free_cells) >= num_particles:
        sample_indices = np.random.choice(len(free_cells), num_particles)
        sampled_cells = free_cells[sample_indices]
    else:
        # If not enough free cells, use all and duplicate
        sampled_cells = free_cells
        while len(sampled_cells) < num_particles:
            extra = min(len(free_cells), num_particles - len(sampled_cells))
            sampled_cells = np.concatenate((sampled_cells, free_cells[:extra]))
    
    for cell in sampled_cells:
        i, j = cell
        x = origin_x + j * MAP_RESOLUTION
        y = origin_y + (map_height - 1 - i) * MAP_RESOLUTION
        theta = np.random.uniform(-np.pi, np.pi)
        particles.append((x, y, theta))
    
    weights = np.ones(len(particles)) / len(particles)
    return particles, weights

def check_convergence(est_pose, true_pose):
    global convergence_start_time, is_converged, diversity_factor
    
    # Calculate position and orientation errors
    pos_error = np.sqrt((est_pose[0] - true_pose[0])**2 + 
                        (est_pose[1] - true_pose[1])**2)
    orient_error = abs(est_pose[2] - true_pose[2])
    orient_error = min(orient_error, 2*np.pi - orient_error)
    
    # Check if below threshold
    if pos_error < CONVERGENCE_THRESH and orient_error < np.radians(5):
        if convergence_start_time is None:
            convergence_start_time = time.time()
        elif time.time() - convergence_start_time > CONVERGENCE_TIME:
            if not is_converged:
                rospy.loginfo("CONVERGED! Position error: %.3fm, Orientation error: %.1fdeg", 
                             pos_error, np.degrees(orient_error))
                is_converged = True
                # Reduce diversity as we converge
                diversity_factor = max(DIVERSITY_FACTOR_END, diversity_factor * 0.7)
            return True
    else:
        convergence_start_time = None
        is_converged = False
    
    return False

def aggressive_downsample(particles, weights, elapsed_time):
    """Downsample aggressively to 5 particles based on time and performance"""
    if len(particles) <= FINAL_PARTICLES:
        return particles, weights
    
    # Time-based aggressive downsampling - much faster progression
    if elapsed_time < 5.0:  # First 5 seconds
        if len(particles) > 800:
            target_count = 800
        elif len(particles) > 400:
            target_count = 400
        else:
            target_count = len(particles)
    elif elapsed_time < 10.0:  # 5-10 seconds
        if len(particles) > 200:
            target_count = 200
        elif len(particles) > 100:
            target_count = 100
        else:
            target_count = len(particles)
    elif elapsed_time < 15.0:  # 10-15 seconds
        if len(particles) > 50:
            target_count = 50
        elif len(particles) > 25:
            target_count = 25
        else:
            target_count = len(particles)
    else:  # After 15 seconds
        if len(particles) > 10:
            target_count = 10
        else:
            target_count = FINAL_PARTICLES
    
    # Don't downsample if we're already at or below target
    if len(particles) <= target_count:
        return particles, weights
    
    # Select best particles (highest weights)
    sorted_indices = np.argsort(weights)[::-1]
    selected_indices = sorted_indices[:target_count]
    
    new_particles = [particles[i] for i in selected_indices]
    new_weights = weights[selected_indices]
    new_weights /= np.sum(new_weights)
    
    rospy.loginfo("Downsampled particles from %d to %d", len(particles), target_count)
    return new_particles, new_weights

def inject_exploration_particles(particles, weights, true_pose):
    """Inject new exploration particles when error is large"""
    global last_exploration_time
    
    current_time = time.time()
    if current_time - last_exploration_time < 5.0:  # Only every 5 seconds
        return particles, weights
    
    num_new = min(100, INITIAL_PARTICLES - len(particles))
    if num_new <= 0:
        return particles, weights
    
    rospy.logwarn("Injecting %d exploration particles (high error)", num_new)
    
    # Create new particles randomly in free space
    free_cells = np.argwhere(binary_map == 1)
    if len(free_cells) == 0:
        return particles, weights
    
    sample_indices = np.random.choice(len(free_cells), num_new)
    sampled_cells = free_cells[sample_indices]
    
    for cell in sampled_cells:
        i, j = cell
        x = origin_x + j * MAP_RESOLUTION
        y = origin_y + (map_height - 1 - i) * MAP_RESOLUTION
        theta = np.random.uniform(-np.pi, np.pi)
        particles.append((x, y, theta))
        weights = np.append(weights, 1.0)  # Add weight
    
    # Renormalize weights
    weights /= np.sum(weights)
    last_exploration_time = current_time
    return particles, weights

def check_stopping_condition(pos_error, orient_error, particle_count, start_time):
    """Check if localization is complete"""
    global localization_complete
    
    # Timeout condition
    elapsed_time = time.time() - start_time
    if elapsed_time > MAX_LOCALIZATION_TIME:
        rospy.logwarn("Localization timeout after %.1f seconds", elapsed_time)
        localization_complete = True
        return True
    
    # Precision condition
    if (pos_error < STOP_THRESH_POS and 
        orient_error < STOP_THRESH_ANG and 
        particle_count <= FINAL_PARTICLES):
        localization_time = time.time() - start_time
        rospy.loginfo("\n\nLOCALIZATION COMPLETE!")
        rospy.loginfo("Final position error: %.4fm", pos_error)
        rospy.loginfo("Final orientation error: %.2fdeg", np.degrees(orient_error))
        rospy.loginfo("Final particle count: %d", particle_count)
        rospy.loginfo("Localization time: %.2f seconds\n", localization_time)
        localization_complete = True
        return True
    
    return False

def main():
    global particles, weights, last_odom, current_odom, new_odom, new_laser
    global convergence_start_time, is_converged, localization_start_time, localization_complete
    global diversity_factor, best_particle_index, last_exploration_time
    
    rospy.init_node('particle_filter_enhanced')
    localization_start_time = time.time()
    last_exploration_time = time.time()
    
    # Setup publishers/subscribers
    rospy.Subscriber('/odom', Odometry, odom_cb)
    rospy.Subscriber('/vector/laser', Range, laser_cb)
    particle_pub = rospy.Publisher('/particle_markers', MarkerArray, queue_size=1)
    pose_pub = rospy.Publisher('/estimated_pose', PoseWithCovarianceStamped, queue_size=1)
    
    # Initialize particles
    particles, weights = initialize_particles(INITIAL_PARTICLES)
    rospy.loginfo("Initialized %d particles", len(particles))
    
    # Wait for initial odometry
    while current_odom is None and not rospy.is_shutdown():
        rospy.sleep(0.1)
    last_odom = current_odom
    
    rospy.loginfo("Enhanced particle filter ready - Starting localization")
    
    last_update_time = rospy.Time.now()
    last_diagnostic_time = rospy.Time.now()
    last_downsample_time = time.time()
    
    rate = rospy.Rate(20)  # 20Hz
    
    while not rospy.is_shutdown() and not localization_complete:
        current_time = rospy.Time.now()
        current_ros_time = time.time()
        
        # Process new odometry (motion update)
        if new_odom and last_odom is not None:
            dx = current_odom[0] - last_odom[0]
            dy = current_odom[1] - last_odom[1]
            dtheta = current_odom[2] - last_odom[2]
            
            if abs(dx) > MIN_TRANS_THRESH or abs(dy) > MIN_TRANS_THRESH or abs(dtheta) > MIN_ROT_THRESH:
                dt = (current_time - last_update_time).to_sec()
                
                # Fix division by zero
                if dt < 1e-5:
                    dt = 0.05  # Default to 50ms if dt is too small
                    
                theta_prev = last_odom[2]
                delta_x_robot = np.cos(-theta_prev) * dx - np.sin(-theta_prev) * dy
                delta_y_robot = np.sin(-theta_prev) * dx + np.cos(-theta_prev) * dy
                v = np.sqrt(delta_x_robot**2 + delta_y_robot**2) / dt
                w = dtheta / dt
                
                particles = apply_motion_model(particles, v, w, dt)
                last_odom = current_odom
            
            new_odom = False
            last_update_time = current_time
        
        # Process new laser data (sensor update)
        if new_laser and latest_laser is not None:
            weights = compute_weights(latest_laser)
            new_laser = False
        
        # Calculate and publish pose estimate
        est_x, est_y, est_theta = calculate_pose_estimate(particles, weights)
        publish_pose_estimate(pose_pub, est_x, est_y, est_theta)
        
        # Diagnostic logging
        if (current_time - last_diagnostic_time).to_sec() > 0.5:
            # Calculate errors
            true_x, true_y, true_theta = current_odom
            pos_error = np.sqrt((est_x - true_x)**2 + (est_y - true_y)**2)
            orient_error = abs(est_theta - true_theta)
            orient_error = min(orient_error, 2*np.pi - orient_error)
            
            # Inject exploration particles if error is high and we have many particles
            if pos_error > EXPLORATION_THRESH and len(particles) > 500:
                particles, weights = inject_exploration_particles(particles, weights, current_odom)
            
            # Calculate effective particles for resampling decision
            effective_particles = 1.0 / np.sum(weights**2) if weights is not None else INITIAL_PARTICLES
            
            # Resample if needed (based on effective particles)
            if effective_particles < len(particles) / 2.0:
                particles = enhanced_diversity_resample(particles, weights, diversity_factor, len(particles), pos_error)
                weights = np.ones(len(particles)) / len(particles)
                rospy.logdebug("Resampled particles (Neff=%.1f, Diversity=%.2f)", 
                              effective_particles, diversity_factor)
            
            # Check convergence
            converged = check_convergence((est_x, est_y, est_theta), (true_x, true_y, true_theta))
            
            # Gradually reduce diversity factor over time
            elapsed_time = time.time() - localization_start_time
            diversity_factor = max(DIVERSITY_FACTOR_END, 
                                 DIVERSITY_FACTOR_START * np.exp(-elapsed_time/30))
            
            # Check stopping condition (with timeout)
            if check_stopping_condition(pos_error, orient_error, len(particles), localization_start_time):
                # Final update to set best particle
                weights = compute_weights(latest_laser) if latest_laser is not None else weights
                break
            
            # Log diagnostics
            rospy.loginfo("PosErr: %.3fm | AngErr: %.1fdeg | Particles: %d | Div: %.2f | Eff: %.1f", 
                         pos_error, np.degrees(orient_error), len(particles), 
                         diversity_factor, effective_particles)
            
            last_diagnostic_time = current_time
        
        # AGGRESSIVE DOWNSAMPLING - happens every 0.2 seconds regardless of convergence
        if current_ros_time - last_downsample_time > 0.2:  # Much more frequent downsampling
            elapsed_time = current_ros_time - localization_start_time
            particles, weights = aggressive_downsample(particles, weights, elapsed_time)
            last_downsample_time = current_ros_time
        
        # Publish particles
        publish_particles(particle_pub, particles, weights)
        rate.sleep()
    
    if localization_complete:
        # Final publication with gold particle
        publish_particles(particle_pub, particles, weights)
        rospy.loginfo("Localization complete - Node will continue running")
        rospy.spin()  # Keep running to show gold particle

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass