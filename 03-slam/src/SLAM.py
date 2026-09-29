#!/usr/bin/env python3


import rospy
import numpy as np
import math
import tf.transformations as tf_trans
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import Range
from geometry_msgs.msg import PoseArray, Pose, Quaternion, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
import os
import random
import time
import threading

# ====== MAP & SENSOR PARAMETERS ======
MAP_RESOLUTION = 0.02  # meters / cell
MAP_WIDTH = 200         # cells (8m)
MAP_HEIGHT = 200        # cells (8m)
MAP_ORIGIN_X = None
MAP_ORIGIN_Y = None

MAX_LASER_RANGE = 0.4    # m (sensor max)
LASER_OFFSET = 0.043     # same offset you used earlier

# Log-odds params
LOG_ODDS_OCC = 1.7
LOG_ODDS_FREE = -0.2
LOG_ODDS_MIN = -6.0
LOG_ODDS_MAX = 6.0

# Hit threshold: require N hits before treating cell as stable occupied
MIN_HITS_FOR_OCC = 2

# Smoothing / cleanup
SMOOTH_EVERY_UPDATES = 50   # apply smoothing every N laser updates
SMOOTH_KERNEL = np.array([[0.02, 0.05, 0.02],
                          [0.05, 0.7, 0.05],
                          [0.02, 0.05, 0.02]])
SMOOTH_KERNEL = SMOOTH_KERNEL / SMOOTH_KERNEL.sum()

# Wall consistency parameters
WALL_CONSISTENCY_THRESHOLD = 0.7
NEIGHBOR_OCCUPIED_THRESHOLD = 3

# Particle filter parameters
INITIAL_PARTICLES = 50
MIN_PARTICLES = 5
CONVERGED_PARTICLES = 20
FINAL_PARTICLES = 5
LINEAR_STD = 0.02
ANGULAR_STD = 0.05
LASER_STD = 0.05
MIN_TRANS_THRESH = 0.01
MIN_ROT_THRESH = 0.02

# Downsampling parameters
CONVERGENCE_THRESHOLD_1 = 0.5  # meters - downsample to 50 when particles within this radius
CONVERGENCE_THRESHOLD_2 = 0.3  # meters - downsample to 20 when particles within this radius
MIN_SAMPLES_FOR_CONVERGENCE = 10  # minimum laser readings before considering downsampling
CONVERGENCE_CHECK_INTERVAL = 20   # check convergence every N laser updates

# Map fusion parameters
TOP_PARTICLES_FOR_FUSION = 5  # Use top N particles for map fusion
FUSION_WEIGHT_THRESHOLD = 0.1  # Only consider particles with weight > threshold

# Sensor model parameters
SENSOR_MODEL_MAX_RANGE = MAX_LASER_RANGE
SENSOR_MODEL_STD = 0.05
SENSOR_MODEL_HIT_PROB = 0.6
SENSOR_MODEL_SHORT_PROB = 0.1
SENSOR_MODEL_MAX_PROB = 0.95

# Global state
particles = None
weights = None
last_odom = None
current_odom = None
latest_laser = None
new_odom = False
new_laser = False
localization_start_time = None
laser_update_count = 0
downsampling_stage = 0  # 0: initial, 1: first downsample, 2: final downsample

# ====== MAPPING UTILITIES ======

def world_to_map(world_x, world_y):
    """Convert world coordinates to map indices"""
    mx = int((world_x - MAP_ORIGIN_X) / MAP_RESOLUTION)
    my = int((world_y - MAP_ORIGIN_Y) / MAP_RESOLUTION)
    return mx, my

def map_to_world(mx, my):
    """Convert map indices to world coordinates"""
    wx = MAP_ORIGIN_X + (mx + 0.5) * MAP_RESOLUTION
    wy = MAP_ORIGIN_Y + (my + 0.5) * MAP_RESOLUTION
    return wx, wy

def is_valid_cell(mx, my):
    """Check if map coordinates are within bounds"""
    return 0 <= mx < MAP_WIDTH and 0 <= my < MAP_HEIGHT

def bresenham(x0, y0, x1, y1):
    """Integer Bresenham line between two map cells"""
    x0 = int(x0)
    y0 = int(y0)
    x1 = int(x1)
    y1 = int(y1)

    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            cells.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            cells.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    cells.append((x1, y1))
    return cells

def is_wall_cell(log_odds_value):
    """Check if a cell is likely part of a wall"""
    prob = 1.0 - 1.0 / (1.0 + math.exp(log_odds_value))
    return prob > WALL_CONSISTENCY_THRESHOLD

def count_occupied_neighbors(log_odds_map, x, y):
    """Count the number of occupied neighbors for a cell"""
    count = 0
    height, width = log_odds_map.shape
    
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                if is_wall_cell(log_odds_map[ny, nx]):
                    count += 1
    return count

def preserve_wall_consistency(log_odds_map, region):
    """Ensure wall consistency by preserving cells with enough occupied neighbors"""
    if not region:
        return log_odds_map
    
    xs = [c[0] for c in region]
    ys = [c[1] for c in region]
    min_x = max(min(xs) - 2, 0)
    max_x = min(max(xs) + 2, MAP_WIDTH - 1)
    min_y = max(min(ys) - 2, 0)
    max_y = min(max(ys) + 2, MAP_HEIGHT - 1)
    
    result = log_odds_map.copy()
    
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if is_wall_cell(log_odds_map[y, x]):
                occupied_neighbors = count_occupied_neighbors(log_odds_map, x, y)
                
                if occupied_neighbors >= NEIGHBOR_OCCUPIED_THRESHOLD:
                    result[y, x] = log_odds_map[y, x]
                else:
                    result[y, x] = log_odds_map[y, x] * 0.8
    
    return result

# ====== CONVERGENCE AND DOWNSAMPLING ======

def calculate_particle_spread(particles):
    """Calculate the spatial spread of particles"""
    if len(particles) == 0:
        return float('inf')
    
    positions = np.array([[p.x, p.y] for p in particles])
    centroid = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - centroid, axis=1)
    return np.max(distances)

def check_convergence_and_downsample(particles, weights):
    """Check if particles have converged and downsample if appropriate"""
    global downsampling_stage, laser_update_count
    
    if laser_update_count < MIN_SAMPLES_FOR_CONVERGENCE:
        return particles, weights, False
    
    if laser_update_count % CONVERGENCE_CHECK_INTERVAL != 0:
        return particles, weights, False
    
    spread = calculate_particle_spread(particles)
    current_count = len(particles)
    
    # Stage 0: Initial -> 50 particles
    if downsampling_stage == 0 and spread < CONVERGENCE_THRESHOLD_1 and current_count > CONVERGED_PARTICLES:
        rospy.loginfo("Convergence detected (spread: %.3fm). Downsampling from %d to %d particles", 
                     spread, current_count, CONVERGED_PARTICLES)
        particles, weights = intelligent_downsample(particles, weights, CONVERGED_PARTICLES)
        downsampling_stage = 1
        return particles, weights, True
    
    # Stage 1: 50 -> 20 particles  
    elif downsampling_stage == 1 and spread < CONVERGENCE_THRESHOLD_2 and current_count > FINAL_PARTICLES:
        rospy.loginfo("High convergence detected (spread: %.3fm). Final downsampling from %d to %d particles", 
                     spread, current_count, FINAL_PARTICLES)
        particles, weights = intelligent_downsample(particles, weights, FINAL_PARTICLES)
        downsampling_stage = 2
        return particles, weights, True
    
    return particles, weights, False

def intelligent_downsample(particles, weights, target_count):
    """Intelligently downsample particles preserving diversity and quality"""
    if len(particles) <= target_count:
        return particles, weights
    
    # Sort particles by weight (descending)
    sorted_indices = np.argsort(weights)[::-1]
    
    # Always keep the best particles (top 30%)
    keep_best = max(1, int(target_count * 0.3))
    selected_indices = sorted_indices[:keep_best].tolist()
    
    # For remaining slots, use diversity-based selection
    remaining_slots = target_count - keep_best
    remaining_indices = sorted_indices[keep_best:]
    
    if remaining_slots > 0 and len(remaining_indices) > 0:
        # Group remaining particles by spatial proximity
        remaining_particles = [particles[i] for i in remaining_indices]
        remaining_weights = weights[remaining_indices]
        
        # Select diverse particles from remaining set
        diverse_indices = select_diverse_particles(remaining_particles, remaining_weights, remaining_slots)
        selected_indices.extend([remaining_indices[i] for i in diverse_indices])
    
    # Create new particle set
    new_particles = [particles[i] for i in selected_indices]
    new_weights = weights[selected_indices]
    new_weights = new_weights / np.sum(new_weights)  # Renormalize
    
    return new_particles, new_weights

def select_diverse_particles(particles, weights, count):
    """Select diverse particles to maintain spatial coverage"""
    if count >= len(particles):
        return list(range(len(particles)))
    
    selected = []
    positions = np.array([[p.x, p.y] for p in particles])
    
    # Start with highest weighted particle
    selected.append(np.argmax(weights))
    
    for _ in range(count - 1):
        if len(selected) >= len(particles):
            break
            
        best_idx = -1
        best_score = -1
        
        for i in range(len(particles)):
            if i in selected:
                continue
            
            # Calculate minimum distance to already selected particles
            min_dist = float('inf')
            for sel_idx in selected:
                dist = np.linalg.norm(positions[i] - positions[sel_idx])
                min_dist = min(min_dist, dist)
            
            # Score combines diversity (distance) and quality (weight)
            score = min_dist * 0.7 + weights[i] * 0.3
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        if best_idx != -1:
            selected.append(best_idx)
    
    return selected

# ====== MAP FUSION ======

def fuse_particle_maps(particles, weights):
    """Fuse maps from top particles weighted by their probability"""
    if len(particles) == 0:
        return np.zeros((MAP_HEIGHT, MAP_WIDTH)), np.zeros((MAP_HEIGHT, MAP_WIDTH))
    
    # Get top particles for fusion
    sorted_indices = np.argsort(weights)[::-1]
    fusion_count = min(TOP_PARTICLES_FOR_FUSION, len(particles))
    
    # Only use particles above weight threshold
    valid_particles = []
    valid_weights = []
    
    for i in range(fusion_count):
        idx = sorted_indices[i]
        if weights[idx] > FUSION_WEIGHT_THRESHOLD:
            valid_particles.append(particles[idx])
            valid_weights.append(weights[idx])
    
    if not valid_particles:
        # Fallback to best particle if no valid ones
        best_particle = particles[sorted_indices[0]]
        with best_particle.map_lock:
            return best_particle.log_odds.copy(), best_particle.hit_count.copy()
    
    # Normalize weights for fusion
    valid_weights = np.array(valid_weights)
    valid_weights = valid_weights / np.sum(valid_weights)
    
    rospy.logdebug("Fusing maps from %d particles with weights: %s", 
                  len(valid_particles), [f"{w:.3f}" for w in valid_weights])
    
    # Initialize fused maps
    fused_log_odds = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.float32)
    fused_hit_count = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.int16)
    
    # Weighted fusion of log-odds
    for particle, weight in zip(valid_particles, valid_weights):
        with particle.map_lock:
            fused_log_odds += weight * particle.log_odds
            fused_hit_count += (weight * particle.hit_count).astype(np.int16)
    
    return fused_log_odds, fused_hit_count

class FusedMap:
    """Manages the globally fused map from multiple particles"""
    def __init__(self):
        self.log_odds = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.float32)
        self.hit_count = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.int16)
        self.map = np.full((MAP_WIDTH, MAP_HEIGHT), 0.5, dtype=np.float32)
        self.lock = threading.Lock()
        self.last_fusion_time = time.time()
    
    def update_from_particles(self, particles, weights):
        """Update fused map from particle ensemble"""
        current_time = time.time()
        if current_time - self.last_fusion_time < 1.0:  # Rate limit fusion
            return
        
        fused_log_odds, fused_hit_count = fuse_particle_maps(particles, weights)
        
        with self.lock:
            self.log_odds = fused_log_odds
            self.hit_count = fused_hit_count
            self.map = 1.0 / (1.0 + np.exp(-self.log_odds))
        
        self.last_fusion_time = current_time
        rospy.logdebug("Updated fused map from %d particles", len(particles))

# ====== PARTICLE CLASS WITH ADVANCED MAPPING ======

class SLAMParticle:
    """Individual particle for SLAM with advanced mapping capabilities"""
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.log_odds = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.float32)
        self.hit_count = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.int16)
        self.map = np.full((MAP_HEIGHT, MAP_WIDTH), 0.5, dtype=np.float32)
        
        self.laser_update_counter = 0
        self.smooth_cells = []
        self.map_lock = threading.Lock()
        self.smooth_event = threading.Event()
        
        self.smoothing_thread = threading.Thread(target=self._apply_smoothing)
        self.smoothing_thread.daemon = True
        self.smoothing_thread.start()
    
    def get_pose(self):
        return (self.x, self.y, self.theta)
    
    def set_pose(self, x, y, theta):
        self.x = x
        self.y = y  
        self.theta = theta
    
    def _apply_smoothing(self):
        """Background thread for applying smoothing to the map"""
        while not rospy.is_shutdown():
            self.smooth_event.wait()
            self.smooth_event.clear()
            
            with self.map_lock:
                if not self.smooth_cells:
                    continue
                cells_to_process = self.smooth_cells.copy()
                self.smooth_cells = []
            
            if not cells_to_process:
                continue
                
            with self.map_lock:
                self.log_odds = preserve_wall_consistency(self.log_odds, cells_to_process)
                
            xs = [c[0] for c in cells_to_process]
            ys = [c[1] for c in cells_to_process]
            min_x = max(min(xs) - 2, 0)
            max_x = min(max(xs) + 2, MAP_WIDTH - 1)
            min_y = max(min(ys) - 2, 0)
            max_y = min(max(ys) + 2, MAP_HEIGHT - 1)

            with self.map_lock:
                sub = self.log_odds[min_y:max_y+1, min_x:max_x+1].copy()
            
            if sub.size == 0:
                continue

            padded = np.pad(sub, ((1,1),(1,1)), mode='edge')
            out = np.copy(sub)
            kh, kw = SMOOTH_KERNEL.shape
            for iy in range(sub.shape[0]):
                for ix in range(sub.shape[1]):
                    if is_wall_cell(sub[iy, ix]) and count_occupied_neighbors(sub, ix, iy) >= NEIGHBOR_OCCUPIED_THRESHOLD:
                        out[iy, ix] = sub[iy, ix]
                    else:
                        patch = padded[iy:iy+kh, ix:ix+kw]
                        out[iy, ix] = np.sum(patch * SMOOTH_KERNEL)

            with self.map_lock:
                self.log_odds[min_y:max_y+1, min_x:max_x+1] = out
                self.map = 1.0 / (1.0 + np.exp(-self.log_odds))
    
    def update_map_with_laser(self, raw_range):
        """Advanced mapping update from second code"""
        if raw_range is None or raw_range < 0.0:
            return {'updated_cells': 0}

        r = raw_range + LASER_OFFSET
        if r < 0.0:
            return {'updated_cells': 0}
        if r > MAX_LASER_RANGE:
            r = MAX_LASER_RANGE
            hit_obstacle = False
        else:
            hit_obstacle = True

        robot_x, robot_y, robot_yaw = self.x, self.y, self.theta

        end_x = robot_x + r * math.cos(robot_yaw)
        end_y = robot_y + r * math.sin(robot_yaw)

        robot_mx, robot_my = world_to_map(robot_x, robot_y)
        end_mx, end_my = world_to_map(end_x, end_y)

        if not is_valid_cell(robot_mx, robot_my):
            return {'updated_cells': 0}

        cells = bresenham(robot_mx, robot_my, end_mx, end_my)
        
        with self.map_lock:
            self.smooth_cells.extend(cells)
            if len(self.smooth_cells) > 1000:
                self.smooth_cells = self.smooth_cells[-1000:]

        interior = cells[:-1]
        for (mx, my) in interior:
            if not is_valid_cell(mx, my):
                continue
            with self.map_lock:
                self.log_odds[my, mx] += LOG_ODDS_FREE
                if self.log_odds[my, mx] < LOG_ODDS_MIN:
                    self.log_odds[my, mx] = LOG_ODDS_MIN

        if hit_obstacle and is_valid_cell(end_mx, end_my):
            with self.map_lock:
                self.hit_count[end_my, end_mx] += 1
                if self.hit_count[end_my, end_mx] >= MIN_HITS_FOR_OCC:
                    self.log_odds[end_my, end_mx] += LOG_ODDS_OCC
                    if self.log_odds[end_my, end_mx] > LOG_ODDS_MAX:
                        self.log_odds[end_my, end_mx] = LOG_ODDS_MAX

        self.laser_update_counter += 1
        if self.laser_update_counter % SMOOTH_EVERY_UPDATES == 0:
            self.smooth_event.set()

        with self.map_lock:
            self.map = 1.0 / (1.0 + np.exp(-self.log_odds))

        return {
            'updated_cells': len(cells),
            'robot_map_pos': (robot_mx, robot_my),
            'obstacle_map_pos': (end_mx, end_my),
            'processed_range': r
        }
    
    def raycast(self, max_range=MAX_LASER_RANGE):
        """Raycast on the particle's map to predict laser reading"""
        step_size = MAP_RESOLUTION * 0.5
        distance = 0.0
        
        while distance < max_range:
            test_x = self.x + distance * math.cos(self.theta)
            test_y = self.y + distance * math.sin(self.theta)
            mx, my = world_to_map(test_x, test_y)
            
            if not is_valid_cell(mx, my):
                return max_range
                
            with self.map_lock:
                if self.log_odds[my, mx] >= 1.0:  # Occupied
                    return distance
                    
            distance += step_size
            
        return max_range

# ====== PARTICLE FILTER FUNCTIONS ======

def initialize_slam_particles(num_particles, robot_initial_pose):
    """Initialize particles around robot's starting area"""
    global MAP_ORIGIN_X, MAP_ORIGIN_Y
    
    particles = []
    robot_x, robot_y, robot_theta = robot_initial_pose
    
    map_half_width = (MAP_WIDTH * MAP_RESOLUTION) / 2.0
    map_half_height = (MAP_HEIGHT * MAP_RESOLUTION) / 2.0
    
    MAP_ORIGIN_X = robot_x - map_half_width
    MAP_ORIGIN_Y = robot_y - map_half_height
    
    rospy.loginfo("Map origin set to (%.2f, %.2f)", MAP_ORIGIN_X, MAP_ORIGIN_Y)
    
    particle_spread = 1.0
    for i in range(num_particles):
        x = robot_x + np.random.uniform(-particle_spread, particle_spread)
        y = robot_y + np.random.uniform(-particle_spread, particle_spread)
        theta = robot_theta + np.random.uniform(-np.pi/3, np.pi/3)
        
        particle = SLAMParticle(x, y, theta)
        particles.append(particle)
    
    weights = np.ones(len(particles), dtype=np.float32) / len(particles)
    
    return particles, weights

def sensor_model(particle, z_actual):
    """Calculate particle weight based on sensor reading"""
    z_predicted = particle.raycast()
    
    if z_actual >= MAX_LASER_RANGE and z_predicted >= MAX_LASER_RANGE:
        return SENSOR_MODEL_MAX_PROB
    
    error = abs(z_actual - z_predicted)
    prob = np.exp(-(error ** 2) / (2 * SENSOR_MODEL_STD ** 2))
    
    return max(prob, 0.01)  # Ensure non-zero probability

def update_weights(particles, weights, laser_reading):
    """Update particle weights based on laser reading"""
    if laser_reading is None:
        return weights
    
    for i, particle in enumerate(particles):
        weights[i] = sensor_model(particle, laser_reading)
    
    if np.sum(weights) > 0:
        weights /= np.sum(weights)
    else:
        weights = np.ones(len(weights)) / len(weights)
    
    return weights

def effective_particles(weights):
    """Calculate effective number of particles"""
    return 1.0 / np.sum(weights ** 2)

def systematic_resample(particles, weights):
    """Systematic resampling of particles"""
    n = len(particles)
    indices = []
    c = np.cumsum(weights)
    u0 = np.random.uniform(0, 1/n)
    
    i = 0
    for j in range(n):
        uj = u0 + j/n
        while uj > c[i] and i < n-1:
            i += 1
        indices.append(i)
    
    new_particles = []
    for idx in indices:
        old_particle = particles[idx]
        new_particle = SLAMParticle(old_particle.x, old_particle.y, old_particle.theta)
        with new_particle.map_lock, old_particle.map_lock:
            new_particle.log_odds = old_particle.log_odds.copy()
            new_particle.hit_count = old_particle.hit_count.copy()
            new_particle.map = old_particle.map.copy()
        new_particles.append(new_particle)
    
    new_weights = np.ones(n) / n
    return new_particles, new_weights

def apply_motion_model(particles, v, w, dt):
    """Apply motion model to all particles"""
    for particle in particles:
        x, y, theta = particle.get_pose()
        
        v_noisy = v + np.random.normal(0, LINEAR_STD)
        w_noisy = w + np.random.normal(0, ANGULAR_STD)
        
        if abs(w_noisy) < 1e-5:
            x_new = x + v_noisy * np.cos(theta) * dt
            y_new = y + v_noisy * np.sin(theta) * dt
            theta_new = theta
        else:
            radius = v_noisy / w_noisy
            center_x = x - radius * np.sin(theta)
            center_y = y + radius * np.cos(theta)
            theta_new = theta + w_noisy * dt
            x_new = center_x + radius * np.sin(theta_new)
            y_new = center_y - radius * np.cos(theta_new)
        
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))
        particle.set_pose(x_new, y_new, theta_new)
    
    return particles

# ====== VISUALIZATION FUNCTIONS ======

def publish_particles(pub, particles, weights=None):
    """Publish particles for visualization"""
    marker_array = MarkerArray()
    
    clear_marker = Marker()
    clear_marker.header.stamp = rospy.Time.now()
    clear_marker.header.frame_id = "map"
    clear_marker.action = Marker.DELETEALL
    marker_array.markers.append(clear_marker)
    
    for i, particle in enumerate(particles):
        x, y, theta = particle.get_pose()
        
        marker = Marker()
        marker.header = Header(stamp=rospy.Time.now(), frame_id='map')
        marker.ns = "slam_particles"
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
        if weights is not None and i < len(weights):
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

def publish_slam_map(pub, fused_map):
    """Publish the fused map estimate"""
    map_msg = OccupancyGrid()
    map_msg.header.stamp = rospy.Time.now()
    map_msg.header.frame_id = "map"
    
    map_msg.info.resolution = MAP_RESOLUTION
    map_msg.info.width = MAP_WIDTH
    map_msg.info.height = MAP_HEIGHT
    map_msg.info.origin.position.x = MAP_ORIGIN_X
    map_msg.info.origin.position.y = MAP_ORIGIN_Y
    map_msg.info.origin.position.z = 0.0
    map_msg.info.origin.orientation.w = 1.0
    
    with fused_map.lock:
        flat_log_odds = fused_map.log_odds.flatten()
    
    occ = np.full(flat_log_odds.shape, -1, dtype=np.int8)
    occ[flat_log_odds >= 1.0] = 100
    occ[flat_log_odds <= -1.0] = 0
    
    map_msg.data = occ.tolist()
    pub.publish(map_msg)

# ====== ROS CALLBACKS ======

def odom_cb(msg):
    global current_odom, new_odom
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    _, _, yaw = tf_trans.euler_from_quaternion([q.x, q.y, q.z, q.w])
    current_odom = (p.x, p.y, yaw)
    new_odom = True

def laser_cb(msg):
    global latest_laser, new_laser, laser_update_count
    latest_laser = msg.range
    new_laser = True
    laser_update_count += 1

# ====== MAIN FUNCTION ======

def main():
    global particles, weights, last_odom, current_odom, new_odom, new_laser
    global localization_start_time, downsampling_stage
    
    rospy.init_node('slam_advanced')
    localization_start_time = time.time()
    
    rospy.Subscriber('/odom', Odometry, odom_cb)
    rospy.Subscriber('/vector/laser', Range, laser_cb)
    particle_pub = rospy.Publisher('/particle_markers', MarkerArray, queue_size=1)
    map_pub = rospy.Publisher('/map', OccupancyGrid, queue_size=1, latch=True)
    
    # Initialize fused map manager
    fused_map = FusedMap()
    
    rospy.loginfo("Waiting for initial odometry...")
    while current_odom is None and not rospy.is_shutdown():
        rospy.sleep(0.1)
    
    robot_initial_pose = current_odom
    rospy.loginfo("Robot initial position: (%.2f, %.2f, %.1fdeg)", 
                  robot_initial_pose[0], robot_initial_pose[1], np.degrees(robot_initial_pose[2]))
    
    particles, weights = initialize_slam_particles(INITIAL_PARTICLES, robot_initial_pose)
    last_odom = current_odom

    rospy.loginfo("SLAM initialized with %d particles", INITIAL_PARTICLES)
    
    last_update_time = rospy.Time.now()
    last_diagnostic_time = rospy.Time.now()
    last_map_stats_time = time.time()
    
    rate = rospy.Rate(10)
    
    while not rospy.is_shutdown():
        current_time = rospy.Time.now()
        
        # Handle odometry updates
        if new_odom and last_odom is not None:
            dx = current_odom[0] - last_odom[0]
            dy = current_odom[1] - last_odom[1]
            dtheta = current_odom[2] - last_odom[2]
            
            while dtheta > np.pi:
                dtheta -= 2 * np.pi
            while dtheta < -np.pi:
                dtheta += 2 * np.pi
            
            if abs(dx) > MIN_TRANS_THRESH or abs(dy) > MIN_TRANS_THRESH or abs(dtheta) > MIN_ROT_THRESH:
                dt = (current_time - last_update_time).to_sec()
                if dt < 1e-5:
                    dt = 0.1
                
                theta_prev = last_odom[2]
                delta_x_robot = np.cos(-theta_prev) * dx - np.sin(-theta_prev) * dy
                delta_y_robot = np.sin(-theta_prev) * dx + np.cos(-theta_prev) * dy
                v = np.sqrt(delta_x_robot**2 + delta_y_robot**2) / dt
                w = dtheta / dt
                
                if delta_x_robot < 0:
                    v = -v
                
                particles = apply_motion_model(particles, v, w, dt)
                last_odom = current_odom
                
                rospy.logdebug("Motion update: v=%.3f m/s, w=%.3f rad/s", v, w)
            
            new_odom = False
            last_update_time = current_time
        
        # Handle laser updates
        if new_laser and latest_laser is not None:
            weights = update_weights(particles, weights, latest_laser)
            
            total_updated_cells = 0
            for particle in particles:
                update_result = particle.update_map_with_laser(latest_laser)
                total_updated_cells += update_result['updated_cells']
            
            # Check for convergence and downsample if needed
            particles, weights, downsampled = check_convergence_and_downsample(particles, weights)
            
            # Regular resampling check
            n_eff = effective_particles(weights)
            if n_eff < MIN_PARTICLES:
                particles, weights = systematic_resample(particles, weights)
                rospy.loginfo("Resampled particles. Effective particles: %.1f", n_eff)
            
            # Update fused map
            fused_map.update_from_particles(particles, weights)
            
            new_laser = False
        
        # Periodic diagnostics and publishing
        if (current_time - last_diagnostic_time).to_sec() > 2.0:
            best_idx = np.argmax(weights)
            best_particle = particles[best_idx]
            best_x, best_y, best_theta = best_particle.get_pose()
            
            particle_spread = calculate_particle_spread(particles)
            n_eff = effective_particles(weights)
            
            stage_names = ["Initial", "Converged", "Final"]
            stage_name = stage_names[min(downsampling_stage, 2)]
            
            rospy.loginfo("SLAM [%s]: Best at (%.2f, %.2f, %.1fdeg) | Particles: %d | Spread: %.3fm | N_eff: %.1f", 
                         stage_name, best_x, best_y, np.degrees(best_theta), 
                         len(particles), particle_spread, n_eff)
            
            publish_slam_map(map_pub, fused_map)
            last_diagnostic_time = current_time
        
        # Map statistics
        if time.time() - last_map_stats_time > 5.0:
            with fused_map.lock:
                flat_log_odds = fused_map.log_odds.flatten()
            
            unknown_cells = np.sum((flat_log_odds > -1.0) & (flat_log_odds < 1.0))
            free_cells = np.sum(flat_log_odds <= -1.0)
            occupied_cells = np.sum(flat_log_odds >= 1.0)
            total_cells = MAP_WIDTH * MAP_HEIGHT
            
            explored_percentage = ((total_cells - unknown_cells) / total_cells) * 100
            
            rospy.loginfo("Fused Map: %.1f%% explored | Free: %d | Occupied: %d | Unknown: %d", 
                         explored_percentage, free_cells, occupied_cells, unknown_cells)
            
            last_map_stats_time = time.time()
        
        publish_particles(particle_pub, particles, weights)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("SLAM node interrupted")