#!/usr/bin/env python3
import rospy
import numpy as np
import tf.transformations as tf_trans
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range
from geometry_msgs.msg import PoseArray, Pose, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
import os
import math
import random

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
NUM_PARTICLES = 500
LINEAR_STD = 0.00    # No motion noise for this test
ANGULAR_STD = 0.00   # No motion noise for this test
LASER_STD = 0.05     # meters for sensor model

# Particle elimination parameters
HIGH_ERROR_THRESHOLD = 0.1  # meters (particles with error > this are candidates for elimination)
ELIMINATION_COUNT = 5       # Number of consecutive high errors needed for elimination
WALL_PENALTY = 3            # Penalty points for being in wall

# Global state
particles = None
last_odom = None
current_odom = None
latest_laser = None
new_odom = False
new_laser = False
particle_errors = []  # Track consecutive errors for each particle
particle_penalties = []  # Track penalties for each particle

def world_to_map(x, y):
    """Convert world coordinates to map coordinates"""
    map_x = int((x - origin_x) / MAP_RESOLUTION)
    map_y = map_height - 1 - int((y - origin_y) / MAP_RESOLUTION)
    return map_x, map_y

def is_free(x, y):
    """Check if a position is in free space"""
    map_x, map_y = world_to_map(x, y)
    
    # Boundary check
    if 0 <= map_x < map_width and 0 <= map_y < map_height:
        return binary_map[map_y, map_x] == 1  # 1 is free space
    return False

def publish_particles(pub, particles, errors=None):
    marker_array = MarkerArray()
    
    # Clear previous markers
    clear_marker = Marker()
    clear_marker.header.stamp = rospy.Time.now()
    clear_marker.header.frame_id = "map"
    clear_marker.action = Marker.DELETEALL
    marker_array.markers.append(clear_marker)
    
    for i, p in enumerate(particles):
        x, y, theta = p
        
        # Create marker for particle
        marker = Marker()
        marker.header = Header(stamp=rospy.Time.now(), frame_id='map')
        marker.ns = "particles"
        marker.id = i
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # Position and orientation
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        quat = tf_trans.quaternion_from_euler(0, 0, theta)
        marker.pose.orientation = Quaternion(*quat)
        
        # Size and color
        marker.scale.x = 0.05  # Length of arrow
        marker.scale.y = 0.01  # Width of arrow
        marker.scale.z = 0.01  # Height of arrow
        
        # Color based on error
        color = ColorRGBA()
        if errors is not None and i < len(errors):
            # Normalize error between 0 and 1
            norm_error = min(errors[i] / max_laser_range, 1.0)
            color.r = norm_error  # Red component: higher error
            color.g = 1.0 - norm_error  # Green component: lower error
            color.b = 0.0
            color.a = 0.8  # Semi-transparent
        else:
            color.r = 0.0
            color.g = 0.0
            color.b = 1.0
            color.a = 0.5
            
        marker.color = color
        marker_array.markers.append(marker)
    
    pub.publish(marker_array)

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
    """Simulate laser reading for a particle at (x, y, theta)"""
    step_size = MAP_RESOLUTION * 5  # Larger step for efficiency
    distance = 0.0

    while distance < max_laser_range:
        test_x = x + distance * np.cos(theta)
        test_y = y + distance * np.sin(theta)
        
        # Convert to map coordinates
        map_x, map_y = world_to_map(test_x, test_y)

        # Boundary check
        if 0 <= map_x < map_width and 0 <= map_y < map_height:
            if binary_map[map_y, map_x] == 0:  # Occupied cell
                return distance
        else:
            return max_laser_range
            
        distance += step_size
    return max_laser_range

def motion_update(delta_x_robot, delta_y_robot, dtheta):
    """Update particle positions based on robot motion"""
    global particles, particle_errors, particle_penalties
    
    new_particles = []
    new_errors = []
    new_penalties = []
    
    for i, (x, y, theta) in enumerate(particles):
        # Add Gaussian noise to motion deltas
        noisy_dx = delta_x_robot + np.random.normal(0, LINEAR_STD)
        noisy_dy = delta_y_robot + np.random.normal(0, LINEAR_STD)
        noisy_dtheta = dtheta + np.random.normal(0, ANGULAR_STD)
        
        # Apply motion in particle's frame
        new_x = x + noisy_dx * np.cos(theta) - noisy_dy * np.sin(theta)
        new_y = y + noisy_dx * np.sin(theta) + noisy_dy * np.cos(theta)
        new_theta = theta + noisy_dtheta
        
        # Normalize angle
        new_theta = np.arctan2(np.sin(new_theta), np.cos(new_theta))
        
        # Check if new position is in wall
        if not is_free(new_x, new_y):
            # Apply penalty for being in wall
            penalty = particle_penalties[i] + WALL_PENALTY if i < len(particle_penalties) else WALL_PENALTY
            # Keep original position but add orientation noise
            noisy_theta = theta + np.random.normal(0, ANGULAR_STD * 2)
            noisy_theta = np.arctan2(np.sin(noisy_theta), np.cos(noisy_theta))
            new_particles.append((x, y, noisy_theta))
            new_penalties.append(penalty)
            if i < len(particle_errors):
                new_errors.append(particle_errors[i])
        else:
            new_particles.append((new_x, new_y, new_theta))
            new_penalties.append(particle_penalties[i] if i < len(particle_penalties) else 0)
            if i < len(particle_errors):
                new_errors.append(particle_errors[i])
    
    particles = new_particles
    particle_penalties = new_penalties
    if particle_errors:
        particle_errors = new_errors

def compute_errors(robot_r):
    """Compute sensor error for each particle and update error tracking"""
    global particle_errors, particle_penalties
    
    errors = []
    new_error_counts = []
    
    # Preprocess real laser reading
    if robot_r < 0.39 and robot_r > 0.043:
        robot_r -= 0.043
    elif robot_r <= 0.043:
        robot_r = 0
    
    for i, (x, y, theta) in enumerate(particles):
        # Skip particles in walls
        if not is_free(x, y):
            errors.append(max_laser_range)  # Max error
            new_error_counts.append(particle_errors[i] + 1 if i < len(particle_errors) else 1)
            continue
            
        # Get simulated laser reading
        sim_r = raycast(x, y, theta)
        
        # Apply same processing to simulated reading
        if sim_r < 0.39 and sim_r > 0.043:
            sim_r -= 0.043
        elif sim_r <= 0.043:
            sim_r = 0
        
        # Calculate absolute error
        error = abs(robot_r - sim_r)
        errors.append(error)
        
        # Update error count
        if i < len(particle_errors):
            if error > HIGH_ERROR_THRESHOLD:
                new_error_counts.append(particle_errors[i] + 1)
            else:
                new_error_counts.append(0)
        else:
            new_error_counts.append(1 if error > HIGH_ERROR_THRESHOLD else 0)
    
    particle_errors = new_error_counts
    return errors

def eliminate_bad_particles():
    """Remove particles with too many consecutive errors or wall penalties"""
    global particles, particle_errors, particle_penalties
    
    if not particles or not particle_errors:
        return
    
    # Identify particles to eliminate
    to_remove = []
    for i in range(len(particles)):
        # Check for consecutive high errors
        if particle_errors[i] >= ELIMINATION_COUNT:
            to_remove.append(i)
        # Check for wall penalties
        elif i < len(particle_penalties) and particle_penalties[i] >= WALL_PENALTY * 2:
            to_remove.append(i)
    
    # Remove in reverse order to preserve indices
    for i in sorted(to_remove, reverse=True):
        particles.pop(i)
        if i < len(particle_errors):
            particle_errors.pop(i)
        if i < len(particle_penalties):
            particle_penalties.pop(i)
    
    return len(to_remove)

def regenerate_particles():
    """Create new particles around the robot's position to maintain count"""
    global particles, particle_errors, particle_penalties
    
    if current_odom is None:
        return
    
    num_needed = NUM_PARTICLES - len(particles)
    if num_needed <= 0:
        return
    
    rospy.loginfo("Regenerating %d particles", num_needed)
    
    for _ in range(num_needed):
        # Generate around robot position with some noise
        x = current_odom[0] + np.random.normal(0, 0.05)  # 5cm noise
        y = current_odom[1] + np.random.normal(0, 0.05)  # 5cm noise
        theta = current_odom[2] + np.random.normal(0, 0.1)  # 0.1 rad noise
        theta = np.arctan2(np.sin(theta), np.cos(theta))  # normalize
        
        particles.append((x, y, theta))
        particle_errors.append(0)
        particle_penalties.append(0)

def initialize_particles():
    """Initialize particles randomly across free space in the map"""
    particles = []
    free_cells = np.argwhere(binary_map == 1)
    sample_indices = np.random.choice(len(free_cells), NUM_PARTICLES)
    sampled_cells = free_cells[sample_indices]
    
    for cell in sampled_cells:
        i, j = cell
        x = origin_x + j * MAP_RESOLUTION
        y = origin_y + (map_height - 1 - i) * MAP_RESOLUTION  # Adjust for map orientation
        theta = np.random.uniform(-np.pi, np.pi)
        particles.append((x, y, theta))
    
    # Initialize tracking arrays
    global particle_errors, particle_penalties
    particle_errors = [0] * len(particles)
    particle_penalties = [0] * len(particles)
    
    return particles

def main():
    global particles, last_odom, new_odom, new_laser, particle_errors, particle_penalties
    
    rospy.init_node('sensor_model_test')
    
    # Setup publishers/subscribers
    rospy.Subscriber('/odom', Odometry, odom_cb)
    rospy.Subscriber('/vector/laser', Range, laser_cb)
    particle_pub = rospy.Publisher('/particle_markers', MarkerArray, queue_size=1)
    
    # Initialize particles randomly across the map
    particles = initialize_particles()
    rospy.loginfo("Initialized %d particles", len(particles))
    
    # Wait for initial odometry
    rospy.loginfo("Waiting for initial odometry...")
    while current_odom is None and not rospy.is_shutdown():
        rospy.sleep(0.1)
    last_odom = current_odom
    
    rospy.loginfo("Sensor model test ready")
    
    rate = rospy.Rate(10)  # 10Hz
    errors = None
    
    while not rospy.is_shutdown():
        # Process new odometry (motion update)
        if new_odom and last_odom is not None:
            # Calculate odometry delta
            dx = current_odom[0] - last_odom[0]
            dy = current_odom[1] - last_odom[1]
            dtheta = current_odom[2] - last_odom[2]
            
            # Convert to robot's previous frame
            theta_prev = last_odom[2]
            delta_x_robot = np.cos(-theta_prev) * dx - np.sin(-theta_prev) * dy
            delta_y_robot = np.sin(-theta_prev) * dx + np.cos(-theta_prev) * dy
            
            # Apply motion update
            motion_update(delta_x_robot, delta_y_robot, dtheta)
            
            last_odom = current_odom
            new_odom = False
        
        # Process new laser data (sensor update)
        if new_laser and latest_laser is not None:
            errors = compute_errors(latest_laser)
            
            # Eliminate bad particles
            num_removed = eliminate_bad_particles()
            if num_removed > 0:
                rospy.loginfo("Removed %d particles with bad sensor match or wall collision", num_removed)
            
            # Regenerate particles to maintain count
            regenerate_particles()
            
            # Log min, max, and average error
            if errors:
                valid_errors = [e for e in errors if e < max_laser_range]
                if valid_errors:
                    min_err = min(valid_errors)
                    max_err = max(valid_errors)
                    avg_err = sum(valid_errors) / len(valid_errors)
                    rospy.loginfo("Sensor errors - Min: %.4f, Max: %.4f, Avg: %.4f", 
                                 min_err, max_err, avg_err)
            
            new_laser = False
        
        # Publish particles with color-coded errors
        publish_particles(particle_pub, particles, errors)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass