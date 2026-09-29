#!/usr/bin/env python3
import rospy
import numpy as np
import os
import tf.transformations as tf_trans
from geometry_msgs.msg import PoseArray, Pose, Quaternion, Twist
from std_msgs.msg import Header
import time

class ParticleFilterMotion:
    def __init__(self):
        rospy.init_node('particle_filter_motion_model')
        
        # Parameters
        self.num_particles = rospy.get_param('~num_particles', 500)
        self.linear_std = rospy.get_param('~linear_std', 0.0)
        self.angular_std = rospy.get_param('~angular_std', 0.0)
        map_path = rospy.get_param('~map_path', os.path.join(os.path.dirname(__file__), '../../maps/binary_map.npy'))
        self.map_resolution = rospy.get_param('~map_resolution', 0.001)
        self.map_origin = rospy.get_param('~map_origin', [-1.25, -1.25])
        
        # Load map data
        self.map_data = np.load(map_path)
        
        # Generate initial particles in free space
        self.particles = self.generate_particles()
        
        # Motion model variables
        self.last_time = rospy.Time.now()
        self.current_vel = 0.0
        self.current_ang_vel = 0.0
        
        # Publishers and Subscribers
        self.particle_pub = rospy.Publisher('/particles', PoseArray, queue_size=1)
        rospy.Subscriber('/vector/cmd_vel', Twist, self.cmd_vel_callback)
        
        # Publish particles and update motion at fixed rate
        self.update_rate = 20  # Hz
        self.update_interval = rospy.Duration(1.0 / self.update_rate)
        self.last_update_time = rospy.Time.now()
        
        rospy.Timer(self.update_interval, self.update_motion)
        rospy.Timer(rospy.Duration(0.1), self.publish_particles)  # Publish at 10 Hz
        
        rospy.loginfo("Particle Filter Motion Model initialized")

    def generate_particles(self):
        """Generate particles only in free space (where map_data == 1)"""
        free_cells = np.argwhere(self.map_data == 1)
        sample_indices = np.random.choice(len(free_cells), self.num_particles)
        sampled_cells = free_cells[sample_indices]
        
        particles = []
        for cell in sampled_cells:
            i, j = cell
            x = self.map_origin[0] + j * self.map_resolution
            y = self.map_origin[1] + i * self.map_resolution
            theta = np.random.uniform(-np.pi, np.pi)
            particles.append([x, y, theta])
            
        return np.array(particles)

    def cmd_vel_callback(self, msg):
        """Store current velocity commands"""
        self.current_vel = msg.linear.x
        self.current_ang_vel = msg.angular.z

    def update_motion(self, event=None):
        """Update particles at fixed rate using stored velocities"""
        current_time = rospy.Time.now()
        dt = (current_time - self.last_update_time).to_sec()
        self.last_update_time = current_time
        
        if dt <= 0:
            return
            
        # Apply motion model to all particles
        self.apply_motion_model(self.current_vel, self.current_ang_vel, dt)

    def apply_motion_model(self, v, w, dt):
        """Update particles using differential drive motion model with noise"""
        # Add Gaussian noise to commands
        v_noisy = v + np.random.normal(0, self.linear_std, self.num_particles)
        w_noisy = w + np.random.normal(0, self.angular_std, self.num_particles)
        
        # Extract components
        x = self.particles[:, 0]
        y = self.particles[:, 1]
        theta = self.particles[:, 2]
        
        # For small angular velocities, use straight-line approximation
        straight_mask = np.abs(w_noisy) < 1e-5
        curved_mask = ~straight_mask
        
        # Initialize new positions and orientations
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
            # Avoid division by zero
            w_nonzero = w_noisy[curved_mask]
            
            # Calculate radius of curvature
            radius = v_noisy[curved_mask] / w_nonzero
            
            # Calculate center of curvature
            center_x = x[curved_mask] - radius * np.sin(theta[curved_mask])
            center_y = y[curved_mask] + radius * np.cos(theta[curved_mask])
            
            # Calculate new angle
            theta_new[curved_mask] = theta[curved_mask] + w_nonzero * dt
            
            # Calculate new position
            x_new[curved_mask] = center_x + radius * np.sin(theta_new[curved_mask])
            y_new[curved_mask] = center_y - radius * np.cos(theta_new[curved_mask])
        
        # Update particles
        self.particles[:, 0] = x_new
        self.particles[:, 1] = y_new
        self.particles[:, 2] = theta_new

    def publish_particles(self, event=None):
        """Publish particles as PoseArray for RViz visualization"""
        pose_array = PoseArray()
        pose_array.header = Header(stamp=rospy.Time.now(), frame_id="map")
        
        for particle in self.particles:
            pose = Pose()
            pose.position.x = particle[0]
            pose.position.y = particle[1]
            pose.position.z = 0.0
            
            # Convert yaw to quaternion
            q = tf_trans.quaternion_from_euler(0, 0, particle[2])
            pose.orientation = Quaternion(*q)
            
            pose_array.poses.append(pose)
            
        self.particle_pub.publish(pose_array)

if __name__ == '__main__':
    try:
        pf = ParticleFilterMotion()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass