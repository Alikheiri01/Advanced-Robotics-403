#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseArray, Pose, Quaternion
from std_msgs.msg import Header
import tf.transformations as tf_trans
import os

def load_particles(file_path):
    return np.load(file_path)  # shape: (N, 3)

def particle_to_pose(particle):
    x, y, theta = particle
    quat = tf_trans.quaternion_from_euler(0, 0, theta)
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = 0.0
    pose.orientation = Quaternion(*quat)
    return pose

def main():
    rospy.init_node('particle_publisher')

    # Load particles
    file_path = os.path.join(os.path.dirname(__file__), '../../maps/initial_particles.npy')
    particles = load_particles(file_path)

    # Publisher
    pub = rospy.Publisher('/particles', PoseArray, queue_size=1)

    rate = rospy.Rate(2)  # 2 Hz update rate

    while not rospy.is_shutdown():
        pose_array = PoseArray()
        pose_array.header = Header()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = "map"

        pose_array.poses = [particle_to_pose(p) for p in particles]

        pub.publish(pose_array)
        rate.sleep()

if __name__ == '__main__':
    main()
