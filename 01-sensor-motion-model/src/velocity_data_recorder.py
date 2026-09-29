#!/usr/bin/env python3
import rospy
import csv
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
import time

class VelocityTester:
    def __init__(self):
        rospy.init_node('velocity_tester_node', anonymous=True)

        self.cmd_pub = rospy.Publisher('vector/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)

        self.current_x = 0.0
        self.got_odom = False
        self.velocities = []
        self.rate = rospy.Rate(10)

        rospy.wait_for_service('/gazebo/set_model_state')
        self.reset_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.got_odom = True

    def reset_position(self):
        state = ModelState()
        state.model_name = 'robot'  
        state.pose.position.x = 0.0
        state.pose.position.y = 0.0
        state.pose.position.z = 0.0
        state.pose.orientation.w = 1.0  # Neutral rotation
        try:
            self.reset_service(state)
            rospy.sleep(1)  # Wait for position to update
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to reset model state: {e}")

    def move_robot(self, speed=0.02, duration=7.5):
        move_cmd = Twist()
        move_cmd.linear.x = speed

        start_x = self.current_x
        self.cmd_pub.publish(move_cmd)
        rospy.sleep(duration)
        self.cmd_pub.publish(Twist())  # Stop

        end_x = self.current_x
        delta_x = end_x - start_x
        velocity = delta_x / duration
        return velocity

    def run_experiment(self, trials=50):
        rospy.loginfo("Starting velocity test...")

        while not self.got_odom:
            rospy.loginfo("Waiting for odom...")
            rospy.sleep(0.5)

        for i in range(trials):
            rospy.loginfo(f"Trial {i+1}/{trials}")
            self.reset_position()
            vel = self.move_robot()
            self.velocities.append(vel)
            rospy.loginfo(f"Measured velocity: {vel:.5f} m/s")

        self.save_to_csv()

    def save_to_csv(self):
        with open("velocities.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trial", "velocity (m/s)"])
            for i, v in enumerate(self.velocities):
                writer.writerow([i+1, v])
        rospy.loginfo("Saved results to velocities.csv")

if __name__ == '__main__':
    try:
        tester = VelocityTester()
        tester.run_experiment()
    except rospy.ROSInterruptException:
        pass
