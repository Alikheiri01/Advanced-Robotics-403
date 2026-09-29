#!/usr/bin/env python
#0.866799
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import GetModelState, GetModelStateRequest
import time
import math
import csv
import os

class MotionAndRecord:
    """
    A class to control a robot's motion through a predefined sequence while
    simultaneously recording its noisy odometry and exact Gazebo position.
    The recorded data is saved to a CSV file upon completion.
    """
    def __init__(self):
        # --- Initialize ROS Node ---
        rospy.init_node('motion_and_record_controller', anonymous=True)

        # --- Class Members ---
        self.robot_name = 'robot'
        self.output_file_name = 'robot_position_log.csv'
        self._latest_odom = None
        self._data_log = []

        # --- ROS Communication ---
        self.velocity_publisher = rospy.Publisher('/vector/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/odom', Odometry, self._odom_callback)
        
        rospy.loginfo("Waiting for '/gazebo/get_model_state' service...")
        rospy.wait_for_service('/gazebo/get_model_state')
        self._get_model_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        rospy.loginfo("Service is available.")

        # Register shutdown hook to save data and stop the robot
        rospy.on_shutdown(self.shutdown)

    def _odom_callback(self, msg):
        """Callback function to store the latest odometry data."""
        self._latest_odom = msg

    def run_sequence(self):
        """
        Executes the main motion and recording sequence.
        """
        # --- Wait for ROS Time to be Initialized ---
        while rospy.Time.now().to_sec() == 0.0:
            rospy.loginfo("Waiting for ROS time to be initialized...")
            time.sleep(0.1)
        
        # --- Motion Parameters ---
        constant_velocity = 0.5
        constant_acceleration = 0.05
        angular_velocity_deg = 10.0 # Using the 10 deg/s that worked
        angular_velocity_rad = math.radians(angular_velocity_deg)
        
        duration_phase_1 = 10.0
        duration_phase_2 = 10.0
        duration_phase_3 = 10.0
        
        turn_linear_velocity = constant_velocity + (constant_acceleration * duration_phase_2)
        
        publish_rate = 10
        rate = rospy.Rate(publish_rate)
        
        twist = Twist()
        model_state_request = GetModelStateRequest(model_name=self.robot_name, relative_entity_name='world')

        rospy.loginfo("Starting robot motion and recording sequence...")
        start_time = rospy.Time.now().to_sec()
        
        phase1_logged, phase2_logged, phase3_logged = False, False, False

        # --- Main Loop ---
        while not rospy.is_shutdown():
            current_time = rospy.Time.now().to_sec()
            elapsed_time = current_time - start_time

            # --- 1. RECORD DATA ---
            if self._latest_odom is not None:
                try:
                    response = self._get_model_state_service(model_state_request)
                    if response.success:
                        odom_x = self._latest_odom.pose.pose.position.x
                        odom_y = self._latest_odom.pose.pose.position.y
                        true_x = response.pose.position.x
                        true_y = response.pose.position.y
                        self._data_log.append([rospy.get_time(), odom_x, odom_y, true_x, true_y])
                except rospy.ServiceException as e:
                    rospy.logwarn_throttle(5, f"Service call failed: {e}")
            else:
                rospy.logwarn_throttle(2, "Waiting for first /odom message to start recording...")

            # --- 2. CONTROL MOTION ---
            if elapsed_time < duration_phase_1:
                if not phase1_logged:
                    rospy.loginfo(f"Phase 1: Constant Velocity at {constant_velocity} m/s")
                    phase1_logged = True
                twist.linear.x = constant_velocity
                twist.angular.z = 0.0

            elif elapsed_time < duration_phase_1 + duration_phase_2:
                if not phase2_logged:
                    rospy.loginfo("Phase 2: Constant Acceleration")
                    phase2_logged = True
                time_in_phase_2 = elapsed_time - duration_phase_1
                twist.linear.x = constant_velocity + (constant_acceleration * time_in_phase_2)
                twist.angular.z = 0.0

            elif elapsed_time < duration_phase_1 + duration_phase_2 + duration_phase_3:
                if not phase3_logged:
                    rospy.loginfo(f"Phase 3: Coordinated Turn at {turn_linear_velocity:.2f} m/s and {angular_velocity_deg} deg/s")
                    phase3_logged = True
                twist.linear.x = turn_linear_velocity
                twist.angular.z = angular_velocity_rad
            else:
                rospy.loginfo("Motion sequence complete.")
                break # Exit loop

            self.velocity_publisher.publish(twist)
            rate.sleep()
        
        # This part is reached when the loop breaks after 30 seconds
        self.shutdown()

    def save_data_to_csv(self):
        """Saves the recorded data to a CSV file in the user's home directory."""
        if not self._data_log:
            rospy.loginfo("No data was recorded.")
            return

        file_path = os.path.join(os.path.expanduser('~'), self.output_file_name)
        rospy.loginfo(f"Saving {len(self._data_log)} data points to {file_path}...")
        
        try:
            with open(file_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                header = ['timestamp', 'odom_x', 'odom_y', 'true_x', 'true_y']
                csv_writer.writerow(header)
                csv_writer.writerows(self._data_log)
            rospy.loginfo("Successfully saved data.")
        except IOError as e:
            rospy.logerr(f"Could not write to file {file_path}. Error: {e}")

    def shutdown(self):
        """Called on script termination."""
        rospy.loginfo("Shutting down. Stopping robot and saving data.")
        
        # Stop the robot
        self.velocity_publisher.publish(Twist())
        time.sleep(0.5) # Give it a moment to send the stop command

        # Save the data
        self.save_data_to_csv()

if __name__ == '__main__':
    try:
        controller = MotionAndRecord()
        controller.run_sequence()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted.")
    except Exception as e:
        rospy.logerr(f"An unhandled exception occurred: {e}")

