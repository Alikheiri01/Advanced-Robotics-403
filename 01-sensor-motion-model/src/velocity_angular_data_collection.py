#!/usr/bin/env python3

import rospy
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Point, Quaternion, Twist
import tf.transformations 
import math
import csv
import time

# --- Configuration ---
ROBOT_MODEL_NAME = "robot"
CMD_VEL_TOPIC = "/vector/cmd_vel"     
SET_STATE_SERVICE = '/gazebo/set_model_state'
GET_STATE_SERVICE = '/gazebo/get_model_state'
NUM_ITERATIONS = 20
CORRECTION_COEF = 2.03
ROTATION_ANGLE_DEG = 90.0 
ANGULAR_SPEED_DEG_PER_SEC = 30.0 
PUBLISH_RATE_HZ = 50 
CSV_FILE_NAME = 'robot_rotation_twist_estimated_angle_data.csv'
SERVICE_WAIT_TIMEOUT = 10.0 
STATE_SETTLE_TIME = 0.5
STOP_WAIT_TIME = 0.5 

# --- Initial Pose (Zero Location and Zero Rotation) ---
initial_pose = Pose()
initial_pose.position = Point(0.0, 0.0, 0.0)
initial_pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0) 
def quaternion_to_yaw(orientation_quat):
   
    euler = tf.transformations.euler_from_quaternion(
        [orientation_quat.x, orientation_quat.y, orientation_quat.z, orientation_quat.w]
    )
    return euler[2] 

def main():
    rospy.init_node('robot_rotation_experiment_twist_estimated')
    rospy.loginfo("Robot Rotation Experiment (Twist Control with Estimated Angle) Node Started.")
    # --- Wait for Gazebo Services ---
    try:
        rospy.wait_for_service(SET_STATE_SERVICE, timeout=SERVICE_WAIT_TIMEOUT)
        rospy.wait_for_service(GET_STATE_SERVICE, timeout=SERVICE_WAIT_TIMEOUT)
        set_state_proxy = rospy.ServiceProxy(SET_STATE_SERVICE, SetModelState)
        get_state_proxy = rospy.ServiceProxy(GET_STATE_SERVICE, GetModelState)
        rospy.loginfo("Gazebo services found.")
    except rospy.ROSException as e:
        rospy.logfatal("Failed to find Gazebo services: %s", e)
        return

    # --- Setup Twist Publisher ---
    try:
        velocity_publisher = rospy.Publisher(CMD_VEL_TOPIC, Twist, queue_size=10)
        rospy.loginfo(f"Publisher created for {CMD_VEL_TOPIC}")
    except rospy.ROSException as e:
         rospy.logfatal(f"Failed to create publisher for {CMD_VEL_TOPIC}: {e}")
         return

    # Allow publisher to connect
    rospy.sleep(1.0)

    # --- Data Storage ---
    experiment_data = []
    experiment_data.append(['Iteration', 'Final_X', 'Final_Y', 'Final_Yaw_Rad', 'Final_Yaw_Deg']) # CSV Header

    # --- Rotation Calculation Setup ---
    target_angle_rad = math.radians(ROTATION_ANGLE_DEG)*CORRECTION_COEF
    angular_speed_rad_per_sec = math.radians(ANGULAR_SPEED_DEG_PER_SEC)

    CCW = 1
    angular_vel_z = angular_speed_rad_per_sec * CCW

  
    vel_msg = Twist()
    vel_msg.angular.z = angular_vel_z

    stop_vel_msg = Twist() 

    rate = rospy.Rate(PUBLISH_RATE_HZ)

    for i in range(NUM_ITERATIONS):
        if rospy.is_shutdown():
            break

        rospy.loginfo(f"--- Starting Iteration {i+1}/{NUM_ITERATIONS} ---")

        # Reset robot to initial state (0, 0, 0 location, zero rotation)
        rospy.loginfo("Resetting robot to initial pose (0,0,0, identity quaternion)...")
        model_state_reset = ModelState()
        model_state_reset.model_name = ROBOT_MODEL_NAME
        model_state_reset.pose = initial_pose
        model_state_reset.twist = Twist() 
        model_state_reset.reference_frame = "world"

        try:
            set_state_resp = set_state_proxy(model_state_reset)
            if not set_state_resp.success:
                 rospy.logwarn(f"Failed to set state for reset: {set_state_resp.status_message}")
            else:
                 rospy.loginfo("Robot reset successfully.")
        except rospy.ServiceException as e:
            rospy.logerr("SetModelState service call failed during reset: %s", e)
            continue 

        # Allow simulation to settle after reset
        rospy.sleep(STATE_SETTLE_TIME)

        #rotate robot
        start_time = rospy.Time.now()
        current_angle_estimated = 0.0 

        while abs(current_angle_estimated) < abs(target_angle_rad) and not rospy.is_shutdown():
            velocity_publisher.publish(vel_msg)
            rate.sleep() 
            elapsed_time = (rospy.Time.now().to_sec() - start_time.to_sec())
            current_angle_estimated = angular_vel_z * elapsed_time 
            
        rospy.loginfo("Estimated target angle reached. Stopping robot...")
        velocity_publisher.publish(stop_vel_msg) 
       
        rospy.sleep(STOP_WAIT_TIME)

        #get current state
        rospy.loginfo("Getting robot's final pose...")
        try:
            get_state_resp = get_state_proxy(ROBOT_MODEL_NAME, "world")
            if get_state_resp.success:
                final_pose = get_state_resp.pose
                final_x = final_pose.position.x
                final_y = final_pose.position.y
                final_yaw_rad = quaternion_to_yaw(final_pose.orientation)
                final_yaw_deg = math.degrees(final_yaw_rad)

                print(f"Measured Final Pose: X={final_x:.3f}, Y={final_y:.3f}, Yaw (Rad)={final_yaw_rad:.3f}, Yaw (Deg)={final_yaw_deg:.3f}")

                experiment_data.append([i+1, final_x, final_y, final_yaw_rad, final_yaw_deg])

            else:
                rospy.logwarn(f"Failed to get state: {get_state_resp.status_message}")
                experiment_data.append([i+1, 'Get State Error', 'Get State Error', 'Get State Error', 'Get State Error'])

        except rospy.ServiceException as e:
            rospy.logerr("GetModelState service call failed: %s", e)
            experiment_data.append([i+1, 'Service Error', 'Service Error', 'Service Error', 'Service Error'])
            continue

        rospy.loginfo(f"--- Iteration {i+1} Complete ---")
        print("\n")

    rospy.loginfo(f"Experiment finished. Saving data to {CSV_FILE_NAME}...")
    try:
        with open(CSV_FILE_NAME, mode='w') as file:
            writer = csv.writer(file)
            writer.writerows(experiment_data)
        rospy.loginfo(f"Data successfully saved to {CSV_FILE_NAME}")
    except IOError as e:
        rospy.logerr(f"Failed to save data to CSV: {e}")

    rospy.loginfo("Robot Rotation Experiment (Twist Control with Estimated Angle) Node Finished.")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Robot Rotation Experiment Interrupted.")