#!/usr/bin/env python3

import math
import rospy
import matplotlib.pyplot as plt
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

current_x = current_y = current_yaw = None
trajectory_x = []
trajectory_y = []
c = 0

def callback_method(msg):
    global current_x, current_y, current_yaw, trajectory_x, trajectory_y
    current_x = msg.pose.pose.position.x
    current_y = msg.pose.pose.position.y
    quaternion = (
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w
    )
    value = euler_from_quaternion(quaternion)
    current_yaw = value[2]

    # store trajectory
    trajectory_x.append(current_x)
    trajectory_y.append(current_y)

def mover(goal_x, goal_y):
    global current_x, current_y, c
    p_controller_linear = 0.8
    rate = rospy.Rate(40)
    speed = Twist()

    while not rospy.is_shutdown():
        if current_x is not None and current_y is not None and current_yaw is not None:
            while not rospy.is_shutdown():
                dist = math.sqrt((goal_x - current_x) ** 2 + (goal_y - current_y) ** 2)

                if dist < 0.02:
                    c += 1
                    break

                linear_speed = dist * p_controller_linear
                linear_speed = min(max(linear_speed, 0.06), 0.1)

                angle_to_goal = math.atan2(goal_y - current_y, goal_x - current_x)

                yaw = current_yaw if current_yaw >= 0 else 6.28 - abs(current_yaw)
                angle_to_goal = angle_to_goal if angle_to_goal >= 0 else 6.28 - abs(angle_to_goal)

                delta_heading = math.atan2(math.sin(angle_to_goal - yaw), math.cos(angle_to_goal - yaw))

                if abs(angle_to_goal - yaw) > 0.02:
                    angular_speed = min(max(delta_heading, -0.15), 0.15)
                else:
                    angular_speed = 0.0

                if abs(angular_speed) > 0.05:
                    linear_speed = 0.0

                speed.linear.x = linear_speed
                speed.angular.z = angular_speed
                pub.publish(speed)
                rate.sleep()
            break

    speed.linear.x = 0.0
    speed.angular.z = 0.0
    pub.publish(speed)

    if c > 0:
        print("Reached this waypoint.")
        c = 0
    else:
        rospy.signal_shutdown("kill")
        print("Keyboard interrupt!")

if __name__ == "__main__":
    rospy.init_node("mover")
    pub = rospy.Publisher('/vector/cmd_vel', Twist, queue_size=10)
    rospy.Subscriber('/odom', Odometry, callback_method)

    # --- Pre-filled parameters ---
    cell_size = 0.0025  # 0.25 cm in meters
    x_offset = 0
    y_offset = 0

    path = [(100, 70), (101, 71), (102, 72), (103, 73), (104, 74), (105, 75), (106, 76), (107, 77), (108, 78), (109, 79), (110, 80), (111, 81), (112, 82), (113, 83), (114, 84), (115, 85), (116, 86), (117, 87), (118, 88), (119, 89), (120, 90), (121, 91), (122, 92), (123, 93), (124, 94), (125, 95), (126, 96), (127, 97), (128, 98), (129, 99), (130, 100), (131, 101), (132, 102), (133, 103), (134, 104), (135, 105), (136, 106), (137, 107), (138, 108), (139, 109), (140, 110), (141, 111), (142, 112), (143, 113), (143, 114), (143, 115), (143, 116), (143, 117), (143, 118), (143, 119), (143, 120), (143, 121), (143, 122), (143, 123), (143, 124), (143, 125), (143, 126), (143, 127), (143, 128), (143, 129), (143, 130), (143, 131), (143, 132), (143, 133), (143, 134), (143, 135), (143, 136), (143, 137), (143, 138), (143, 139), (143, 140), (143, 141), (143, 142), (143, 143), (143, 144), (143, 145), (143, 146), (143, 147), (143, 148), (143, 149), (143, 150), (143, 151), (144, 152), (144, 153), (144, 154), (144, 155), (144, 156), (144, 157), (144, 158), (144, 159), (144, 160), (144, 161), (144, 162), (144, 163), (145, 164), (146, 164), (147, 164), (148, 164), (149, 164), (150, 164), (151, 164), (152, 164), (153, 164), (154, 164), (155, 164), (156, 164), (157, 164), (158, 164), (159, 164), (160, 164), (161, 164), (162, 164), (163, 164), (164, 164), (165, 164), (166, 164), (167, 164), (168, 164), (169, 164), (170, 164), (171, 164), (172, 164), (173, 164), (174, 164), (175, 164), (176, 164), (177, 164), (178, 163), (179, 162), (180, 161), (181, 160), (182, 159), (183, 158), (184, 157), (185, 156), (186, 155), (187, 154), (188, 153), (189, 152), (190, 151), (191, 150), (192, 149), (193, 148), (194, 147), (195, 146), (196, 145), (197, 144), (198, 143), (199, 142), (200, 141), (200, 140), (200, 139), (200, 138), (200, 137), (200, 136), (200, 135), (200, 134), (200, 133), (200, 132), (200, 131), (200, 130), (200, 129), (200, 128), (200, 127), (200, 126), (200, 125), (200, 124), (200, 123), (200, 122), (200, 121), (200, 120), (200, 119), (200, 118), (200, 117), (200, 116), (200, 115), (200, 114), (200, 113), (200, 112), (200, 111), (200, 110), (200, 109), (200, 108), (200, 107), (200, 106), (200, 105), (200, 104), (200, 103), (200, 102), (200, 101), (200, 100), (200, 99), (200, 98), (200, 97), (200, 96), (200, 95), (200, 94), (200, 93), (200, 92), (200, 91), (200, 90), (200, 89), (200, 88), (200, 87), (200, 86), (200, 85), (200, 84), (200, 83), (200, 82), (200, 81), (200, 80), (200, 79), (200, 78), (200, 77), (200, 76), (200, 75), (200, 74), (200, 73), (200, 72), (200, 71), (200, 70)] 
    print("Starting navigation...")
    for node in path:
        x_goal = (node[0] - x_offset) * cell_size
        y_goal = (node[1] - y_offset) * cell_size
        print("Moving to:", x_goal, y_goal)
        mover(x_goal, y_goal)

    print("Reached final destination.")

    # ---- Plot trajectory ----
    min_len = min(len(trajectory_x), len(trajectory_y))
    trajectory_x = trajectory_x[:min_len]
    trajectory_y = trajectory_y[:min_len]

    plt.figure()
    plt.plot(trajectory_x, trajectory_y, 'b-', label="Robot Path")
    plt.scatter(trajectory_x[0], trajectory_y[0], c='g', marker='o', label="Start")
    plt.scatter(trajectory_x[-1], trajectory_y[-1], c='r', marker='x', label="Goal")
    plt.title("Robot Trajectory from Odom")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend()
    plt.axis("equal")
    plt.grid()
    plt.show()