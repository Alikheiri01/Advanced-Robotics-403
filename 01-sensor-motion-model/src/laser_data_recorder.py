#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Range
import csv

class RangeLogger:
    def __init__(self):
        self.count = 0
        self.max_count = 500
        self.data = []
        rospy.Subscriber("/vector/laser", Range, self.callback)

    def callback(self, msg):
        if self.count < self.max_count:
            # Record timestamp and range data
            record = [
                msg.header.stamp.to_sec(),
                msg.radiation_type,
                msg.field_of_view,
                msg.min_range,
                msg.max_range,
                msg.range
            ]
            self.data.append(record)
            self.count += 1
            rospy.loginfo(f"Captured {self.count}/500 messages")
        else:
            self.save_to_csv()
            rospy.signal_shutdown("Data collection complete.")

    def save_to_csv(self):
        filename = "laser_range_data.csv"
        rospy.loginfo(f"Saving to {filename}")
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "timestamp", "radiation_type", "field_of_view",
                "min_range", "max_range", "range"
            ])
            writer.writerows(self.data)

if __name__ == '__main__':
    rospy.init_node('laser_range_logger_node')
    logger = RangeLogger()
    rospy.spin()
