#!/usr/bin/env python

import rospy
from nav_msgs.msg import Path
import math

class TrajectoryLength():
    def __init__(self):
        #self.current_length = 0

        rospy.Subscriber('/trajectory', Path, self.trajectory_callback)
    
    def trajectory_callback(self, data):
        total_distance = 0
        if len(data.poses) > 1:
            for i, pose in enumerate(data.poses):
                total_distance += self.get_distance(pose.pose.position, data.poses[i-1].pose.position)

        rospy.loginfo('Poses length: {}'.format(len(data.poses)))
        rospy.loginfo('Total distance travelled: {}'.format(total_distance))

    def get_distance(self, p1, p2):
        x = round(p1.x) - round(p2.x)
        y = round(p1.y) - round(p2.y)
        return math.sqrt(x**2 + y**2)

if __name__ == '__main__':
    rospy.init_node('get_trajectory_length', anonymous=True)

    trajectory_length = TrajectoryLength()

    rospy.spin()


