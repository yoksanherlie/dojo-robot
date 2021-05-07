#!/usr/bin/env python

import rospy
import actionlib
from move_base_msgs.msg import *
from hector_nav_msgs.srv import GetRobotTrajectory

class ExplorationClient:

    def __init__(self):
        self.plan_service = rospy.ServicePoxy('get_exploration_path', GetRobotTrajectory)
        self.move_base_action = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    def start(self):
        r = rospy.Rate(1 / 10.0)
        while not rospy.is_shutdown():
            result = self.explore()
            r.sleep()
    
    def explore(self):
        rospy.loginfo('Waiting for plan service...')
        path = self.plan_service().trajectory
        if len(path.poses) > 0:
            rospy.loginfo('Moving to frontier...')
            return self.navigate(path.poses[-1])
        return False
    
    def navigate(self, pose, timeout=20.0):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = '/map'
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose = pose

        self.move_base_action.send_goal(goal)

        self.move_base_action.wait_for_result()

        return self.move_base_action.get_result()

if __name__ == '__main__':
    rospy.init_node('hector_exploration_client', anonymous=True)
    client = ExplorationClient()
    client.start()