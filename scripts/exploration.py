#!/usr/bin/env python

import rospy
import actionlib
from move_base_msgs.msg import *
from hector_nav_msgs.srv import GetRobotTrajectory

class ExplorationClient:

    def __init__(self):
        self.plan_service = rospy.ServiceProxy('get_exploration_path', GetRobotTrajectory)
        self.move_base_action = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    def start(self):
        start_time = rospy.Time.now()
        r = rospy.Rate(1/7.0)
        rospy.loginfo('Start exploring...')
        while not rospy.is_shutdown():
            result = self.explore()
            r.sleep()
            time = rospy.Time.now()
            duration = time - start_time
            rospy.loginfo('Time so far: {}s'.format(duration.to_sec()))
            if not result:
                rospy.loginfo('done')
                break
        end_time = rospy.Time.now()
        duration = end_time - start_time
        rospy.loginfo('Exploration done in {}s'.format(duration.to_sec()))
    
    def explore(self):
        rospy.loginfo('Waiting for plan service...')
        path = self.plan_service().trajectory
        if len(path.poses) > 0:
            rospy.loginfo('Moving to frontier...')
            return self.navigate(path.poses[-1])
        else:
            rospy.logwarn('No frontiers left!')
        return False
    
    def navigate(self, pose, timeout=5.0):
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
