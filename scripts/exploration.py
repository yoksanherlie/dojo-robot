#!/usr/bin/env python
import rospy
import actionlib
from hector_nav_msgs.srv import *
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Point

def hector_exploration_client():
    rospy.init_node('hector_exploration_client', anonymous=True)
    rospy.loginfo('[hector_exploration_client] start -->')
    rospy.wait_for_service('get_exploration_path')
    try:
        exploration_service = rospy.ServiceProxy('get_exploration_path', GetRobotTrajectory)
        response = exploration_service()
        # print(response)
        waypoints = [Point(2.0, 1.5, 0)]

        action = actionlib.SimpleActionClient('move_base', MoveBaseAction)

        while (not action.wait_for_server(rospy.Duration.from_sec(5.0))):
            rospy.loginfo('move_base action ready')
        
        for pose in response.trajectory.poses:
            rospy.loginfo('new pose in trajectory list')
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = 'map'
            goal.target_pose.header.stamp = rospy.Time.now()

            goal.target_pose.pose.position = pose.pose.position
            goal.target_pose.pose.orientation = pose.pose.orientation
            #goal.target_pose.pose.position = pose
            #goal.target_pose.pose.orientation.x = 0.0
            #goal.target_pose.pose.orientation.y = 0.0
            #goal.target_pose.pose.orientation.z = 0.0
            #goal.target_pose.pose.orientation.w = 1.0

            rospy.loginfo('Sending goal location...')
            rospy.loginfo(goal)
            action.send_goal(goal)

            action.wait_for_result(rospy.Duration.from_sec(20.0))
    except rospy.ServiceException as e:
        rospy.logwarn('Service call failed: {}'.format(e))

if __name__ == '__main__':
    try:
        hector_exploration_client()
    except rospy.ROSInterruptException:
        pass
