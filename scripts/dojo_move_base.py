#!/usr/bin/env python

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Point

def test():
    rospy.init_node('testing_aja', anonymous=True)

    while not rospy.is_shutdown():
        rospy.loginfo('[testing_aja] start -->')
        waypoints = [Point(2.0, 1.5, 0)]

        action = actionlib.SimpleActionClient('move_base', MoveBaseAction)

        while (not action.wait_for_server(rospy.Duration.from_sec(5.0))):
            rospy.loginfo('waiting move_base action')
            continue

        rospy.loginfo('move base action ready')
        for point in waypoints:
            rospy.loginfo('new point in waypoints')

            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = 'map'
            goal.target_pose.header.stamp = rospy.Time.now()

            goal.target_pose.pose.position = pose
            goal.target_pose.pose.orientation.x = 0.0
            goal.target_pose.pose.orientation.y = 0.0
            goal.target_pose.pose.orientation.z = 0.0
            goal.target_pose.pose.orientation.w = 1.0

            rospy.loginfo('Sending goal location...')
            rospy.loginfo(goal)
            action.send_goal(goal)

            action.wait_for_result(rospy.Duration.from_sec(20.0))

if __name__ == '__main__':
    try:
        test()
    except rospy.ROSInterruptException:
        print('mati')
        pass
