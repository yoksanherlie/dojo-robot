#!/usr/bin/env python
import rospy

#from actionlib import SimpleActionClient, GoalStatus
import actionlib
from move_base_msgs.msg import *
from hector_nav_msgs.srv import GetRobotTrajectory

class ExplorationController:

    def __init__(self):
        self._plan_service = rospy.ServiceProxy('get_exploration_path', GetRobotTrajectory)
        self._move_base = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    def run(self):
        r = rospy.Rate(1 / 7.0)
        while not rospy.is_shutdown():
            result = self.run_once()
            r.sleep()
            #rospy.loginfo(result)
            if result == False:
                break

    def run_once(self):
        path = self._plan_service().trajectory
        poses = path.poses
        rospy.loginfo('Get poses')
        if not path.poses:
            rospy.loginfo('No frontiers left.')
            return False
        rospy.loginfo('Moving to frontier...')
        rospy.loginfo('Pose: {}'.format(poses[-1]))
        status = self.move_to_pose(poses[-1])
        rospy.loginfo('Goal Status: {}'.format(status))

    def move_to_pose(self, pose_stamped, timeout=6.0):
        goal = MoveBaseGoal()
        goal.target_pose = pose_stamped
        self._move_base.send_goal(goal)
        self._move_base.wait_for_result(rospy.Duration(timeout))
        return self._move_base.get_state() == actionlib.GoalStatus.SUCCEEDED

if __name__ == '__main__':
    rospy.init_node('hector_exploration_client', anonymous=True)
    controller = ExplorationController()
    controller.run()

