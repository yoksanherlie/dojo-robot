#!/usr/bin/env python

import rospy
import torch
import time
from DuelingDQN import DuelingDQN
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose

class AgentTest():
    def __init__(self, goal_x, goal_y):
        model = DuelingDQN(
            num_inputs=28,
            num_outputs=5
        )
        self.model = self.load_model('/home/yoksanherlie/catkin_ws/src/dojo-robot/training_results/path_planning/dueling_stage_3_ep_190_resume_810.pt', model)

        self.goal_x = goal_x
        self.goal_y = goal_y
        self.position = Pose()
        self.arrived = False
        self.step = 1

        #rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=5)

    def load_model(self, path, model):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def preprocess_state(self, state):
        return torch.autograd.Variable(torch.FloatTensor(state).unsqueeze(0))

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        return goal_distance

    def odom_callback(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)

    def get_scan_range_data(self):
        scan = None
        while scan is None:
            try:
                scan = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        scan_range = []

        for i in range(len(scan.ranges)):
            if i % 15 == 0:
                if scan.ranges[i] == float('Inf'):
                    scan_range.append(3.5)
                elif np.isnan(scan.ranges[i]):
                    scan_range.append(0)
                else:
                    scan_range.append(scan.ranges[i])

        return scan_range

    def run(self, scan_range):
        scan_range = self.get_scan_range_data()

        heading = self.heading
        obstacle_min_range = round(min(scan_range), 2)
        obstacle_angle = np.argmin(scan_range)

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)
        arrived = False
        if current_distance < 0.3:
            self.arrived = True
            self.pub_cmd_vel.publish(Twist())
            rospy.logwarn('arrived, step: {}'.format(step))
            return

        state = scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle]
        state = self.preprocess_state(state)
        rospy.loginfo('State: {}'.format(state))

        action = self.model.get_action(state)
        self.step(action)

    def step(self, action):
        max_angular_vel = 1.5
        ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.15
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        rospy.loginfo('Step: {}'.format(self.step))

        self.step += 1

if __name__ == '__main__':
    rospy.init_node('dojo_robot_path_planning_d3qn_test', anonymous=True, log_level=rospy.INFO)

    observation_space_size = 28
    action_space_size = 5

    goal_x = 1.0
    goal_y = 1.0

    agent = AgentTest(goal_x, goal_y)
    agent.run()

    #rospy.Subscriber('/scan', LaserScan, scan_callback)

    rospy.spin()

