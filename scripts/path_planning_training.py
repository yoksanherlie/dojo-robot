#!/usr/bin/env python

import rospy
from DQN import DQN
from PathPlanningDQNAgent import PathPlanningDQNAgent

if __name__ == '__main__':
    rospy.init_node('dojo_robot_path_planning_dqn', anonymous=True, log_level=rospy.INFO)

    # environment setup
    action_space_size = 5
    observation_space_size = 26 # from env (24 laser scans, heading, current distance to goal)
    env = Env(action_space_size)

    # load parameters from config file
    learning_rate = rospy.get_param('/dqn_robot/learning_rate')
    epsilon_start = rospy.get_param('/dqn_robot/epsilon_start')
    epsilon_min = rospy.get_param('/dqn_robot/epsilon_min')
    epsilon_decay = rospy.get_param('/dqn_robot/epsilon_decay')
    gamma = rospy.get_param('/dqn_robot/gamma')
    n_episodes = rospy.get_param('/dqn_robot/n_episodes')
    n_steps = rospy.get_param('/dqn_robot/n_steps')
    sync_frequency = rospy.get_param('/dqn_robot/sync_frequency')
    batch_size = rospy.get_param('/dqn_robot/batch_size')

    model = DQN(
        n_inputs=observation_space_size,
        n_outputs=action_space_size,
        learning_rate=learning_rate
    )

    agent = PathPlanningDQNAgent(
        model=model,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        gamma=gamma,
        sync_frequency=sync_frequency,
        batch_size=batch_size
    )
    agent.train(n_episodes, n_steps)
