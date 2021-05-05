#!/usr/bin/env python

import rospy
import torch
from DQN import DQN
from PathPlanningDQNAgent import PathPlanningDQNAgent
import sys
sys.path.append('/home/yoksanherlie/Documents/univ/envs/custom_envs')
#from environment_stage_2 import Env
from environment_stage_3 import Env

if __name__ == '__main__':
    rospy.init_node('dojo_robot_path_planning_dqn', anonymous=True, log_level=rospy.INFO)

    # environment setup
    action_space_size = 5
    observation_space_size = 28 # from env (24 laser scans, heading, current distance to goal, obs_min_range, obs_angle)
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

    load_model = False
    if load_model:
        checkpoint = torch.load('/home/yoksanherlie/catkin_ws/src/dojo-robot/training_results/path_planning/stage_2_ep_500.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(model)
        rospy.loginfo('haihai')
        rospy.loginfo(model)

    agent = PathPlanningDQNAgent(
        env=env,
        model=model,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        gamma=gamma,
        sync_frequency=sync_frequency,
        batch_size=batch_size
    )
    agent.train(n_episodes, n_steps)
