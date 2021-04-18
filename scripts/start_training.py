#!/usr/bin/env python

import gym
import rospy
import rospkg
from DQN import DQN
from DQNAgent import DQNAgent
from ReplayMemory import ReplayMemory

# env
import gym_slam

if __name__ == '__main__':

    rospy.init_node('dqn_dojo_robot', anonymous=True, log_level=rospy.INFO)

    # environment setup
    env_name = rospy.get_param('/dqn_robot/environment_name')
    env = gym.make(env_name)
    rospy.loginfo('Gym environment done')

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
        n_inputs=env.observation_space.shape[0],
        n_outputs=env.action_space.n,
        learning_rate=learning_rate
    )

    dqn_agent = DQNAgent(
        env=env,
        model=model,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        gamma=gamma,
        sync_frequency=sync_frequency,
        batch_size=batch_size
    )
    dqn_agent.train(n_episodes, n_steps)

