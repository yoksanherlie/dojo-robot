#!/usr/bin/env python

import rospy
import torch
from DuelingDQN import DuelingDQN
from PathPlanningDuelingDQNAgent import PathPlanningDuelingDQNAgent
import sys
sys.path.append('/home/yoksanherlie/Documents/univ/envs/custom_envs')
#from environment_stage_2 import Env
#from environment_stage_3 import Env
from environment_stage_4 import Env
import torch

def load_model_checkpoint(path, model, target_model, optimizer):
    checkpoint = torch.load(path)
    print(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    target_model.load_state_dict(checkpoint['target_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, target_model, optimizer, checkpoint['epoch']


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

    model = DuelingDQN(
        num_inputs=observation_space_size,
        num_outputs=action_space_size,
    )
    target_model = DuelingDQN(
        num_inputs=observation_space_size,
        num_outputs=action_space_size
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    load_model = False
    if load_model:
        rospy.loginfo('loading model...')
        path = '/home/yoksanherlie/catkin_ws/src/dojo-robot/training_results/path_planning/NEW_dueling_stage_3_ep_500.pt'
        model, target_model, optimizer, ep = load_model_checkpoint(path, model, target_model, optimizer)
        epsilon_start = 0.05

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    agent = PathPlanningDuelingDQNAgent(
        env=env,
        model=model,
        target_model=target_model,
        optimizer=optimizer,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        gamma=gamma,
        sync_frequency=sync_frequency,
        batch_size=batch_size,
    )
    agent.train(n_episodes, n_steps)
