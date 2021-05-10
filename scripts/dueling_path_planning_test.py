#!/usr/bin/env python

import rospy
import torch
import time
from DuelingDQN import DuelingDQN
import sys
sys.path.append('/home/yoksanherlie/Documents/univ/envs/custom_envs')
#from environment_stage_3_test import Env
from environment_stage_4_test import Env
import plotly.graph_objects as go
import torch.autograd as autograd
import numpy as np

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

def load_model(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def preprocess_state(state):
    return Variable(torch.FloatTensor(state).unsqueeze(0))

if __name__ == '__main__':
    rospy.init_node('dojo_robot_path_planning_dqn', anonymous=True, log_level=rospy.INFO)

    # environment setup
    action_space_size = 5
    observation_space_size = 28 # from env (24 laser scans, heading, current distance to goal, obs_min_range, obs_angle)
    env = Env(action_space_size)
    max_episodes = 100

    model = DuelingDQN(
        num_inputs=observation_space_size,
        num_outputs=action_space_size,
    )
    model = load_model('../training_results/path_planning/NEW_dueling_stage_4_ep_600_2.pt', model)

    if USE_CUDA:
        model.cuda()

    arrived_counter = 0
    average_rewards = []
    steps = []
    # test
    for e in range(1, max_episodes + 1):
        state = preprocess_state(env.reset())
        done = False
        arrived = False
        step = 1
        total_reward = 0
        while not done and not arrived and step <= 500:
            rospy.logwarn('Episode: {} - Step {}'.format(e, step))
            
            action = model.get_action(state)
            next_state, reward, done, arrived = env.step(action)
            state = preprocess_state(next_state)
            total_reward += reward

            rospy.loginfo('Reward: {}'.format(reward))

            step += 1
        
        if arrived:
            arrived_counter += 1
            rospy.loginfo('arrived! waiting 10s')
            time.sleep(10.0)
            rospy.loginfo('done waiting, continue')
            steps.append(step)

        average_rewards.append(total_reward/step)

        rospy.loginfo('episode: {}/{}, reward: {}, done: {}, arrived: {} - {}/{}'.format(e, max_episodes, total_reward, done, arrived, arrived_counter, e))

    rospy.loginfo('average_rewards: {}'.format(average_rewards))
    rospy.loginfo('mean rewards: {}'.format(np.mean(average_rewards)))
    rospy.loginfo('step: {}, mean: {}'.format(steps, np.mean(steps)))
    #fig = go.Figure(go.Scatter(x=[i for i in range(1, len(average_rewards)+1)], y=average_rewards, name='mean rewards', mode='lines'))
    #fig.show()
