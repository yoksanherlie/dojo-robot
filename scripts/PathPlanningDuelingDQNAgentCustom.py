from ReplayMemory import ReplayMemory
from copy import deepcopy
import torch.nn as nn
import time
import rospy
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from std_msgs.msg import Float32MultiArray
import random
import torch
import torch.autograd as autograd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class PathPlanningDuelingDQNAgentCustom():
    def __init__(self, env, model, target_model, optimizer, epsilon_start, epsilon_min, epsilon_decay, gamma, sync_frequency, batch_size):
        self.env = env
        self.model = model
        self.target_model = target_model # target network
        self.optimizer = optimizer
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.sync_frequency = sync_frequency
        self.batch_size = batch_size
        self.result_pub = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.update_counter = 0
        self.fig = make_subplots(rows=2, cols=1, subplot_titles=('Rewards', 'Losses'))

        if USE_CUDA:
            rospy.loginfo('cuda')
            self.model.cuda()
            self.target_model.cuda()

        self.initialize()
    
    def select_action(self, state, decay_step):
        epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * np.exp(-1 * decay_step * self.epsilon_decay)

        if np.random.random() < epsilon:
            action = random.randrange(0,5) # 5 actions
        else:
            state = Variable(torch.FloatTensor(state).unsqueeze(0))
            rospy.loginfo('state cuda: {}'.format(state.is_cuda))
            action = self.model.get_action(state)
        return action, epsilon
    
    def train(self, max_frames=1000, start_frame=1):
        result = Float32MultiArray()
        episode = 1
        episode_start_time = time.time()

        state = self.env.reset()
        for idx in range(start_frame, max_frames+start_frame):
            rospy.loginfo('Episode: {} - frame: {}'.format(episode, idx))

            action, epsilon = self.select_action(state)
            next_state, reward, done = self.env.step(action)
            self.reward += reward

            self.memory.append(state, action, reward, done, next_state)
            state = next_state

            self.update()

            if idx % 1000 == 0:
                self.save_model(episode)
                self.update_target_model()

            if done:
                episode_end_time = self.calculate_time(episode_start_time)
                state = self.env.reset()
                rospy.logwarn('Episode {} - [reward: {}, epsilon: {:.3}, mean_loss: {}] - time: {}'.format(
                    episode,
                    self.reward,
                    epsilon,
                    0,
                    episode_end_time,
                ))
                episode += 1
                self.reward = 0

            # logging data
            result.data = [self.reward, mean_reward, self.episode_losses[-1]]
            self.result_pub.publish(result)

        self.save_model(episode)
        self.plot_results()
    
    def update(self):
        if self.memory.can_provide_sample():
            rospy.loginfo('Updating model by preprocessing samples from experience replay...')
            batch = self.memory.sample(self.batch_size)
            loss = self.calculate_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            self.losses.append(loss.item())
    
    def calculate_loss(self, batch):
        states, actions, rewards, dones, next_states = [i for i in batch]
        state = Variable(torch.FloatTensor(np.float32(states)))
        next_state = Variable(torch.FloatTensor(np.float32(next_states)))
        action = Variable(torch.LongTensor(actions))
        reward = Variable(torch.FloatTensor(rewards))
        done = Variable(torch.FloatTensor(dones))

        q_values = self.model(state)
        next_q_values = self.target_model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = nn.MSELoss()(q_value, expected_q_value.detach())

        return loss.mean()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def calculate_time(self, start_time):
        current = time.time()
        m, s = divmod(int(current - start_time), 60)
        h, m = divmod(m, 60)
        return '{:02d}:{:02d}:{:02d}'.format(h, m, s)

    def plot_results(self):
        self.plot_rewards()
        self.plot_losses()
        #plt.show()
        self.fig.update_layout(title='Training Results')
        self.fig.show()

    def plot_rewards(self):
        print('Rewards: {}'.format(self.rewards))
        self.fig.add_trace(go.Scatter(x=[i for i in range(1, len(self.rewards)+1)], y=self.rewards, name='rewards', mode='lines'), row=1, col=1)
        self.fig.add_trace(go.Scatter(x=[i for i in range(1, len(self.mean_rewards)+1)], y=self.mean_rewards, name='mean_rewards'), row=1, col=1)
        self.fig.update_xaxes(title_text='Episodes', row=1, col=1)
        self.fig.update_yaxes(title_text='Rewards', row=1, col=1)
        #fig.update_layout(title='Training rewards', xaxis_title='Episodes', yaxis_title='Rewards')
        #fig.show()
        '''plt.figure(num='Training Rewards', figsize=(12,8))
        plt.plot(self.rewards, label='Rewards')
        plt.plot(self.mean_rewards, label='Mean Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.legend()'''

    def plot_losses(self):
        print('Losses: {}'.format(self.episode_losses))
        self.fig.add_trace(go.Scatter(x=[i for i in range(1, len(self.episode_losses)+1)], y=self.episode_losses, name='losses', mode='lines'), row=2, col=1)
        self.fig.update_xaxes(title_text='Episodes', row=2, col=1)
        self.fig.update_yaxes(title_text='Loss', row=2, col=1)
        #fig.update_layout(title='Training losses', xaxis_title='Episodes', yaxis_title='Losses')
        #fig.show()
        '''plt.figure(num='Training Losses', figsize=(12,8))
        plt.plot(self.episode_losses, label='Loss')
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.legend()'''

    def save_model(self, episode):
        path = '/home/yoksanherlie/catkin_ws/src/dojo-robot/training_results/path_planning/NEW_dueling_stage_3_frame_{}_2.pt'.format(episode)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': episode,
            'loss': self.episode_losses[-1]
        }, path)

    def initialize(self):
        self.reward = 0
        self.memory = ReplayMemory(100000, 64)
        self.rewards = []
        self.mean_rewards = []
        self.losses = []
        self.episode_losses = []
        self.window = 20
