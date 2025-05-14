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
import torch
import torch.autograd as autograd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import csv
import secrets

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class PathPlanningDuelingDQNAgent():
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
            action = secrets.SystemRandom().randrange(0,5) # 5 actions
        else:
            state = Variable(torch.FloatTensor(state).unsqueeze(0))
            rospy.loginfo('state cuda: {}'.format(state.is_cuda))
            action = self.model.get_action(state)
        return action, epsilon
    
    def train(self, max_episodes=1000, max_steps=300, start_episode=1):
        start_time = time.time()
        decay_step = 1
        result = Float32MultiArray()

        for episode in range(start_episode, start_episode + max_episodes):
            rospy.logwarn('=========== START EPISODE {} ==========='.format(episode))

            episode_start_time = time.time()

            state = self.env.reset()
            self.reward = 0
            
            done = False
            step = 1
            while not done and step <= max_steps:
                rospy.logwarn('Episode: {} - Step {}'.format(episode, step))

                # choose action from primary network (policy network)
                action, epsilon = self.select_action(state, decay_step)
                # execute action in the environment and get feedback
                next_state, reward, done = self.env.step(action)
                self.reward += reward

                # rospy.loginfo('State: {}'.format(state))
                rospy.loginfo('Reward: {}'.format(reward))

                # add experience to replay memory
                self.memory.append(state, action, reward, done, next_state)
                state = next_state

                # update model (policy network)
                if decay_step % 1 == 0:
                    self.update()

                # Sync target model with current model (policy network -> target network)
                if decay_step % self.sync_frequency == 0:
                    rospy.logwarn('Update target model: step {}'.format(decay_step))
                    self.update_target_model()
                    self.update_counter += 1
                
                step += 1
                decay_step += 1
            
            # end of episode
            episode_time = self.calculate_time(episode_start_time)
            total_time = self.calculate_time(start_time)

            # logging for plotting
            self.rewards.append(self.reward)
            mean_reward = np.mean(self.rewards[-self.window:])
            self.mean_rewards.append(mean_reward)
            self.episode_losses.append(np.mean(self.losses))
            self.losses = []

            result.data = [self.reward, mean_reward, self.episode_losses[-1]]
            self.result_pub.publish(result)

            # save model per episode
            if episode % 10 == 0:
                self.save_model(episode)

            if episode % 100 == 0:
                path = '/home/yoksanherlie/catkin_ws/src/dojo-robot/data/'
                with open(path+'reward.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.rewards)

                with open(path+'loss.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.episode_losses)

            rospy.logwarn('Episode {} - [reward: {}, epsilon: {:.3}, mean_loss: {}, decay_step: {}, update_counter: {}, time: {}] - Total time: {}'.format(
                episode,
                self.reward,
                epsilon,
                self.episode_losses[-1],
                decay_step,
                self.update_counter,
                episode_time,
                total_time
            ))
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
        path = '/home/yoksanherlie/catkin_ws/src/dojo-robot/training_results/path_planning/NEW_dueling_stage_4_ep_{}_2.pt'.format(episode)
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
