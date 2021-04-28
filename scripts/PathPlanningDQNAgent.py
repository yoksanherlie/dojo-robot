from ReplayMemory import ReplayMemory
from copy import deepcopy
import torch.nn as nn
import time
import rospy
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

class PathPlanningDQNAgent():
    def __init__(self, model, epsilon_start, epsilon_min, epsilon_decay, gamma, sync_frequency, batch_size):
        self.model = model
        self.target_model = deepcopy(model) # target network
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.sync_frequency = sync_frequency
        self.batch_size = batch_size

        self.initialize()
    
    def select_action(self, state, decay_step):
        epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * np.exp(-1 * decay_step * self.epsilon_decay)

        if np.random.random() < epsilon:
            action = 1
        else:
            action = self.model.get_action(state)
        return action, epsilon
    
    def train(self, max_episodes=1000, max_steps=300):
        start_time = time.time()
        decay_step = 1
        for episode in range(1, max_episodes + 1):
            rospy.logwarn('=========== START EPISODE {} ==========='.format(episode))

            episode_start_time = time.time()

            state = self.env.reset()
            self.reward = 0
            
            done = False
            step = 1
            while not done and step <= max_steps:
                rospy.logwarn('---------- EPISODE: {} - START STEP {} ----------'.format(episode, step))

                # choose action from primary network (policy network)
                action, epsilon = self.select_action(state, decay_step)
                # execute action in the environment and get feedback
                next_state, reward, done, info = self.env.step(action)
                self.reward += reward

                rospy.logwarn('Reward: {}'.format(reward))

                # add experience to replay memory
                self.memory.append(state, action, reward, done, next_state)
                state = next_state

                # update model (policy network)
                self.update()

                # Sync target model with current model (policy network -> target network)
                if episode % self.sync_frequency == 0:
                    rospy.logwarn('Update target model: episode {}'.format(episode))
                    self.update_target_model()
                
                rospy.logwarn('---------- EPISODE: {} - END STEP {} ----------'.format(episode, step))
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

            rospy.logwarn('Episode {} - [reward: {}, epsilon: {:.2}, decay_step: {}, time: {}] - Total time: {}'.format(
                episode,
                self.reward,
                epsilon,
                decay_step,
                episode_time,
                total_time
            ))
        self.save_model(episode)
        self.plot_results()
    
    def update(self):
        if self.memory.can_provide_sample():
            rospy.logwarn('Updating model by preprocessing samples from experience replay...')
            self.model.optimizer.zero_grad()
            batch = self.memory.sample(self.batch_size)
            loss = self.calculate_loss(batch)
            loss.backward()
            self.model.optimizer.step()
            self.losses.append(loss.detach().numpy())
    
    def calculate_loss(self, batch):
        states, actions, rewards, dones, next_states = [i for i in batch]
        rewards_t = torch.FloatTensor(rewards).reshape(-1, 1)
        actions_t = torch.LongTensor(np.array(actions)).reshape(-1, 1)
        dones_t = torch.ByteTensor(dones)

        qvals = torch.gather(self.model.get_qvals(states), 1, actions_t)
        next_actions = torch.max(self.network.get_qvals(next_states), dim=-1)[1]
        next_actions_t = torch.LongTensor(next_actions).reshape(-1,1)
        target_qvals = self.target_network.get_qvals(next_states)
        qvals_next = torch.gather(target_qvals, 1, next_actions_t).detach()
        qvals_next[dones_t] = 0 # zero-out terminal states
        expected_qvals = self.gamma * qvals_next + rewards_t
        loss = nn.MSELoss()(qvals, expected_qvals)
        return loss
    
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
        plt.show()

    def plot_rewards(self):
        print('Rewards: {}'.format(self.rewards))
        plt.figure(num='Training Rewards', figsize=(12,8))
        plt.plot(self.rewards, label='Rewards')
        plt.plot(self.mean_rewards, label='Mean Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.legend()

    def plot_losses(self):
        print('Losses: {}'.format(self.episode_losses))
        plt.figure(num='Training Losses', figsize=(12,8))
        plt.plot(self.episode_losses, label='Loss')
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.legend()

    def save_model(self, episode):
        path = '/home/yoksanherlie/catkin_ws/src/dojo-robot/training_results/path_planning/stage_1_ep_{}.pt'.format(episode)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.model.optimizer.state_dict(),
            'target_optimizer_state_dict': self.target_model.optimizer.state_dict()
        }, path)

    def initialize(self):
        self.reward = 0
        self.memory = ReplayMemory(50000, 300)
        self.rewards = []
        self.mean_rewards = []
        self.losses = []
        self.episode_losses = []
        self.window = 20
