from ReplayMemory import ReplayMemory
from copy import deepcopy
import torch.nn as nn
import time
import rospy
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

class DQNAgent():
    def __init__(self, env, model, epsilon_start, epsilon_min, epsilon_decay, gamma, sync_frequency, batch_size):
        self.env = env
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
            action = self.env.action_space.sample()
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
                rospy.logwarn('---------- START STEP {} ----------'.format(step))

                # choose action from primary network (policy network)
                action, epsilon = self.select_action(state, decay_step)
                # execute action in the environment and get feedback
                next_state, reward, done, _ = self.env.step(action)
                self.reward += reward

                # add experience to replay memory
                self.memory.append(state, action, reward, done, next_state)
                state = next_state

                # update model (policy network)
                self.update()

                # Sync target model with current model (policy network -> target network)
                if episode % self.sync_frequency == 0:
                    rospy.logwarn('Update target model: episode {}'.format(episode))
                    self.update_target_model()

                rospy.logwarn('---------- END STEP {} ----------'.format(step))
                step += 1
                decay_step += 1

            # end of episode
            episode_time = self.calculate_time(episode_start_time)
            total_time = self.calculate_time(start_time)

            # reward logging for plotting
            self.rewards.append(self.reward)
            mean_reward = np.mean(self.rewards[-self.window:])
            self.mean_rewards.append(mean_reward)

            rospy.logwarn('Episode {} - [reward: {}, epsilon: {:.2}, decay_step: {}, time: {}] - Total time: {}'.format(
                episode,
                self.reward,
                epsilon,
                decay_step,
                episode_time,
                total_time
            ))
        self.env.close()
        self.save_model(episode)
        self.plot_rewards()

    def update(self):
        if self.memory.can_provide_sample():
            rospy.logwarn('Updating model by preprocessing samples from experience replay...') # back propagation and gradient descent for network
            self.model.optimizer.zero_grad()
            batch = self.memory.sample(self.batch_size)
            loss = self.calculate_loss(batch)
            loss.backward()
            self.model.optimizer.step()
        # self.update_loss.append()

    def calculate_loss(self, batch):
        states, actions, rewards, dones, next_states = [i for i in batch]
        rewards_t = torch.FloatTensor(rewards)
        actions_t = torch.LongTensor(np.array(actions)).reshape(-1, 1)
        dones_t = torch.ByteTensor(dones)

        # addition
        states = np.squeeze(states, axis=1)
        next_states = np.squeeze(next_states, axis=1)

        qvals = torch.gather(self.model.get_qvals(states), 1, actions_t).squeeze()
        qvals_next = torch.max(self.target_model.get_qvals(next_states), dim=-1)[0].detach()
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

    def plot_rewards(self):
        print(self.rewards)
        plt.figure(figsize=(12,8))
        plt.plot(self.rewards, label='Rewards')
        plt.plot(self.mean_rewards, label='Mean Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.show()

    def save_model(self, episode):
        path = '/home/yoksanherlie/catkin_ws/src/dojo-robot/training_results/saved_model_ep_{}.pt'.format(episode)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.model.optimizer.state_dict(),
            'target_optimizer_state_dict': self.target_model.optimizer.state_dict()
        }, path)

    def initialize(self):
        self.reward = 0
        self.memory = ReplayMemory(10000, 100)
        self.rewards = []
        self.mean_rewards = []
        self.window = 20

