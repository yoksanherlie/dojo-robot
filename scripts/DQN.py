import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self, n_inputs, n_outputs, learning_rate):
        super(DQN, self).__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_nodes = 128

        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, self.n_hidden_nodes),
            nn.ReLU(),
            nn.Linear(self.n_hidden_nodes, self.n_hidden_nodes),
            nn.ReLU(),
            nn.Linear(self.n_hidden_nodes, self.n_hidden_nodes),
            nn.ReLU(),
            nn.Linear(self.n_hidden_nodes, self.n_outputs)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def get_action(self, state):
        qvals = self.get_qvals(state)
        return torch.max(qvals, dim=-1)[1].item()

    def get_qvals(self, state):
        if type(state) is tuple:
            state = np.array([np.ravel(s) for s in state])
        state_t = torch.FloatTensor(state)
        return self.network(state_t)

