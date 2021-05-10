import torch.nn as nn

class DuelingDQN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DuelingDQN, self).__init__()

        self.hidden_layers = 512
        self.body_size = 128

        self.feature = nn.Sequential(
            nn.Linear(num_inputs, self.body_size),
            nn.ReLU(),
        )

        self.advantage = nn.Sequential(
            nn.Linear(self.body_size, self.hidden_layers),
            nn.ReLU(),
            nn.Linear(self.hidden_layers, num_outputs)
        )

        self.value = nn.Sequential(
            nn.Linear(self.body_size, self.hidden_layers),
            nn.ReLU(),
            nn.Linear(self.hidden_layers, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()

    def get_action(self, state):
        q_value = self.forward(state)
        action = q_value.max(1)[1].data[0]
        return action
    
