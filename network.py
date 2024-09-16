import torch.nn as nn


class DeepQNetwork(nn.Module):
    def __init__(self, n_inputs, n_actions, h1 = 256, h2 = 256):
        super().__init__()
        self.Q = nn.Sequential(
            nn.Linear(n_inputs, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, n_actions)
        )

    def forward(self, state):
        actions = self.Q(state)

        return actions