import numpy as np
import torch
import torch.optim as optim 
from network import DeepQNetwork

class Agent():
    def __init__(self, n_inputs, n_actions, lr = 3e-4, gamma = 0.95,
                eps = 1.0, eps_dec = 5e-4, eps_end = 0.01, batch_size = 64):

        self.gamma = gamma
        self.eps = eps
        self.eps_dec = eps_dec
        self.eps_end = eps_end
        self.lr = lr

        self.action_space = [i for i in range(n_actions)]
        self.Q = DeepQNetwork(n_inputs, n_actions)
        self.batch_size = batch_size

    