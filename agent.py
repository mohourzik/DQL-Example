import numpy as np
import torch
import torch.optim as optim 
from network import DeepQNetwork

class Agent():
    def __init__(self, n_inputs, n_actions, lr = 3e-4, gamma = 0.95,
                eps = 1.0, eps_dec = 5e-4, eps_end = 0.01, batch_size = 64, max_mem_size = 100000):

        self.gamma = gamma
        self.eps = eps
        self.eps_dec = eps_dec
        self.eps_end = eps_end
        self.lr = lr

        self.action_space = [i for i in range(n_actions)]
        self.Q = DeepQNetwork(n_inputs, n_actions)
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.mem_size = max_mem_size
        self.mem_count = 0
        self.state_memory       = np.zeros((self.mem_size, n_inputs), dtype = np.float32)
        self.new_state_memory   = np.zeros((self.mem_size, n_inputs), dtype = np.float32)
        self.action_memory      = np.zeros(self.mem_size, dtype = np.int32)
        self.reward_memory      = np.zeros(self.mem_size, dtype = np.float32)
        self.terminal_memory    = np.zeros(self.mem_size, dtype = np.bool_)


    def choose_action(self, obs):
        if np.random.random() > self.eps:
            # greedy selection action
            state = torch.tensor(obs, dtype = torch.float32).to(self.device)
            actions = self.Q(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action
    
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_count % self.mem_size
        
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_count += 1

    