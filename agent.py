import numpy as np
import torch
import torch.optim as optim 
from network import DeepQNetwork
import torch.nn.functional as F

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
        self.opimizer = optim.Adam(self.Q.parameters(), lr = self.lr)

        self.mem_size = max_mem_size
        self.mem_count = 0
        self.state_memory       = np.zeros((self.mem_size, n_inputs), dtype = np.float32)
        self.new_state_memory   = np.zeros((self.mem_size, n_inputs), dtype = np.float32)
        self.reward_memory      = np.zeros(self.mem_size, dtype = np.float32)
        self.action_memory      = np.zeros(self.mem_size, dtype = np.int32)
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

    def learn(self):
        if self.mem_count > self.batch_size:
            self.opimizer.zero_grad()
            max_mem = min(self.mem_count, self.mem_size)

            batch_index = np.arange(self.batch_size, dtype = np.int32)
            batch = np.random.choice(max_mem, self.batch_size, replace = False)

            states  = torch.tensor(self.state_memory[batch]).to(self.device)
            states_ = torch.tensor(self.new_state_memory[batch]).to(self.device)
            rewards = torch.tensor(self.reward_memory[batch]).to(self.device)
            dones   = torch.tensor(self.terminal_memory[batch]).to(self.device)
            actions = self.action_memory[batch]

            q_pred = self.Q(states) # out.shape = (batch_size, n_actions)
            q_pred = q_pred[batch_index, actions] # (batch_size, )
            q_next = self.Q(states_) # out.shape = (batch_size, n_actions)
            q_next[dones] = 0.0
            q_n = torch.max(q_next, dim = 1)[0] # out.shape = (batch_size, )
            q_target = rewards + self.gamma * q_n

            loss = F.mse_loss(q_target, q_pred)
            loss.backward()

            self.opimizer.step()
            
            self.eps = self.eps - self.eps_dec if self.eps > self.eps_end else self.eps_end