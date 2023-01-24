import numpy as np
import torch
from DQNNetwork import DQNNetwork
from utils import ReplayBuffer


class DDQNAgent(object):
    def __init__(self, gamma, eps, lr, n_actions, input_dims,
                 buffer_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.eps = eps
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(buffer_size, input_dims)

        eval_name = algo + "_" + env_name + "_eval"
        self.eval = DQNNetwork(self.lr, self.n_actions, eval_name, self.input_dims, self.chkpt_dir)

        next_name = algo + "_" + env_name + "_next"
        self.next = DQNNetwork(self.lr, self.n_actions, next_name, self.input_dims, self.chkpt_dir)

    #epsilon greedy action to either explore or exploit
    def epsilon_greedy(self, observation):
        if np.random.rand() > self.eps:
            state = torch.tensor(np.array([observation]), dtype=torch.float32).to(self.eval.device)
            actions = self.eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def add_transition(self, state, action, reward, state_, done):
        self.memory.add_transition(state, action, reward, state_, done)

    #take data form replay buffer and conver to device tensor
    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        states = torch.tensor(state).to(self.eval.device)
        rewards = torch.tensor(reward).to(self.eval.device)
        dones = torch.tensor(done).to(self.eval.device)
        actions = torch.tensor(action).to(self.eval.device)
        next_states = torch.tensor(new_state).to(self.eval.device)
        return states, actions, rewards, next_states, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.next.load_state_dict(self.eval.state_dict())

    #reduce epsilon - chance for exploitation
    def decrement_epsilon(self):
        self.eps = self.eps - self.eps_dec if self.eps > self.eps_min else self.eps_min

    #saving models
    def save_models(self):
        self.eval.save_checkpoint()
        self.next.save_checkpoint()

    #load models
    def load_models(self):
        self.eval.load_checkpoint()
        self.next.load_checkpoint()

    #single learning step
    def learn(self):
        if self.memory.buffer_cntr < self.batch_size:
            return

        self.eval.optimizer.zero_grad()
        self.replace_target_network()

        states, actions, rewards, next_states, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        pred = self.eval.forward(states)[indices, actions]
        #here is a difference
        next_p = self.next.forward(next_states)
        eval_p = self.eval.forward(next_states)

        max_actions = torch.argmax(eval_p, dim=1)
        next_p[dones] = 0.0
        target = rewards + self.gamma * next_p[indices, max_actions]

        loss = self.eval.loss(target, pred).to(self.eval.device)
        loss.backward()
        self.eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
