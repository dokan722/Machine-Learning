import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DuelingDQNNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DuelingDQNNetwork, self).__init__()
        #for saving our network
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        #convolutional part of network
        self.cnn = nn.Sequential(
            nn.Conv2d(input_dims[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )

        #linear part
        fc_input_dims = self.conv_dims(input_dims)
        self.fc1 = nn.Linear(fc_input_dims, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def conv_dims(self, input_dims):
        base = torch.zeros(*input_dims)
        dims = self.cnn(base)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv = self.cnn(state)
        conv_state = conv.view(conv.size()[0], -1)
        flat1 = F.relu(self.fc1(conv_state))
        flat2 = F.relu(self.fc2(flat1))
        V = self.V(flat2)
        A = self.A(flat2)
        return V, A

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))