from gym import spaces
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch
from torch.distributions import Categorical


def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ReplayBuffer:
    """
    Simple storage for transitions from an environment.
    """

    def __init__(self, size):
        """
        Initialise a buffer of a given size for storing transitions
        :param size: the maximum number of transitions that can be stored
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer. Old transitions will be overwritten if the buffer is full.
        :param state: the agent's initial state
        :param action: the action taken by the agent
        :param reward: the reward the agent received
        :param next_state: the subsequent state
        :param done: whether the episode terminated
        """
        data = (state, action, reward, next_state, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, indices):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            data = self._storage[i]
            state, action, reward, next_state, done = data
            states.append(np.array(state, copy=False))
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions from the buffer.
        :param batch_size: the number of transitions to sample
        :return: a mini-batch of sampled transitions
        """
        indices = np.random.randint(0, len(self._storage) - 1, size=batch_size)
        return self._encode_sample(indices)



class DQN(nn.Module):
    def __init__(self, state, actions):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        # self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        # self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = nn.Linear(1024, actions.n)
        
        self.proj = nn.Linear(in_features=64*7*7, out_features=512)

        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.01)
        x = F.leaky_relu(self.conv2(x), 0.01)
        x = F.leaky_relu(self.conv3(x), 0.01)
        feature = x.view(x.size()[0],-1)
        proj = F.leaky_relu(self.proj(feature), 0.01)
        x = F.leaky_relu(self.fc1(feature), 0.01)
        return self.fc2(x), F.normalize(proj, p=2, dim=1)

class DQN2(nn.Module):
    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Discrete):                                                                                             
        super().__init__()
        assert type(
            observation_space) == spaces.Box, 'observation_space must be of type Box'
        assert len(
            observation_space.shape) == 3, 'observation space must have the form channels x width x height'
        assert type(
            action_space) == spaces.Discrete, 'action_space must be of type Discrete'
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=observation_space.shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU(0.01)
        )
        self.proj = nn.Linear(in_features=64*7*7, out_features=512)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64*7*7 , out_features=512),
            nn.ReLU()
        )
        self.classifier = nn.Linear(in_features=512, out_features=action_space.n)

    def forward(self, x):
        conv_out = self.encoder(x).view(x.size()[0],-1)
        proj = self.proj(conv_out)
        return self.classifier(self.fc1(conv_out)), F.normalize(proj, p=2, dim=1)
    
    
    
    
class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

num_channels = 4

class DQNBase(BaseNetwork):

    def __init__(self, num_channels):
        super(DQNBase, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            Flatten(),
        ).apply(initialize_weights_he)

    def forward(self, states):
        return self.net(states)


class QNetwork(BaseNetwork):

    def __init__(self, num_actions, shared=False,
                 dueling_net=False):
        super().__init__()

        if not shared:
            self.conv = DQNBase(num_channels)

        if not dueling_net:
            self.head = nn.Sequential(
                nn.Linear(7 * 7 * 64, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions))
        else:
            self.a_head = nn.Sequential(
                nn.Linear(7 * 7 * 64, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions))
            self.v_head = nn.Sequential(
                nn.Linear(7 * 7 * 64, 512),
                nn.ReLU(),
                nn.Linear(512, 1))
        self.projection = nn.Linear(7 * 7 * 64, 512)
        self.shared = shared
        self.dueling_net = dueling_net

    def forward(self, states, return_feature=False):
        if not self.shared:
            states = self.conv(states)

        if not self.dueling_net:
            if return_feature:
                return self.projection(states)
            return self.head(states)
        else:
            a = self.a_head(states)
            v = self.v_head(states)
            return v + a - a.mean(1, keepdim=True)


class TwinnedQNetwork(BaseNetwork):
    def __init__(self, num_actions, shared=False,
                 dueling_net=False):
        super().__init__()
        self.Q1 = QNetwork(num_actions, shared, dueling_net)
        self.Q2 = QNetwork(num_actions, shared, dueling_net)
        
    def forward(self, states, return_feature=False):
        q1 = self.Q1(states, return_feature)
        q2 = self.Q2(states, return_feature)
        return q1, q2


class CateoricalPolicy(BaseNetwork):

    def __init__(self, num_actions, shared=False):
        super().__init__()
        if not shared:
            self.conv = DQNBase(num_channels)

        self.head = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions))

        self.projection = nn.Linear(7 * 7 * 64, 512)
        
        self.shared = shared

    def act(self, states):
        if not self.shared:
            states = self.conv(states)

        action_logits = self.head(states)
        greedy_actions = torch.argmax(
            action_logits, dim=1, keepdim=True)
        return greedy_actions

    def sample(self, states, return_feature=False):
        if not self.shared:
            states = self.conv(states)
            
        if return_feature:
            return self.projection(states)
        
        action_probs = F.softmax(self.head(states), dim=1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # Avoid numerical instability.
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        
        
        return actions, action_probs, log_action_probs