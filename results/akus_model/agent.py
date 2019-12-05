import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt

class PolicyConv(torch.nn.Module):
    def __init__(self, action_space, hidden=64):
        super().__init__()
        self.action_space = action_space
        self.hidden = hidden
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 2)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, 2)
        self.reshaped_size = 128*11*11
        self.fc1_actor = torch.nn.Linear(self.reshaped_size, self.hidden)
        self.fc1_critic = torch.nn.Linear(self.reshaped_size, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        self.fc2_value = torch.nn.Linear(self.hidden, 1)

    def actor(self, x):
        #Save up on the grad
        with torch.no_grad():
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.conv3(x)
            x = F.relu(x)

            x = x.reshape(-1, self.reshaped_size)
            x_ac = self.fc1_actor(x)
            x_ac = F.relu(x_ac)
            x_mean = self.fc2_mean(x_ac)

            x_probs = F.softmax(x_mean, dim=-1)
            dist = Categorical(x_probs)
        return dist

    def critic(self, x):
        #Runs both, as the distribution is also needed for training.
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.reshape(-1, self.reshaped_size)

        x_ac = self.fc1_actor(x)
        x_ac = F.relu(x_ac)
        x_mean = self.fc2_mean(x_ac)
        x_probs = F.softmax(x_mean, dim=-1)
        dist = Categorical(x_probs)

        x_cr = self.fc1_critic(x)
        x_cr = F.relu(x_cr)
        value = self.fc2_value(x_cr)
        return dist, value.squeeze(-1)

class Agent(object):
    def __init__(self):
        self.policy_old = PolicyConv(3, 128).to("cpu")
        self.prev_obs = None

    def load_model(self):
        checkpoint = torch.load("checkpoint.485700", map_location="cpu")
        self.policy_old.load_state_dict(checkpoint["policy"])

    def reset(self):
        self.prev_obs = None

    def get_action(self, observation):
        if self.prev_obs is None:
            self.prev_obs = observation
        prev = self.prev_obs
        cur = observation
        x = self.preprocess_frame(prev, cur)
        x = x.unsqueeze(0)
        dist = self.policy_old.actor(x)
        action = torch.argmax(dist.probs)
        self.prev_obs = observation
        return action.numpy()

    def get_name(self):
        return "Björn Wimbleborg"

    def preprocess(self, observation):
        prepro = observation[::2,::2].mean(axis=-1)
        prepro[prepro < 60] = 0
        prepro[prepro >=60] = 1
        return prepro

    def diff(self, preprocessed, preprocessed_prev):
        return preprocessed - preprocessed_prev

    def preprocess_frame(self, prev, cur):
        #Stateless version of observation 
        prev = self.preprocess(prev)
        prev = torch.from_numpy(prev).float().unsqueeze(-1)
        prev = prev.transpose(0,2) #Conv input order

        cur = self.preprocess(cur)
        cur = torch.from_numpy(cur).float().unsqueeze(-1)
        cur = cur.transpose(0,2) #Conv input order

        out = self.diff(cur, prev)
        return out
