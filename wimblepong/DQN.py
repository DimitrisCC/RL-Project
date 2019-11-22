import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter, init
from torch.nn import functional as F
import math
import collections

### https://github.com/andri27-ts/Reinforcement-Learning/tree/master/Week3

class DQN(nn.Module):
	'''
	Deep Q newtork following the architecture used in the DeepMind paper (https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
	'''
	def __init__(self, input_shape, n_actions, noisy_net):
		super(DQN, self).__init__()

		# 3 convolutional layers. Take an image as input (NB: the BatchNorm layers aren't in the paper but they increase the convergence)
		self.conv = nn.Sequential(
			nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1),
			nn.BatchNorm2d(64),
			nn.ReLU())

		# Compute the output shape of the conv layers
		conv_out_size = self._get_conv_out(input_shape)

		# 2 fully connected layers
		self.fc = nn.Sequential(
				nn.Linear(conv_out_size, 512),
				nn.ReLU(),
				nn.Linear(512, n_actions))

	def _get_conv_out(self, shape):
		# Compute the output shape of the conv layers
		o = self.conv(torch.zeros(1, *shape)) # apply convolution layers..
		return int(np.prod(o.size())) # ..to obtain the output shape

	def forward(self, x):
		batch_size = x.size()[0]
		conv_out = self.conv(x).view(batch_size, -1) # apply convolution layers and flatten the results
		return self.fc(conv_out) # apply fc layers


class ReplayBuffer():
	'''
	Replay Buffer class to keep the agent memories memorized in a deque structure.
	'''
	def __init__(self, size, n_multi_step, gamma):
		self.buffer = collections.deque(maxlen=size)
		self.n_multi_step = n_multi_step
		self.gamma = gamma

	def __len__(self):
		return len(self.buffer)

	def append(self, memory):
		'''
		append a new 'memory' to the buffer
		'''
		self.buffer.append(memory)

	def sample(self, batch_size):
		'''
		Sample batch_size memories from the buffer.
		NB: It deals the N-step DQN
		'''
		# randomly pick batch_size elements from the buffer
		indices = np.random.choice(len(self.buffer), batch_size, replace=False)

		states = []
		actions = []
		next_states = []
		rewards = []
		dones = []

		# for each indices
		for i in indices:
			sum_reward = 0
			states_look_ahead = self.buffer[i].new_obs
			done_look_ahead = self.buffer[i].done

			# N-step look ahead loop to compute the reward and pick the new 'next_state' (of the n-th state)
			for n in range(self.n_multi_step):
				if len(self.buffer) > i+n:
					# compute the n-th reward
					sum_reward += (self.gamma**n) * self.buffer[i+n].reward
					if self.buffer[i+n].done:
						states_look_ahead = self.buffer[i+n].new_obs
						done_look_ahead = True
						break
					else:
						states_look_ahead = self.buffer[i+n].new_obs
						done_look_ahead = False

			# Populate the arrays with the next_state, reward and dones just computed
			states.append(self.buffer[i].obs)
			actions.append(self.buffer[i].action)
			next_states.append(states_look_ahead)
			rewards.append(sum_reward)
			dones.append(done_look_ahead)

		return (np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64), np.array(next_states, dtype=np.float32), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8))
