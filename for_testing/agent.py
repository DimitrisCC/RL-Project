# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import optim
import cv2
from collections import deque

from model import RainbowDQN
from utils import preprocess_frame


class Agent():
    def __init__(self, args=None, env=None):
        if args is None:
            self.name = 'Agent Rainbow'
            self.action_space=3
            self.atoms=51
            self.Vmin=-10
            self.Vmax=10
            self.crop_opponent=False
            self.window=4
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
            self.support = torch.linspace(self.Vmin, self.Vmax, self.atoms).to(device=self.device)
            self.state_buffer = deque([], maxlen=self.window)
            self.val_state_buffer = deque([], maxlen=self.window)
            self.online_net = RainbowDQN().to(device=self.device)
            self.last_stacked_obs = None
            self.online_net.eval()
        else:
            self.name = 'Agent Rainbow'
            self.action_space = env.action_space.n
            self.atoms = args.atoms
            self.Vmin = args.V_min
            self.Vmax = args.V_max
            self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
            self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
            self.batch_size = args.batch_size
            self.n = args.multi_step
            self.discount = args.discount
            self.device = args.device
            self.window = args.history_length
            self.crop_opponent = args.crop_opponent
            self.state_buffer = deque([], maxlen=self.window)
            self.val_state_buffer = deque([], maxlen=self.window)
            self.last_stacked_obs = None

            self.online_net = RainbowDQN(args, self.action_space).to(device=args.device)
            if args.model:  # Load pretrained model if provided
                if os.path.isfile(args.model):
                    state_dict = torch.load(args.model, map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
                    # if 'conv1.weight' in state_dict.keys():
                    # 	for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
                    # 		state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
                    # 		del state_dict[old_key]  # Delete old keys for strict load_state_dict
                    self.online_net.load_state_dict(state_dict)
                    print("Loading pretrained model: " + args.model)
                else:  # Raise error if incorrect model path provided
                    raise FileNotFoundError(args.model)

            self.online_net.train()

            self.target_net = RainbowDQN(args, self.action_space).to(device=args.device)
            self.update_target_net()
            self.target_net.train()
            for param in self.target_net.parameters():
                param.requires_grad = False

            self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

            self.reset()
        
    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(50, 50, device=self.device))
    
    def reset_val_buffer(self):
        for _ in range(self.window):
            self.val_state_buffer.append(torch.zeros(50, 50, device=self.device))

    def get_name(self):
        return self.name

    def reset(self):
        self._reset_buffer()
        self.reset_val_buffer()

    # TODO 
    def load_model(self):
        # if os.path.isfile(args.model):
        #     state_dict = torch.load(args.model, map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
        #     self.online_net.load_state_dict(state_dict)
        #     print("Loading pretrained model: " + args.model)
        # else:  # Raise error if incorrect model path provided
        #     raise FileNotFoundError(args.model)
        
        if os.path.isfile('/u/11/jaina5/unix/RL-Project/results/Rainbow-1/checkpoint.pth'):
            state_dict = torch.load('/u/11/jaina5/unix/RL-Project/results/Rainbow-1/checkpoint.pth', map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
            self.online_net.load_state_dict(state_dict)
            print("Loading pretrained model: ")
        else:  # Raise error if incorrect model path provided
            raise FileNotFoundError("Not found")
        
    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()

    # Acts based on single state (no batch)
    def get_action(self, observation):
        observation = preprocess_frame(observation, self.device, self.crop_opponent)
        self.state_buffer.append(observation)
        observation = torch.stack(list(self.state_buffer), 0)
        self.last_stacked_obs = observation
        with torch.no_grad():
            return (self.online_net(observation.unsqueeze(0)) * self.support).sum(2).argmax(1).item()

    # Acts with an ε-greedy policy (used for evaluation only)
    def act_e_greedy(self, observation, epsilon=0.001):  # High ε can reduce evaluation scores drastically
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_space)
        else:
            observation = preprocess_frame(observation, self.device, self.crop_opponent)
            self.val_state_buffer.append(observation)
            observation = torch.stack(list(self.val_state_buffer), 0)
            with torch.no_grad():
                return (self.online_net(observation.unsqueeze(0)) * self.support).sum(2).argmax(1).item()

    def train_step(self, mem):
        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)
 
        # Calculate current state probabilities (online network noise already sampled)
        log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
        log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

        with torch.no_grad():
            # Calculate nth next state probabilities
            pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
            dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
            argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            self.target_net.reset_noise()  # Sample new target net noise
            pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
            pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

            # Compute Tz (Bellman operator T applied to z)
            Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = states.new_zeros(self.batch_size, self.atoms)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        self.online_net.zero_grad()
        (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        self.optimiser.step()

        mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Save model parameters on current device (don't move model between devices)
    def save(self, path, name='model.pth'):
        torch.save(self.online_net.state_dict(), os.path.join(path, name))

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        with torch.no_grad():
            return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()
