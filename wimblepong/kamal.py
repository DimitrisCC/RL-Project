import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image

def prepro(frame):
     """ prepro 210x210x3 uint8 frame into 6400 (80x80) 1D float vector """
     im = Image.fromarray(frame)
     frame = np.asarray(im).copy()
     frame = cv2.resize(frame, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
     frame = cv2.resize(frame, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
     frame = frame.mean(2, keepdims=True)
     frame = frame.astype(np.float32)
     frame *= (1.0 / 255.0)
     frame = np.moveaxis(frame, -1, 0)
     return frame

class ObsNorm():
     def __init__(self, env=None):
          self.state_mean = 0
          self.state_std = 0
          self.alpha = 0.9999
          self.num_steps = 0

     def prepro(self, frame):
          return self.normalize(prepro(frame))
          
     def reset(self):
          self.state_mean = 0
          self.state_std = 0
          self.alpha = 0.9999
          self.num_steps = 0
          
     def normalize(self, observation):
          self.num_steps += 1
          self.state_mean = self.state_mean * self.alpha + \
          observation.mean() * (1 - self.alpha)
          self.state_std = self.state_std * self.alpha + \
          observation.std() * (1 - self.alpha)
          unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
          unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))
          
          return (observation - unbiased_mean) / (unbiased_std + 1e-8)




def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std/ torch.sqrt(out.pow(2).sum(1, keepdim=True))    
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        self.to('cpu')
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)

        num_outputs = action_space
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = self.conv1(inputs)
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(-1, 32 * 3 * 3)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
       
        critic = self.critic_linear(x)
        actor = self.actor_linear(x)
        return critic, actor, (hx, cx)


class Agent():
    def __init__(self):
         self.model = ActorCritic(1, 3)
         self.obsNorm = ObsNorm()     
         self.cx = torch.zeros(1, 256)
         self.hx = torch.zeros(1, 256)         
         
         self.model.eval()
         
         
    def load_model(self,filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def reset(self):
        self.obsNorm.reset()
        self.cx = torch.zeros(1, 256)
        self.hx = torch.zeros(1, 256) 
        
    def get_name(self):
        return 'Shaikh ur Razaail'
    
    def get_action(self, frame):
        frame = self.obsNorm.prepro(frame)
        state = torch.from_numpy(frame)
        with torch.no_grad():
            inputTensor = state.unsqueeze(0);
            value, logit, (hx, cx) = self.model((inputTensor, (self.hx, self.cx)))
        self.hx = hx
        self.cx = cx
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()
        
        return action[0,0]