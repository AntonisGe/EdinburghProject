from networks import *
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class QUAactionNoise(object):
    def __init__(self,mu,sigma=0.15,theta=0.2,dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        
        self.reset()
        
    def __call_(self):
        x = self.x_prev + self.theta*(self.mu - self.x_prev)*self.dt + \
            self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
            
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class ReplayBuffer(object):
    def __init__(self,max_size,input_shape,n_actions):
        self.mem_size = max_size
        self.mem_cntr=0
        self.state_memory = np.zeros((self.mem_size,*input_shape))
        self.new_state_memory = np.zeros((self.mem_size,*input_shape))
        self.action_memory = np.zeros((self.mem_size,n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size,dtype = np.float32)

    def store_memory(self,state,action,reward,state_,done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.done_memory[index] = 1-done
        
        self.mem_cntr += 1
    
    def sample_buffer(self,batch_size):
        max_mem = min(self.mem_cntr,batch_size)
        
        batch = np.random.choice(max_mem,batch_size)
        
        states = self.state_memory[batch]
        reward = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]
        new_states = self.new_state_memory[batch]
        
        return states,actions,rewards,new_states,terminal
        
class DDPG_Agent:
    pass
    