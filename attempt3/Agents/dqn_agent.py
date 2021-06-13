import torch as T
import torch.nn as nn
import numpy as np
from copy import deepcopy
from networks import *
from numpy import load,save
        
class DQN_Agent:
    def __init__(self,directory,gamma=0.95,max_mem_size = 100000,C=30,batch_size=30,epsilon=0.1, eps_min = 0.003, eps_dec = 5e-5
    ,num_layers=2,hidden_size=120,added_size=0,fc1_dims=100,fc2_dims=40,lr=0.001,wd=0,lookback=20,x2_size=4):
    
        input_size = 7
        num_output = 3
        
        self.directory = directory + '\\'

        self.gamma = gamma
        self.epsilon = epsilon
        self.wd = wd
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.lr = lr   
        self.C = C
        self.counter = 0
        
        self.action_space = [i for i in range(num_output)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size      
        self.mem_cntr = 0
        
        self.Q_eval = network(input_size,num_layers,hidden_size,num_output,added_size,x2_size,fc1_dims,fc2_dims,lr,wd)
        #self.Q_eval = hacked_network(input_size,num_layers,hidden_size,lr,wd)
        self.Q_target = deepcopy(self.Q_eval)
        
        self.x1_memory = np.zeros((self.mem_size,lookback,input_size),dtype = np.float32)
        self.x2_memory = np.zeros((self.mem_size,x2_size),dtype = np.float32)
        self.new_x1_memory = np.zeros((self.mem_size,lookback,input_size),dtype = np.float32)
        self.new_x2_memory = np.zeros((self.mem_size,x2_size),dtype = np.float32)
        
        self.action_memory = np.zeros(self.mem_size, dtype = np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype = np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.bool)
                
    def store_transition(self,x1,x2,action,reward,new_x1,new_x2,done):
                
        if self.mem_cntr < self.mem_size:
            index = self.mem_cntr % self.mem_size
        else:
            index = np.random.choice(range(self.mem_size))
            
        self.x1_memory[index] = x1
        self.x2_memory[index] = x2
        self.new_x1_memory[index] = new_x1
        self.new_x2_memory[index] = new_x2
        
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        
        self.mem_cntr += 1
        
    def choose_action(self, x1,x2,explore = True):
    
        x1 = T.tensor([x1],dtype = T.float32).to(self.Q_eval.device)
        x2 = T.tensor([x2],dtype = T.float32).to(self.Q_eval.device)
        
        rand = np.random.random()
        
        if explore:
            epsilon = self.epsilon
        else:
            epsilon = 0
        
        if np.random.random() > epsilon:
            actions = self.Q_eval.forward(x1,x2)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
            
        return action
        
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()
        
        max_mem = min(self.mem_cntr,self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace = False)
        
        batch_index = np.arange(self.batch_size,dtype=np.int32)
        
        x1_batch = T.tensor(self.x1_memory[batch]).to(self.Q_eval.device)
        x2_batch = T.tensor(self.x2_memory[batch]).to(self.Q_eval.device)
        new_x1_batch = T.tensor(self.new_x1_memory[batch]).to(self.Q_eval.device)
        new_x2_batch = T.tensor(self.new_x2_memory[batch]).to(self.Q_eval.device)
        
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        
        q_eval = self.Q_eval.forward(x1_batch,x2_batch)[batch_index,action_batch]
        q_next = self.Q_target.forward(new_x1_batch,new_x2_batch)
        
        q_next[terminal_batch] = 0.0
        
        q_target = reward_batch + self.gamma*T.max(q_next,dim=1)[0]
        
        loss = self.Q_eval.loss(q_target,q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        
        if self.counter % self.C == 0:
            self.Q_target = deepcopy(self.Q_eval)
        
        self.counter += 1
        
    def save_buffer(self):
        save(self.directory + 'x1_memory.npy',self.x1_memory)
        save(self.directory + 'x2_memory.npy',self.x2_memory)
        save(self.directory + 'new_x1_memory.npy',self.new_x1_memory)
        save(self.directory + 'new_x2_memory.npy',self.new_x2_memory)
        save(self.directory + 'action_memory.npy',self.action_memory)
        save(self.directory + 'reward_memory.npy',self.reward_memory)
        save(self.directory + 'terminal_memory.npy',self.terminal_memory)
        
    def load_buffer(self):
        self.x1_memory = load(self.directory + 'x1_memory.npy')
        self.x2_memory = load(self.directory + 'x2_memory.npy')
        self.new_x1_memory = load(self.directory + 'new_x1_memory.npy')
        self.new_x2_memory = load(self.directory + 'new_x2_memory.npy')
        self.action_memory = load(self.directory + 'action_memory.npy')
        self.reward_memory = load(self.directory + 'reward_memory.npy')
        self.terminal_memory = load(self.directory + 'terminal_memory.npy')
        
    def save_weights(self,name = 'last'):
        T.save(self.Q_eval.state_dict(),self.directory + name + '.pt')
        
    def load_weights(self,name = 'last'):
        
        path = self.directory + name + '.pt'
    
        self.Q_eval.load_state_dict(T.load(path))
        self.Q_target.load_state_dict(T.load(path))

                  
if __name__ == '__main__':

    # make sure it learns simple stuff

    def quick():
        if is_Friday:
            if action == 1:
                reward = 1
            else:
                reward = 0
        else:
            if action ==0:
                reward = 1
            else:
                reward = 0
              
        return reward
    
    import sys,os
    sys.path.append(r'D:\Project\attempt3\Enviroment')
    from enviroment import *
    from utils import *
    
    start = 23 
    end = 43
    port = [0.1]
    stock = stocks[0]
    
    env = Enviroment(start = start,end = end)
    SSA = DQN_Agent(lr = 0.001,epsilon = 0.05,eps_min = 0.01,fc1_dims = 100,fc2_dims = 20)
    
    
    for itter in range(500):
    
        treward = 0
    
        current_position = np.random.choice(3)            
        is_Friday = day_names_signals[env.timestep+1] == 1        
        x1 = env.SSA_matrix(stock)
        x2 = [0,0,0,int(is_Friday)]
        x2[current_position] = 1
    
        for day in range(env.start,env.end):
            
            action = SSA.choose_action(x1,x2)
            reward = quick()
            
            treward += reward
            
            env.step(port) # we jsut test we just keep the same port each day!
            
            old_x1 = x1
            old_x2 = x2
            
            current_position = np.random.choice(3)            
            is_Friday = day_names_signals[env.timestep+1] == 1       
            x1 = env.SSA_matrix(stock)
            x2 = [0,0,0,int(is_Friday)]
            x2[current_position] = 1
            
            done = False
            if day == env.end - 1:done = True
            
            SSA.store_transition(old_x1,old_x2,action,reward,x1,x2,done)
            SSA.learn()
        
        env.reset()
        print(f'At itteration {itter} we got reward {treward}.')

    
