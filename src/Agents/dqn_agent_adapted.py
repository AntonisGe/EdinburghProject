import torch as T
import torch.nn as nn
import numpy as np
from copy import deepcopy
from networks import *
from numpy import load,save

class DQN:
    def __init__(self,directory,gamma=0.95,max_mem_size = 100000,C=30,batch_size=30,epsilon=0.1, eps_min = 0.003, eps_dec = 5e-5
    ,num_layers=2,hidden_size=120,added_size=0,fc1_dims=100,fc2_dims=40,lr=0.001,wd=0,lookback=20,x2_size=4,scheduler_max = 100,scheduler_initial=100,multiplication_constant = 0.8,input_size = 20, num_output = 6):
    
        self.directory = directory + '\\'
        
        self.multiplication_constant = multiplication_constant
        self.gamma = gamma
        self.epsilon = epsilon
        self.wd = wd
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.lr = lr   
        self.C = C
        self.counter = 0
        
        self.lookback = lookback
        self.input_size = input_size
        self.x2_size = x2_size
        self.num_output = num_output
        
        self.action_space = [i for i in range(num_output)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size      
        self.mem_cntr = 0
        
        self.Q_eval = simple_network(input_size,fc1_dims,fc2_dims,num_output)
        self.Q_target = deepcopy(self.Q_eval)
        self.scheduler_max = scheduler_max
        self.reset_scheduler(scheduler_initial)
    
        self.input_size = input_size
        
    
        self.reset_memory()
        
    def reset_memory(self):
    
        self.mem_cntr = 0
    
        self.x_memory = np.zeros((self.mem_size,self.input_size),dtype = np.float32)
        self.new_x_memory = np.zeros((self.mem_size,self.input_size),dtype = np.float32)
        
        self.action_memory = np.zeros(self.mem_size, dtype = np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype = np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.bool)
        
                
    def store_transition(self,x,action,reward,new_x,done):
                
        if self.mem_cntr < self.mem_size:
            index = self.mem_cntr % self.mem_size
        else:
            index = np.random.choice(range(self.mem_size))
            
        self.x_memory[index] = x
        self.new_x_memory[index] = new_x
     
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        
        self.mem_cntr += 1
        
    def choose_action(self, x,explore = True):
    
        x = T.tensor(x,dtype = T.float32).to(self.Q_eval.device)
        
        rand = np.random.random()
        
        if explore:
            epsilon = self.epsilon
        else:
            epsilon = 0
        
        actions = self.Q_eval.forward(x)
        action = T.argmax(actions).item()
        
        if np.random.random() > epsilon:
            random = 0
        else:
            random = np.random.choice((-3,3))
            
        action += random
        
        action = max(action,0)
        action = min(action,self.num_output-1)
        
        return action
              
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr,self.mem_size)    
        batch = np.random.choice(max_mem, self.batch_size, replace = False)
        
        batch_index = np.arange(self.batch_size,dtype=np.int32)
        
        x_batch = T.tensor(self.x_memory[batch]).to(self.Q_eval.device)
        new_x_batch = T.tensor(self.new_x_memory[batch]).to(self.Q_eval.device)
        
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        
        q_eval = self.Q_eval.forward(x_batch)[batch_index,action_batch]
        q_next = self.Q_target.forward(new_x_batch)
        
        #print(q_next,terminal_batch)

        q_next[terminal_batch] = 0.0

        
        q_target = reward_batch + self.gamma*T.max(q_next,dim=1)[0]
        
        loss = self.Q_eval.loss(q_target,q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        
        if self.counter % self.C == 0:
            self.Q_target = deepcopy(self.Q_eval)
        
        self.counter += 1
        self.scheduler.step()
        
    def reset_scheduler(self,value= None):
    
        if value == None: value = self.scheduler_max
    
        self.scheduler = T.optim.lr_scheduler.CosineAnnealingLR(self.Q_eval.optimizer, value)
        
    def save_buffer(self):
        save(self.directory + 'x1_memory.npy',self.x1_memory)
        save(self.directory + 'x2_memory.npy',self.x2_memory)
        save(self.directory + 'new_x1_memory.npy',self.new_x1_memory)
        save(self.directory + 'new_x2_memory.npy',self.new_x2_memory)
        save(self.directory + 'action_memory.npy',self.action_memory)
        save(self.directory + 'reward_memory.npy',self.reward_memory)
        save(self.directory + 'terminal_memory.npy',self.terminal_memory)
        save(self.directory + 'circle_memory.npy',self.circle_memory)
        
    def load_buffer(self):
        self.x1_memory = load(self.directory + 'x1_memory.npy')
        self.x2_memory = load(self.directory + 'x2_memory.npy')
        self.new_x1_memory = load(self.directory + 'new_x1_memory.npy')
        self.new_x2_memory = load(self.directory + 'new_x2_memory.npy')
        self.action_memory = load(self.directory + 'action_memory.npy')
        self.reward_memory = load(self.directory + 'reward_memory.npy')
        self.terminal_memory = load(self.directory + 'terminal_memory.npy')
        self.circle_memory = load(self.directory + 'circle_memory.npy')
        
    def save_weights(self,name = 'last'):
        T.save(self.Q_eval.state_dict(),self.directory + name + '.pt')
        
    def load_weights(self,directory = None, name = 'last'):
        
        if directory == None: directory = self.directory
        
        path = directory + name + '.pt'
    
        self.Q_eval.load_state_dict(T.load(path))
        self.Q_target.load_state_dict(T.load(path))

      
class DQN_Agent_Adapted:
    def __init__(self,directory,gamma=0.95,max_mem_size = 100000,C=30,batch_size=30,epsilon=0.1, eps_min = 0.003, eps_dec = 5e-5
    ,num_layers=2,hidden_size=120,added_size=0,fc1_dims=100,fc2_dims=40,lr=0.001,wd=0,lookback=20,x2_size=4,scheduler_max = 100,scheduler_initial=100,multiplication_constant = 0.8):
    
        input_size = 56
        num_output=3
        self.directory = directory + '\\'
        
        self.multiplication_constant = multiplication_constant
        self.gamma = gamma
        self.epsilon = epsilon
        self.wd = wd
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.lr = lr   
        self.C = C
        self.counter = 0
        
        self.lookback = lookback
        self.input_size = input_size
        self.x2_size = x2_size
        
        self.action_space = [i for i in range(num_output)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size      
        self.mem_cntr = 0
        
        self.Q_eval = network(input_size,num_layers,hidden_size,num_output,added_size,x2_size,fc1_dims,fc2_dims,lr,wd)
        #self.Q_eval = hacked_network(input_size,num_layers,hidden_size,lr,wd)
        self.Q_target = deepcopy(self.Q_eval)
        self.scheduler_max = scheduler_max
        self.reset_scheduler(scheduler_initial)
    
        self.reset_memory()
        
    def reset_memory(self):
    
        self.mem_cntr = 0
    
        self.x1_memory = np.zeros((self.mem_size,self.lookback,self.input_size),dtype = np.float32)
        self.x2_memory = np.zeros((self.mem_size,self.x2_size),dtype = np.float32)
        self.new_x1_memory = np.zeros((self.mem_size,self.lookback,self.input_size),dtype = np.float32)
        self.new_x2_memory = np.zeros((self.mem_size,self.x2_size),dtype = np.float32)
        
        self.action_memory = np.zeros(self.mem_size, dtype = np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype = np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.bool)
        
        self.circle_memory = np.zeros(self.mem_size, dtype = np.float32)
                
    def store_transition(self,x1,x2,action,reward,new_x1,new_x2,done,circle):
                
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
        
        self.circle_memory[index] = circle
        
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
        
    def prob_finder(self):
        
        max_mem = min(self.mem_cntr,self.mem_size)   

        soft = self.circle_memory[:max_mem]
        
        min_value = min(soft)
        
        soft = soft- min_value + 1
        soft = soft*self.multiplication_constant
        soft = np.exp(soft)
        
        return soft/sum(soft)
        
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr,self.mem_size)    
        batch = np.random.choice(max_mem, self.batch_size, replace = False, p = self.prob_finder())
        
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
        self.scheduler.step()
        
    def reset_scheduler(self,value= None):
    
        if value == None: value = self.scheduler_max
    
        self.scheduler = T.optim.lr_scheduler.CosineAnnealingLR(self.Q_eval.optimizer, value)
        
    def save_buffer(self):
        save(self.directory + 'x1_memory.npy',self.x1_memory)
        save(self.directory + 'x2_memory.npy',self.x2_memory)
        save(self.directory + 'new_x1_memory.npy',self.new_x1_memory)
        save(self.directory + 'new_x2_memory.npy',self.new_x2_memory)
        save(self.directory + 'action_memory.npy',self.action_memory)
        save(self.directory + 'reward_memory.npy',self.reward_memory)
        save(self.directory + 'terminal_memory.npy',self.terminal_memory)
        save(self.directory + 'circle_memory.npy',self.circle_memory)
        
    def load_buffer(self):
        self.x1_memory = load(self.directory + 'x1_memory.npy')
        self.x2_memory = load(self.directory + 'x2_memory.npy')
        self.new_x1_memory = load(self.directory + 'new_x1_memory.npy')
        self.new_x2_memory = load(self.directory + 'new_x2_memory.npy')
        self.action_memory = load(self.directory + 'action_memory.npy')
        self.reward_memory = load(self.directory + 'reward_memory.npy')
        self.terminal_memory = load(self.directory + 'terminal_memory.npy')
        self.circle_memory = load(self.directory + 'circle_memory.npy')
        
    def save_weights(self,name = 'last'):
        T.save(self.Q_eval.state_dict(),self.directory + name + '.pt')
        
    def load_weights(self,directory = None, name = 'last'):
        
        if directory == None: directory = self.directory
        
        path = directory + name + '.pt'
    
        self.Q_eval.load_state_dict(T.load(path))
        self.Q_target.load_state_dict(T.load(path))

                  
if __name__ == '__main__':

    # make sure it learns simple stuff

    agent = DQN_Agent_Adapted('D:',batch_size=30,x2_size=5,scheduler_initial=5000,lr = 0.001,lookback=10)
    
    def generate():
        x1 = np.random.randn(10,4)
        i = np.random.choice(range(10))
        if i == 0:
            i1 = 1
            i2 = 0
        else:
            i1 = 0
            i2 = i
            
        x2 = [i1,i2,np.random.choice(range(2)),np.random.choice(range(2)),np.random.choice(range(2))]
    
        return i,x1,x2
        
    for _ in range(5000):
        i,x1,x2 = generate()
        _,x1_next,x2_next = generate()
        
        done = False
        
        agent.Q_eval.eval()
        with T.no_grad():
            action = agent.choose_action(x1,x2)
        agent.Q_eval.train()

        if i == 0:
            if action == 0:
                reward = -1
            else:
                reward = 0
        else:
            if action ==0:
                reward = 0
            else:
                reward = -1
        
        agent.store_transition(x1,x2,action,reward,x1_next,x2_next,done,0)
        agent.learn()
        
    treward = 0
    for _ in range(500):
        i,x1,x2 = generate()
        _,x1_next,x2_next = generate()
        
        done = False
        
        agent.Q_eval.eval()
        with T.no_grad():
            action = agent.choose_action(x1,x2,False)
        agent.Q_eval.train()

        if i == 0:
            if action == 0:
                reward = -9999999999
            else:
                reward = 0
        else:
            if action ==0:
                reward = 0
            else:
                reward = -999999999
        print(i,action,reward)
        
        treward += reward

    
