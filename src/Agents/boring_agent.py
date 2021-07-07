from networks import *
from utils import *

class addicted_long_agent:
    def __init__(self,directory,gamma=0.95,max_mem_size = 100000,C=30,batch_size=30,epsilon=0.1, eps_min = 0.003, eps_dec = 5e-5
    ,num_layers=2,hidden_size=120,added_size=0,fc1_dims=100,fc2_dims=40,lr=0.001,wd=0,lookback=20,x2_size=4,scheduler_max = 100,scheduler_initial=100,multiplication_constant = 0.8):
    
        input_size = 50
        num_output = 3
        
        self.Q_eval = network(input_size,num_layers,hidden_size,num_output,added_size,x2_size,fc1_dims,fc2_dims,lr,wd)
        
    def store_transition(self,x1,x2,action,reward,new_x1,new_x2,done,circle):
        pass        
    def choose_action(self, x1,x2,explore = True):            
        return 2
    def prob_finder(self):
        return None
    def learn(self):
        pass
    def reset_scheduler(self,value= None):
        pass        
    def save_buffer(self):
        pass
    def load_buffer(self):
        pass
    def save_weights(self,name = 'last'):
        pass
    def load_weights(self,directory = None, name = 'last'):
        pass
        
class random_agent:
    def __init__(self,directory,gamma=0.95,max_mem_size = 100000,C=30,batch_size=30,epsilon=0.1, eps_min = 0.003, eps_dec = 5e-5
    ,num_layers=2,hidden_size=120,added_size=0,fc1_dims=100,fc2_dims=40,lr=0.001,wd=0,lookback=20,x2_size=4,scheduler_max = 100,scheduler_initial=100,multiplication_constant = 0.8):
    
        input_size = 50
        num_output = 3
        
        self.Q_eval = network(input_size,num_layers,hidden_size,num_output,added_size,x2_size,fc1_dims,fc2_dims,lr,wd)
        
    def store_transition(self,x1,x2,action,reward,new_x1,new_x2,done,circle):
        pass        
    def choose_action(self, x1,x2,explore = True):            
        return np.random.choice(3)
    def prob_finder(self):
        return None
    def learn(self):
        pass
    def reset_scheduler(self,value= None):
        pass        
    def save_buffer(self):
        pass
    def load_buffer(self):
        pass
    def save_weights(self,name = 'last'):
        pass
    def load_weights(self,directory = None, name = 'last'):
        pass