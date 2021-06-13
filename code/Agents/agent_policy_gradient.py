import torch
import torch as T
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.categorical import Categorical
from utils import *

class network_ssa(nn.Module):
    def __init__(self,input_size = 7, hidden_size = 120, num_layers = 2,
                num_classes=3,fc1_dims=40):
                
        super(network_ssa,self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size,hidden_size,num_layers,batch_first=True)
        
        self.fc1 = nn.Linear(hidden_size,fc1_dims)
        self.fc2 = nn.Linear(fc1_dims,num_classes)
        self.fc3 = nn.Linear(num_classes+4,30)
        self.fc4 = nn.Linear(30,20)
        self.fc5 = nn.Linear(20,num_classes)
        
    def forward(self,x1,x2):
    
        h0 = torch.zeros(self.num_layers,x1.size(0),self.hidden_size).cuda()
        
        out,_ = self.rnn(x1,h0)
        out = out[:,-1,:]
    
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        out = torch.cat([out,x2],dim=-1)

        out = torch.sigmoid(self.fc3(out))
        out = torch.sigmoid(self.fc4(out))
        out = torch.softmax(self.fc5(out),dim=-1)

        return out
        
class network_ssa2(nn.Module):
    def __init__(self,input_size = 7, hidden_size = 120, num_layers = 2,
                num_classes=3,fc1_dims=40):        
        super(network_ssa,self).__init__()
        
        self.fc1 = nn.Linear(140,3)
       
        self.fc2 = nn.Linear(4,30)
        self.fc3 = nn.Linear(30,30)
        self.fc4 = nn.Linear(30,3)
        
        
        
    def forward(self,x1,x2):
    
        x1 = x1.view((1,140))
        x2 = x2.view((1,4))
        
        x1 = torch.sigmoid(self.fc1(x1))
        
        x = torch.cat([x1,x2],dim=-1)
        
        out = torch.sigmoid(self.fc2(x2))
        out = torch.sigmoid(self.fc3(out))
        out = torch.softmax(self.fc4(out),dim=-1)

        return out
        
class agent_ssa:
    def __init__(self,input_size = 7, hidden_size = 120, num_layers = 2,
                num_classes=3,fc1_dims=40,gamma = 1, lr = 0.001):
        self.gamma = gamma
        self.learning_rate = lr
        self.policy = network_ssa(input_size, hidden_size, num_layers, num_classes,fc1_dims).cuda()
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate,weight_decay = 0)
        
        self.learn_cycle = 0
        
    def convert_observation(self,x):
        
        return torch.Tensor([x]).cuda()
        
    def choose_action(self,x1,x2,explore):
      
        probs = self.policy.forward(x1,x2)
        
        if explore:
            distribution = Categorical(probs = probs)

            action = distribution.sample()
            log_prob = distribution.log_prob(action)
            action = action.cpu().detach().numpy()[0]
        
            return action,log_prob
            
        else:
            probs = list(probs.cpu().detach().numpy()[0])
            action = probs.index(max(probs))
            
            return action
            
        
    def hyper(self):
        
        new_lr = max(1/self.learn_cycle,self.learning_rate)
        
        self.policy_optim.param_groups[0]['lr'] = new_lr
        
    def update(self,rewards,log_probs):
        self.policy_optim.zero_grad()
    
        p_loss = 0.0
        G= 0
        t = len(rewards)-1
        while t != -1:
            log_prob = log_probs[t]
            G = rewards[t] + self.gamma*G
            
            p_loss -= G*log_prob
            t -= 1

        p_loss.backward()
        self.policy_optim.step()
        
        self.learn_cycle += 1
        #self.hyper()

# ignore here just me testing if it learns simple stuff

if __name__ == '__main__':

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

    from enviroment import *
    
    start = 23 
    end = 43
    
    SSA = agent_ssa(lr = 0.001,hidden_size=5,gamma=0)
    env = Enviroment(start=start, end = end,stocks = stocks)
    port = [0.1]
    stock = stocks[0]
    
    
    for itter in range(500):
    
        rewards = []
        log_probs = []
        
        
        for day in range(env.start,env.end):
            
            current_position = np.random.choice(3)
            
            is_Friday = day_names_signals[env.timestep+1] == 1
            x1 = env.SSA_matrix(stock)
            x2 = [0,0,0,int(is_Friday)]
            x2[current_position] = 1
            
            x1 = torch.log(torch.Tensor([x1]).cuda())
            x2 = torch.tensor([x2],dtype= torch.float32).cuda()
            
            action,log_prob = SSA.choose_action(x1,x2,True)
            
            
            reward = quick()
            
            log_probs.append(log_prob)
            rewards.append(reward)
            
            
                
            env.step(port) # we jsut test we just keep the same port each day!
        
        env.reset()
        SSA.update(rewards,log_probs)
        log_probs = []
        rewards = []
    
    actions = []
    rewards = []
    for day in range(env.start,env.end):
        
        is_Friday = day_names_signals[env.timestep+1] == 1
        x1 = env.SSA_matrix(stock)
        x2 = [0,0,1,int(is_Friday)]

        x1 = torch.log(torch.Tensor([x1]).cuda())
        x2 = torch.tensor([x2],dtype=torch.float32).cuda()
        
        action = SSA.choose_action(x1,x2,False)
        
        reward = quick()
        
        rewards.append(reward)
        actions.append(action)

        env.step(port) # we jsut test we just keep the same port each day!
        
    print(f'At test got {sum(rewards)}')
            