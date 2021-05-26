# took the code from https://github.com/chingyaoc/pytorch-REINFORCE/blob/master/reinforce_continuous.py
from agents import Agents
from networks import *

import sys
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
import torchvision.transforms as T
from torch.autograd import Variable

pi = Variable(torch.FloatTensor([math.pi])).cuda()

def normal(x, mu, sigma_sq):
    a = (-1*(Variable(x)-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq*pi)**0.5
    return a*b

class CAA_Agent(Agents):
    def __init__(self,hidden_size,num_layers,variance = 0.01,learning_rate=0.01,alpha=0.003,gamma = 1,n_stocks = len(stocks)):
    
        Agents.__init__(self,hidden_size,num_layers,learning_rate,alpha,gamma)
        self.n_stocks = n_stocks
        output = 3
    
        self.variance = variance
    
        self.policy = RNN_CAA(self.n_stocks,hidden_size,num_layers,output,n_stocks = n_stocks)
        self.policy = self.policy.cuda()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

    def choose_action(self, state,z1,z3,port,explore):
        state = self.convert_observation(state)
                
        z1= torch.Tensor([z1])
        z3= torch.Tensor([z3])
        
        port = torch.Tensor(port)
        add = torch.cat([z1,z3])
        add = torch.cat([add,port])
        
        mu = self.policy(Variable(state).cuda(),Variable(add).cuda())

        if explore:
            eps = torch.randn(mu.size())
        else:
            eps = torch.zero(mu.size())
        
        #print(mu + ((self.variance)**0.5)*Variable(eps).cuda())
        # calculate the probability
        action = F.softmax((mu + ((self.variance)**0.5)*Variable(eps).cuda()).data,dim=-1)
        print(action)
        prob = normal(action, mu, self.variance)
        entropy = -0.5*((self.variance+2*pi).log()+1)

        log_prob = prob.log()
        return action, log_prob, entropy

    def update(self, rewards, log_probs, entropies):
        R = torch.zeros(1, 1)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = self.gamma * R + rewards[i]
            #loss = loss - (log_probs[i]*(Variable(R).expand_as(log_probs[i])).cuda()).sum() - (0.0001*entropies[i].cuda()).sum()
            loss = loss - (log_probs[i]*(Variable(R)).cuda()).sum() - (0.0001*entropies[i].cuda()).sum()
        loss = loss / len(rewards)
		
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm(self.policy.parameters(), 40)
        self.optimizer.step()
        
        
if __name__ == '__main__':
    from enviroment import Enviroment 
    env = Enviroment()
    
    obs = env.CAA_matrix()
    
    CAA = CAA_agent(40,2,n_stocks=3)
    action,log_prob,entropy = CAA.choose_action(state= obs,explore =True)
    
    CAA.update([-2],[log_prob],[entropy])