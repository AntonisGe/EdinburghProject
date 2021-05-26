import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from utils import *

class RNN_SSA(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(RNN_SSA,self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
        
        self.fc = nn.Linear(hidden_size,num_classes)
        
        
    def forward(self,x):
    
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size).cuda()
        
        out,_ = self.rnn(x,h0)
        out = out[:,-1,:]
        out = self.fc(out)
        
        return out
        
class RNN_CAA(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes,n_stocks):
        super(RNN_CAA,self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size,self.hidden_size,num_layers,batch_first=True)
        
        self.fc = nn.Linear(self.hidden_size+2+n_stocks,20)
        self.fc2 = nn.Linear(20,num_classes)
        
        
    def forward(self,x,add):
    
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size).cuda()
        
        out,_ = self.rnn(x,h0)
        out = out[:,-1,:]
        
        out = torch.cat([out[0],add],dim = 0)
        out = torch.sigmoid(self.fc(out))
        out = self.fc2(out)
        
        return out
        
if __name__ == '__main__':
    from enviroment import *
        
    env = Enviroment()
    test = torch.Tensor([env.SSA_matrix('AAPL')]).cuda()
    
    model = RNN_CAA(input_size=6,hidden_size=40,num_layers=2,num_classes=3,n_stocks=3).cuda()
    a = model.forward(test,1,1,[0.1,0.2,-0.1])
    