import torch as T
import torch.nn as nn

class network(nn.Module):
    def __init__(self,input_size,num_layers,hidden_size,num_output,added_size,x2_size,fc1_dims,fc2_dims,lr,wd):
                
        super(network,self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size,self.hidden_size,self.num_layers,batch_first=True)
              
        self.fc = nn.Linear(self.hidden_size,num_output+added_size)
              
        self.fc1 = nn.Linear(num_output+x2_size+added_size,fc1_dims)
        self.fc2 = nn.Linear(fc1_dims,fc2_dims)
        self.fc3 = nn.Linear(fc2_dims,num_output)
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.optimizer = T.optim.Adam(self.parameters(), lr = lr, weight_decay = wd)
        self.loss = nn.MSELoss()
        
        self.to(self.device)
        
    def forward(self,x1,x2):
    
        self.rnn.flatten_parameters() # it throws up a mistake if I do not use this 
        # solution found at https://discuss.pytorch.org/t/rnn-module-weights-are-not-part-of-single-contiguous-chunk-of-memory/6011/13
        h0 = T.zeros(self.num_layers,x1.size(0),self.hidden_size).to(self.device)
        
        x1,_ = self.rnn(x1,h0)
        x1 = x1[:,-1,:]
   
        x1 = self.fc(x1)
        
        x = T.cat([x1,x2],dim=-1)
        out = T.relu(self.fc1(x))
        out = T.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out
        
class hacked_network(nn.Module):
    def __init__(self,input_size=7,num_layers=2,hidden_size=120,lr=0.001,wd=0):
                
        super(hacked_network,self).__init__()
        
        num_output=3
        x2_size=4
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size,self.hidden_size,self.num_layers,batch_first=True)
              
        self.fc = nn.Linear(self.hidden_size,num_output)
              
        self.fc1 = nn.Linear(x2_size,6)
        self.fc2 = nn.Linear(9,num_output)
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.optimizer = T.optim.Adam(self.parameters(), lr = lr, weight_decay = wd)
        self.loss = nn.MSELoss()
        
        with T.no_grad():
            # Hack fc1 layer
            for i in range(4):
                for j in range(6):
                    self.fc1.weight[j,i] = 0
   
            for b in [0,2,4]:
                self.fc1.bias[b] = -1
                
            for b in [1,3,5]:
                self.fc1.bias[b] = 0
            
            for j in [0,2,4]:
                self.fc1.weight[j,3] = 1
                
            for j in [1,3,5]:
                self.fc1.weight[j,3] = -10
                
            self.fc1.weight[0,0] =1
            self.fc1.weight[1,0] =1
            self.fc1.weight[2,1] =1
            self.fc1.weight[3,1] =1
            self.fc1.weight[4,2] =1
            self.fc1.weight[5,2] =1
   
                
            # Hack fc2 layer
            for i in range(3):
                for j in range(9):
                    self.fc2.weight[i,j] = 0
                
                self.fc2.weight[i,i] = 1
                
                if i ==0:
                    self.fc2.weight[i,3] = -1000
                    self.fc2.weight[i,4] = -1000
                    self.fc2.weight[i,6] = -1000
                    
                elif i ==1:
                    pass
                    # turns out we can always just call constant
                elif i == 2:
                    self.fc2.weight[i,6] = -1000
                    self.fc2.weight[i,7] = -1000
                    self.fc2.weight[i,8] = -1000
                    
            for i in range(3):
                self.fc2.bias[i] = 0
            
        # stop from updating
        for p in self.fc1.parameters():
            p.requires_grad = False

        for p in self.fc2.parameters():
            p.requires_grad = False

        self.to(self.device)
        
    def forward(self,x1,x2):
    
        self.rnn.flatten_parameters() # it throws up a mistake if I do not use this 
        # solution found at https://discuss.pytorch.org/t/rnn-module-weights-are-not-part-of-single-contiguous-chunk-of-memory/6011/13
        h0 = T.zeros(self.num_layers,x1.size(0),self.hidden_size).to(self.device)
        
        x1,_ = self.rnn(x1,h0)
        x1 = x1[:,-1,:]
   
        x1 = self.fc(x1)
        
        x2 = T.relu(self.fc1(x2))
        
        x = T.cat([x1,x2],dim=-1)  
        out = self.fc2(x)
        
        return out
        
if __name__ == '__main__':
    net = hacked_network()
    x = T.tensor([[0.32,0.1,-0.32,0,0,0,0,0,1]], dtype=T.float32).to(T.device('cuda'))
   
    a = net.fc2(x)
    print(a)
    