from networks import *
from utils import *
import torch

class Agents:
    def __init__(self,hidden_size,num_layers,learning_rate=0.01,alpha=0.003,gamma = 1):
        self.alpha = alpha
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        
    def convert_observation(self,obs):
        
        return torch.Tensor([obs]).cuda()
        
    def clean_memory(self):
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

class SSA_Agent(Agents):
    def __init__(self,hidden_size,num_layers,learning_rate=0.01,alpha=0.003,gamma = 1):
        Agents.__init__(self,hidden_size,num_layers,learning_rate,alpha,gamma)

        self.n_actions = 3
        input_size = 6        
        self.policy = RNN_SSA(input_size,hidden_size,num_layers,self.n_actions)
        self.policy = self.policy.cuda()
        self.policy_optim = torch.optim.SGD(self.policy.parameters(), lr=self.learning_rate)

    def choose_action(self,observation,explore: bool):
        observation = self.convert_observation(observation)
        feed_forward = self.policy.forward(observation)
        feed_forward = feed_forward.cpu().detach().numpy()[0]
        if explore:
            feed_forward = feed_forward - feed_forward.max()
            exp = np.exp(feed_forward)
            probs = exp/np.sum(exp)
            action = np.random.choice(len(probs),1,p=probs).item()

        else:
            feed_forward = list(feed_forward)
            action = feed_forward.index(max(feed_forward))

        return action
        
    def update(self,rewards,observations,actions):
    
        p_loss = 0.0
        G= 0
        t = len(rewards)-1
        T = t
        while t != -1:
            action = actions[t]
            observation = self.convert_observation(observations[t])
            G = rewards[t] + self.gamma*G
            log_prob = torch.log(self.policy.forward(observation)[0][action])
            p_loss -= G*log_prob
            t -= 1
        p_loss = p_loss/T
        
        self.policy_optim.zero_grad()
        p_loss.backward()
        self.policy_optim.step()
        
    def hyper(self):
        pass
        
        
if __name__ == '__main__':
    from enviroment import *
    
    env = Enviroment()
    
    #SSA = SSA_Agent(6,40,2)
    #obs = env.SSA_matrix('AAPL')
    #action = SSA.choose_action(obs,explore = True)
    
    CAA = CAA_Agent(40,2,n_stocks = 3)
    obs = env.CAA_matrix()
    action = CAA.choose_action(obs,explore = True)
    
    CAA.update([-1,3],[obs,obs],[action,action])
        
        