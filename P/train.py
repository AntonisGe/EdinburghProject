from enviroment import *
from cvx import *
from agents import *
from caa_agent import *
from agents import *
from utils import *
from risks import *

hidden = 40
num_layers = 2
learning_rate = 0.1
variance = 2
gamma = 1
lookback = 20
train_on = 90
test_on = 30

stocks = ['AAPL','BA','AXP']

env = Enviroment(start = lookback+1,end=lookback+train_on,lookback=20,stocks = stocks)
SSA = SSA_Agent(hidden,num_layers,learning_rate = learning_rate,gamma = gamma)
CAA = CAA_Agent(hidden,num_layers,learning_rate = learning_rate,gamma = gamma, variance = variance,n_stocks = len(stocks))

test = []
special_stock = 'AAPL'
for attempt in range(20000):
    rewards = []
    log_probs = []
    entropies = []
    
    rewardsSSA = []
    observations = []
    actions = []
    
    for _ in range(train_on-1):
        port = [0.0 for _ in range(len(stocks))]
        
        
        signals = []
        for stock in stocks:
            obs = env.SSA_matrix(stock)
            
            signal = SSA.choose_action(obs,explore = True) -1
            signals.append(signal)
            
        cov = sharpe(env.values,env.timestep,lookback)
        
        z1 = 1 if 1 in signals else 0
        z3 = 1 if -1 in signals else 0
        
        obs = env.CAA_matrix()
        output_CAA,log_prob,entropy = CAA.choose_action(obs,z1,z3,port,explore = True)
        
        # output_CAA = output_CAA.cpu().numpy()
        c1 = float(output_CAA[0])
        c2 = float(output_CAA[0])
        c3 = -float(output_CAA[2])
        
        long_signals,short_signals,constant_signals = cvx_helper(port,signals,len(port))
        
        long_signals_exist = 1 in long_signals
        short_signals_exist = 1 in short_signals
        
        if not long_signals_exist:
            c1 = 0.0
        if not short_signals_exist:
            c3 = 0.0
        
        if long_signals_exist or short_signals_exist:
            port = quadratic(cov,port,signals,c1,c3)
                
        log_probs.append(log_prob)
        entropies.append(entropy)
        
        if z1 == 1 and c2 <0.1:
            rewards.append(1)
        elif z1 == 0 and c2 > 0.9:
            rewards.append(1)
        else:
            rewards.append(-1)
        
        action = signals[1]
        
        if action == -1:
            reward = 1
        else:
            reward = -1
            
        observations.append(env.SSA_matrix(stocks[1]))
        rewardsSSA.append(reward)
        actions.append(action)
        
        env.step(port)
    
    print(sum(rewards))
    test.append(len(rewards))   
    env.reset()
    CAA.update(rewards,log_probs,entropies)
    SSA.update(rewardsSSA,observations,actions)
    SSA.hyper()
    
    rewardsSSA = []
    actions = []
    observations = []
    rewards = []
    log_probs = []
    entropies = []