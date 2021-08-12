from settings_SSA import *
from dqn_agent_adapted import *
from enviroment import *
from normalizer import *
import torch as T
from p_finders import shares

directory = 'D:\Project\code\Weights\SSA\long_10_back_to_bigger_switch'

gamma=0.95
max_mem_size = 40000
C=100
batch_size=30
epsilon=0.20
eps_min = 0.005
eps_dec = 5e-5
num_layers=1
hidden_size=20
added_size=15
fc1_dims=200
fc2_dims=100
lr=0.0001#0.000007
wd=0
x2_size = 6


agent =  DQN_Agent_Adapted(directory,gamma,max_mem_size,C,batch_size,epsilon, eps_min, eps_dec, num_layers,hidden_size,added_size,fc1_dims,fc2_dims,lr,wd,lookback,x2_size,scheduler_max_value,scheduler_initial)
agent.load_weights(name = '25_400')

env = Enviroment(start = 400, end = 420, lookback = lookback, p_finder = shares, cash = 1e6)

normalize = normalizer_function(Enviroment(start = 25, end = 400, lookback = lookback, p_finder = shares,variance = 0))

stock = 'AAPL'


actions = []
running_count = [0 for _ in range(len(stocks))]
for day in range(env.start,env.end):
    action_days = []
    for number,stock in enumerate(stocks):
        
        rc = running_count[number]
        
        x1 = normalize(env.SSA_matrix(stock))
        x2 = [rc] + env.paper_1_data(stock)

        agent.Q_eval.eval()
        with T.no_grad():
            action = agent.choose_action(x1,x2,explore = False)
        agent.Q_eval.train()
        
        action_days.append(action)
        
        if action == 2:
            running_count[number] = rc +1
        elif action == 1:
            running_count[number] = rc
        elif action == 0:
            running_count[number] = 0
        else:
            print(f'Did not find action {action}')
            
    env.step(running_count)
    actions.append(action_days)
env.step([0 for _ in range(len(stocks))])