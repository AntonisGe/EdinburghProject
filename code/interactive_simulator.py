from settings_SSA import *
from enviroment import *
from dqn_agent import *


def reward_running_count_finder(env,action,running_count,list_of_buy_days):
    '''
    output: reward,running_count,list_of_buy_days
    '''

    if action == 0:
        env.step([0]) # you sell all shares and have 0 shares now!
    
        if running_count ==0:
            return -1,0,[]
            
            
        else:
            treward = 0
            for buy_day in list_of_buy_days:
                price_bought = env.values[0,buy_day]
                t_cost = env.record.tcosts[env.timestep,0]/running_count + env.record.tcosts[buy_day+1,0]
                price_sold = env.values[0,env.timestep-1]
                
                reward = np.sign(price_sold - price_bought - t_cost)
                
                treward += reward
                
            return treward,0,[]
                   
    elif action == 1:
        env.step([running_count]) # you take a step with just holding the number of shares of running count
        return 0,running_count,list_of_buy_days

    elif action == 2:
        running_count += 1
        list_of_buy_days.append(env.timestep)
        
        env.step([running_count])
        
        return 0,running_count,list_of_buy_days
          

env = Enviroment(stocks = ['AAPL'], variance = variance, lookback = lookback,p_finder = shares)
long_agent = DQN_Agent(path_to_save,gamma,max_mem_size,C,batch_size,epsilon, eps_min, eps_dec, num_layers,hidden_size,added_size,fc1_dims,fc2_dims,lr,wd,lookback,x2_size)

start = 25
end = 30
env.change_and_reset(start = start, end = end)

stock = input('What stock to play (random/AAPL/BA/AXP/V/...)?')
if stock == 'random':
    stock = np.random.choice(stocks)
env.change_and_reset(stocks = [stock])

running_count = 0
list_of_buy_days = []

treward = 0
for day in range(env.start,env.end):

    done = day == env.end-1

    x1 = env.SSA_matrix(stock)
    x2 = [running_count]

    print('Price today is {}'.format(env.values[0,env.timestep]))
   
    #action = long_agent.choose_action(x1,x2)
    action = int(input('What action should I take today (0/1/2): '))
    reward,running_count,list_of_buy_days = reward_running_count_finder(env,action,running_count,list_of_buy_days)
    if done and running_count !=0:
        done_reward,running_count,list_of_buy_days = reward_running_count_finder(env,0,running_count,list_of_buy_days)
        reward = reward + done_reward
        
    treward += reward
    print(f'You took reward {reward}')
    print('===========================================')
    x1_next = env.SSA_matrix(stock,env.timestep+1)
    x2_next = [running_count]
    
    long_agent.store_transition(x1,x2,action,reward,x1_next,x2_next,done)

    
print(f'Total reward is {treward}')