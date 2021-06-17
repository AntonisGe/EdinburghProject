from settings_SSA import *
from enviroment import *
from dqn_agent import *
from dqn_agent_adapted import *
from helper import *
from time import time

env = Enviroment(stocks = ['AAPL'], variance = variance, lookback = lookback,p_finder = shares, transaction_cost = transaction_cost)
env_test = Enviroment(stocks = ['AAPL'], variance = 0, lookback = lookback,p_finder = shares)
long_agent = DQN_Agent_Adapted(path_to_save,gamma,max_mem_size,C,batch_size,epsilon, eps_min, eps_dec, num_layers,hidden_size,added_size,fc1_dims,fc2_dims,lr,wd,lookback,x2_size,scheduler_max_value,scheduler_initial)

#long_agent.load_weights(name = '25_400')

starting_time = time()

start = 25
end = warm_up
circle = 0
summy = 0

progression = []
while end <= last:
    env.change_and_reset(start = start, end = end)
    env_test.change_and_reset(start = end, end = end + switch)
    
    itt = warm_up_itterations if end == warm_up else itterations
    
    for itter in range(itt):
        stock = np.random.choice(stocks)
        env.change_and_reset(stocks = [stock])

        running_count = 0
        list_of_buy_days = []

        treward = 0
        for day in range(env.start,env.end):

            done = day == env.end-1

            x1 = env.SSA_matrix(stock)
            x2 = [running_count] + env.paper_1_data(stock)
           
            action = long_agent.choose_action(x1,x2)
            reward,running_count,list_of_buy_days = reward_running_count_finder(env,action,running_count,list_of_buy_days)
            if done and running_count !=0:
                done_reward,running_count,list_of_buy_days = reward_running_count_finder(env,0,running_count,list_of_buy_days)
                reward = reward + done_reward
                
            treward += reward
            x1_next = env.SSA_matrix(stock,env.timestep+1)
            x2_next = [running_count] + env.paper_1_data(stock,env.timestep)
            
            long_agent.store_transition(x1,x2,action,reward,x1_next,x2_next,done,circle)
            long_agent.learn()
            
        env.reset()
        progression.append(treward)
        if itter % 50 == 0: print(sum(progression[-20:])/20)
    
    long_agent.epsilon = epsilon/3
    aver = sum(progression[-20:])/20
    aver = round(aver,2)
    
    test_reward = test(env_test,long_agent,reward_running_count_finder)
    
    print(f'Circle {circle} train reward is {aver} and validation reward is {test_reward}.')
    summy += test_reward
    long_agent.save_weights(name = f'{start}_{end}')
    long_agent.reset_scheduler()
        
    start = end if warm_up == end else start + switch
    end += switch
    
    circle += 1
    
long_agent.save_buffer()
end_time = time()

f_ave = round(summy/circle,2)
print(f'Final average is {f_ave}')
print(f'Took {int((end_time - starting_time) // 60)} minutes for the whole process.')