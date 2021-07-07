import torch as T
from settings_SSA import *
from enviroment import *
from dqn_agent_adapted import *
from helper import *
from time import time
from normalizer import *
from boring_agent import *
from numpy import save
from cluster import *

c = cluster_class(n_features = 8, n_clusters = 7)
c.train(50,400)
env = Enviroment(stocks = ['AAPL'], variance = variance, lookback = lookback,p_finder = shares, transaction_cost = transaction_cost,cluster = c)
env_test = Enviroment(stocks = ['AAPL'], variance = 0, lookback = lookback,p_finder = shares,cluster = c)
long_agent = DQN_Agent_Adapted(path_to_save,gamma,max_mem_size,C,batch_size,epsilon, eps_min, eps_dec, num_layers,hidden_size,added_size,fc1_dims,fc2_dims,lr,wd,lookback,x2_size,scheduler_max_value,scheduler_initial,mc)
#long_agent = addicted_long_agent(path_to_save,gamma,max_mem_size,C,batch_size,epsilon, eps_min, eps_dec, num_layers,hidden_size,added_size,fc1_dims,fc2_dims,lr,wd,lookback,x2_size,scheduler_max_value,scheduler_initial)
#long_agent = random_agent(path_to_save,gamma,max_mem_size,C,batch_size,epsilon, eps_min, eps_dec, num_layers,hidden_size,added_size,fc1_dims,fc2_dims,lr,wd,lookback,x2_size,scheduler_max_value,scheduler_initial)
reward_running_count_finder = long
#long_agent.load_weights(name = '25_400')


starting_time = time()

start = 50
end = warm_up
circle = 0
valid_rewards = []

policy = np.zeros((last-start,len(stocks)))

progression = []
while end <= last:
    best_score = -100
    
    env.change_and_reset(start = start, end = end) #if start ==50 else start - 3*switch, end = end) #3
    env_test.change_and_reset(start = end, end = end + switch)
    
    #normalize = normalizer_function(Enviroment(stocks = stocks, start = start, end = end, variance = 0, lookback = lookback))
    
    itt = warm_up_itterations if end == warm_up else itterations
    
    for itter in range(itt):
        stock = np.random.choice(stocks)
        env.change_and_reset(stocks = [stock])

        running_count = 0
        list_of_buy_days = []

        treward = 0
        for day in range(env.start,env.end):

            done = day == env.end-1

            x1 = env.feature_matrix(stock)
            x2 = [running_count] + env.GOAT_features(stock)
           
            long_agent.Q_eval.eval()
            with T.no_grad():
                action = long_agent.choose_action(x1,x2)
            long_agent.Q_eval.train()
            
            reward,running_count,list_of_buy_days = reward_running_count_finder(env,action,running_count,list_of_buy_days)
            if done and running_count !=0:
                done_reward,running_count,list_of_buy_days = reward_running_count_finder(env,0,running_count,list_of_buy_days)
                reward = reward + done_reward
                
            treward += reward
            x1_next = env.feature_matrix(stock)
            x2_next = [running_count] + env.GOAT_features(stock)
            
            long_agent.store_transition(x1,x2,action,reward,x1_next,x2_next,done,circle)
            long_agent.learn()
            
        env.reset()
        progression.append(treward)
        if itter % 10 == 0:
            score = sum(progression[-40:])/len(progression[-40:])
            if score> best_score:
                long_agent.save_weights(name = 'best')
                best_score = score
                #print(f'New best score is {round(score,4)}. Saved the weights.')
            if end == warm_up:
                print(score)

    long_agent.load_weights(name = 'best')
    # find train_policy
    env_special = Enviroment(start = start, end  = end,lookback = lookback)
    stock_policy = [0 for _ in range(len(stocks))]
    
    for day in range(env_special.start,env_special.end):
    
        day_policy = [-1 for _ in range(len(stocks))]
        
        running_counts = [0 for _ in range(len(stocks))]
        for n,stock in enumerate(stocks):
        
            x1 = env_special.feature_matrix(stock)
            x2 = [running_count] + env_special.GOAT_features(stock)
           
            long_agent.Q_eval.eval()
            with T.no_grad():
                action = long_agent.choose_action(x1,x2,explore = False)
            long_agent.Q_eval.train()

            if action == 2: running_counts[n] = running_counts[n] + 1
            if action == 0: running_counts[n] = 0
            
            day_policy[n] = action

    policy[day - 50,:] = day_policy
    # stop finding training policy 
            
            
        
    
    long_agent.epsilon = epsilon/3
    aver = sum(progression[-20:])/20
    aver = round(aver,2)
    
    test_reward = test(env_test,long_agent,reward_running_count_finder,'123')
    
    valid_rewards.append(test_reward)
    print(f'Circle {circle} train reward is {aver} and validation reward is {round(test_reward,4)}. Current average valid is {round(sum(valid_rewards)/len(valid_rewards),4)}')
    
    long_agent.save_weights(name = f'{start}_{end}')
    long_agent.reset_scheduler()
    #long_agent.reset_memory()
    #long_agent = DQN_Agent_Adapted(path_to_save,gamma,max_mem_size,C,batch_size,epsilon, eps_min, eps_dec, num_layers,hidden_size,added_size,fc1_dims,fc2_dims,lr,wd,lookback,x2_size,scheduler_max_value,scheduler_initial,mc)
        
    start = end if warm_up == end else start + switch
    end += switch
    
    circle += 1
    
long_agent.save_buffer()
end_time = time()

f_ave = round(sum(valid_rewards)/len(valid_rewards),4)
print(f'Final average is {f_ave}')
print(f'Took {int((end_time - starting_time) // 60)} minutes for the whole process.')

df = pd.DataFrame()
df['Validations'] = valid_rewards
df.to_csv(path_to_save + '\\results.csv',index=False)
save(path_to_save+'\\policy.npy',policy)


df = pd.DataFrame()
df['Train'] = progression
df.to_csv(path_to_save + '\\train.csv',index = False)