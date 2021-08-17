# this file is here to calculate the reward of the Stock Selection Agents.
# Note that the reward is calculated from this file and NOT the enviroment

from Enviroment.utils import *
import torch as T

def long(env,action,running_count,list_of_buy_days,train = True):
    '''
    output: reward,running_count,list_of_buy_days
    '''

    if action == 0:
        env.step([0]) # you sell all shares and have 0 shares now!
    
        if running_count ==0:
            if train:
                return -1,0,[]
            else:
                return 0,0,[]
            
            
        else:
            treward = 0
            for buy_day in list_of_buy_days:
                price_bought = env.values[0,buy_day]
                t_cost = env.record.tcosts[env.timestep,0]/running_count + env.record.tcosts[buy_day+1,0]
                price_sold = env.values[0,env.timestep-1]
                
                reward = (price_sold - price_bought - t_cost)/price_bought
                
                if train:
                    reward = np.sign(reward)
                    reward = min(reward,0.9)
                
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
        
def short(env,action,running_count,list_of_buy_days,train = True):
    '''
    output: reward,running_count,list_of_buy_days
    '''

    if action == 0:
        env.step([0]) # you sell all shares and have 0 shares now!
    
        if running_count ==0:
            if train:
                return -1,0,[]
            else:
                return 0,0,[]
            
            
        else:
            treward = 0
            for buy_day in list_of_buy_days:
                price_sold = env.values[0,buy_day]
                t_cost = env.record.tcosts[env.timestep,0]/abs(running_count) + env.record.tcosts[buy_day+1,0]
                price_bought = env.values[0,env.timestep-1]
                
                reward = (price_sold - price_bought - t_cost)/price_sold
                
                if train:
                    reward = np.sign(reward)
                
                treward += reward

            return treward,0,[]
                
            
    
    elif action == 1:
        env.step([running_count]) # you take a step with just holding the number of shares of running count
        return 0,running_count,list_of_buy_days

    elif action == 2:
        running_count -= 1
        list_of_buy_days.append(env.timestep)
        
        env.step([running_count])
        
        return 0,running_count,list_of_buy_days
     
def test(env,agent,reward_function,normalize):
    
    agent.Q_eval.eval()
    
    with T.no_grad():
    
        total_running_count = 0
        treward = 0
        actions = []
        for stock in stocks:
            env.change_and_reset(stocks = [stock])
            
            running_count = 0
            list_of_buy_days = []
            
            for day in range(env.start,env.end):
                
                old_running_count = running_count # just for finding how many actions it has taken!
            
                done = day == env.end - 1

                x1 = env.feature_matrix(stock)
                x2 = [running_count] + env.GOAT_features(stock)
               
                action = agent.choose_action(x1,x2,explore = False)
                actions.append(action)
                reward,running_count,list_of_buy_days = reward_function(env,action,running_count,list_of_buy_days,train = False)
                if done and running_count !=0:
                    done_reward,running_count,list_of_buy_days = reward_function(env,0,running_count,list_of_buy_days,train = False)
                    reward = reward + done_reward
                    
                treward += reward
                if action==2:#or (action==-1 and old_running_count==0):
                    total_running_count += 1
        if len(np.unique(actions)) == 1:
            print(f'Agent took only {actions[0]} action')
        
        if total_running_count > 0:
            #print(treward,total_running_count)
            # percentage = treward/2 + total_running_count/2
            # percentage = percentage/total_running_count
            percentage = treward/total_running_count
        else:
            percentage = 0
            print('No trades for this round!!')
            
    agent.Q_eval.train()
    
    return percentage