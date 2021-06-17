from Enviroment.utils import *

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
        
        
def test(env,agent,reward_function):
    
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

            x1 = env.SSA_matrix(stock)
            x2 = [running_count] + env.paper_1_data(stock)
           
            action = agent.choose_action(x1,x2,explore = False)
            actions.append(action)
            reward,running_count,list_of_buy_days = reward_function(env,action,running_count,list_of_buy_days)
            if done and running_count !=0:
                done_reward,running_count,list_of_buy_days = reward_running_count_finder(env,0,running_count,list_of_buy_days)
                reward = reward + done_reward
                
            treward += reward
            if action==2 or (action==-1 and old_running_count==0):
                total_running_count += 1
    if len(np.unique(actions)) == 1:
        print(f'Agent took only {actions[0]} action')
    
    if total_running_count > 0:
        percentage = treward/2 + total_running_count/2
        percentage = percentage/total_running_count
    else:
        percentage = 0.5
        print('No trades for this round!!')
    
    return percentage