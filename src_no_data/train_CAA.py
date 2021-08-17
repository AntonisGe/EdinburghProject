# This file trains a CA mechnism version 3.

from settings_CAA import *
import torch as T
from enviroment import *
from helper import *
from time import time
from numpy import save
from ppo import *
from risks import *
from cvx import *
from dqn_agent_adapted import *
import pandas as pd
from decoders import *

def reward_finder(env):
    return env.record.valuation[env.timestep] - env.record.valuation[env.timestep - 1]

CUDA_LAUNCH_BLOCKING=1
env = Enviroment(variance = variance, lookback = lookback,p_finder = weight, transaction_cost = transaction_cost, caa_lookback = caa_lookback)
env_test = Enviroment(variance = 0, lookback = lookback,p_finder = weight, caa_lookback = caa_lookback)

agent = DQN(directory = path_to_save,num_output = 20, input_size = caa_lookback*6 + 2,gamma = gamma,C = C, batch_size = batch_size, epsilon = epsilon,eps_min = eps_min,eps_dec = eps_dec, lr = lr, wd = wd)

starting_time = time()

start = 50
end = warm_up
circle = 0
valid_rewards = []

progression = []

while end <= last:
    env.change_and_reset(start = start if start ==50 else start - 3*switch, end = end) #3
    env_test.change_and_reset(start = end, end = end + switch)
    
    itt = warm_up_itterations if end == warm_up else itterations
    
    for it in range(itt):
        
        c1,c3 = (0,0)
        port = [0.0 for _ in range(len(stocks))]
        
        reward_to_store = 0
        state_to_store = None
        action_to_store = None
        next_state_to_store = None
        done_to_store = None
        learn = False
        c1s = []
        
        for day in range(env.start, env.end-1):
            
            is_Friday = day_names_signals[day] == 1
            done = day == env.end-2
        
            sigma = sharpe(env.values,env.timestep,30)
            ssignals = list(spolicy[env.timestep-50,:])
            lsignals = list(lpolicy[env.timestep-50,:])
            
            signals,port = decoder(ssignals,lsignals,port)

            if is_Friday:
                if sum([abs(i) for i in signals]) != 0:
                
                    if learn and not done:
                        agent.store_transition(state_to_store,action_to_store,reward_to_store,next_state_to_store,done_to_store)
                        agent.learn()
                
                    state = np.concatenate((env.CAA_matrix(),[c1,c3]))      
                    action = agent.choose_action(state)
                
                    c1 = action * 0.05
                    c3 = 1 - c1
                    c1s.append(c1)
                    
                    for _ in range(1):
                        try:
                            port = quadratic(sigma,port,signals,c1,-c3)
                            port = [0 if abs(i)>1 else i for i in port]
                            break
                        except:
                            port = quadratic(sigma,port,signals,c1,-c3)
                            
                    port = env.step(port)
                    
                    reward = reward_finder(env)
                    reward_to_store = reward
                    
                    if not done:
                        next_state = np.concatenate((env.CAA_matrix(),[c1,c3]))
                        action_to_store = action
                        next_state_to_store = next_state
                        state_to_store = state
                        done_to_store = done
                        learn = True
                                   
                                    
                else:
                    port = env.step(port)
                    reward_to_store += reward_finder(env)
                    
            else:
                port = env.step(port)
                reward_to_store += reward_finder(env)
                
            if done:
                env.step([0 for _ in range(29)])
                reward_to_store += reward_finder(env)
                if learn:
                    agent.store_transition(state_to_store,action_to_store,reward_to_store,next_state_to_store,done_to_store)
                    agent.learn()
                
        if it % 50 == 0: 
            print(env.valuation,sum(c1s)/len(c1s))
        env.reset()
    
    agent.save_weights(name = f'{start}_{end}')
    #test
    c1,c3 = (0,0)
    port = [0.0 for _ in range(len(stocks))]
    for day in range(env_test.start,env_test.end-1):
        is_Friday = day_names_signals[day] == 1
        done = day == env.end-2
    
        sigma = sharpe(env.values,env.timestep,30)
        ssignals = list(valispolicy[env_test.timestep - warm_up,:])
        lsignals = list(valilpolicy[env_test.timestep - warm_up,:])
        
        signals,port = decoder(ssignals,lsignals,port)
        
        if is_Friday:
            if sum([abs(i) for i in signals]) != 0:
            
                state = np.concatenate((env_test.CAA_matrix(),[c1,c3]))      
                action = agent.choose_action(state,explore = False)
            
                c1 = action * 0.05
                c3 = 1 - c1
                
                for _ in range(1):
                    try:
                        port = quadratic(sigma,port,signals,c1,-c3)
                        port = [0 if abs(i)>1 else i for i in port]
                        break
                    except:
                        pass
                        
                port = env_test.step(port)

            
            elif done:
                env_test.step([0 for _ in range(len(stocks))])
                
            else:
                port = env_test.step(port)
                
        else:
            port = env_test.step(port)
           
    env_test.step([0 for _ in range(29)])
    result = env_test.valuation
    valid_rewards.append(result/env_test.initial_cash)
    r = 1
    for i in valid_rewards: r = r*i
    print(f'At training cycle {circle} the result is {round(result,3)}. Current cumm is {round(r,5)}.')
            
    # end test
        
    start = end if warm_up == end else start + switch
    end += switch
    circle += 1
    
agent.save_buffer()   
df = pd.DataFrame()
df['Validation'] = valid_rewards

df.to_csv(f'{path_to_save}\\results.csv',index=False)
