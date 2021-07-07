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
CUDA_LAUNCH_BLOCKING=1
env = Enviroment(variance = variance, lookback = lookback,p_finder = weight, transaction_cost = transaction_cost, caa_lookback = caa_lookback)
env_test = Enviroment(variance = 0, lookback = lookback,p_finder = weight, caa_lookback = caa_lookback)

agent = DQN(directory = 'test',num_output = 20, input_size = caa_lookback*2+2,gamma = gamma,C = C, batch_size = batch_size, epsilon = epsilon,eps_min = eps_min,eps_dec = eps_dec, lr = lr, wd = wd)

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
        for day in range(env.start, env.end-1):
            
            is_Friday = day_names_b == 1
            done = day == env.end-2
        
            state = np.concatenate((env.CAA_matrix(),[c1,c3]))
            
            action = agent.choose_action(state)
            
            c1 = action * 0.05
            c3 = 1 - c1
            sigma = sharpe(env.values,env.timestep,30)
            ssignals = list(spolicy[env.timestep-50,:])
            lsignals = list(lpolicy[env.timestep-50,:])
            
            signals = []
            for s,l,p, n in zip(ssignals,lsignals,port,range(len(stocks))):
                if p == 0:
                    if s ==2 and l == 2:
                        signals.append(0)
                    elif s == 2 and l != 2:
                        signals.append(-1)
                    elif s != 2 and l == 2:
                        signals.append(1)
                    else:
                        signals.append(0)
                elif p > 0:
                    if s == 2 or l == -1:
                        signals.append(0)
                        port[n] = 0.0
                    else:
                        signals.append(0)
                        
                elif p < 0:
                    if s == -1 or l == 2:
                        signals.append(0)
                        port[n] = 0.0
                    else:
                        signals.append(0)                   
                else:
                    print(p)
            
            port = [float(i) for i in port]
            
            if sum([abs(i) for i in signals]) != 0 or done:
                for _ in range(1):
                    try:
                        port = quadratic(sigma,port,signals,c1,-c3)
                        port = [0 if abs(i)>1 else i for i in port]
                        break
                    except:
                        print('Problemo')
                        
                port = env.step(port)
                
                reward = np.sign(env.record.valuation[env.timestep] - env.record.valuation[env.timestep - 1])
                
                if done:
                    _ = [0 for _ in range(len(stocks))]
                    done_reward = np.sign(env.record.valuation[env.timestep] - env.record.valuation[env.timestep - 1])
                    reward += done_reward
                
                next_state = np.concatenate((env.CAA_matrix(),[c1,c3]))
                
                agent.store_transition(state,action,reward,next_state,done)
                agent.learn()
                
            else:
                env.step(port)
            
        print(env.valuation)
        env.reset()
        
    #test
    c1,c3 = (0,0)
    port = [0.0 for _ in range(len(stocks))]
    for day in range(env_test.start,env_test.end-1):
        state = np.concatenate((env_test.CAA_matrix(),[c1,c3]))
        action = agent.choose_action(state,explore = False)
        
        c1 = action * 0.05
        c3 = 1 - c1
        sigma = sharpe(env_test.values,env_test.timestep,30)
        ssignals = list(valispolicy[env_test.timestep - warm_up,:])
        lsignals = list(valilpolicy[env_test.timestep - warm_up,:])
        
        signals = []
        for s,l,p, n in zip(ssignals,lsignals,port,range(len(stocks))):
                if p == 0:
                    if s ==2 and l == 2:
                        signals.append(0)
                    elif s == 2 and l != 2:
                        signals.append(-1)
                    elif s != 2 and l == 2:
                        signals.append(1)
                    else:
                        signals.append(0)
                elif p > 0:
                    if s == 2 or l == -1:
                        signals.append(0)
                        port[n] = 0.0
                    else:
                        signals.append(0)
                        
                elif p < 0:
                    if s == -1 or l == 2:
                        signals.append(0)
                        port[n] = 0.0
                    else:
                        signals.append(0)                   
                else:
                    print(p)
        
        port = [float(i) for i in port]
        
        if sum([abs(i) for i in signals]) != 0:
            for _ in range(1):
                try:
                    port = quadratic(sigma,port,signals,c1,-c3)
                    port = [0 if abs(i)>1 else i for i in port]
                    break
                except:
                    pass

        port = env_test.step(port)

        if day == env_test.end - 2:
            env_test.step([0 for _ in range(len(stocks))])
    result = env_test.valuation
    print(f'At training cycle {circle} the result is {round(result,3)}')
            
    # end test
        
    start = end if warm_up == end else start + switch
    end += switch
    circle += 1
    
    
        