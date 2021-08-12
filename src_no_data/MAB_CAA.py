from settings_CAA import *
from utils import *
from risks import *
from cvx import *
from decoders import *
from env import *

def reward_finder(env):
    return env.record.valuation[env.timestep] - env.record.valuation[env.timestep - 1]
        
lr = 0.2
Q_values = np.zeros(21)
rc = 0

env = Enviroment(weights = True)
env_test = Enviroment(start = 407, end = 1749,weights = True)

test_port = [0 for _ in range(29)]

rewards = [0 for _ in range(21)]
actions = []

for start,end in zip(Friday_timesteps[82:-1],Friday_timesteps[83:]):
    start += 1
    end += 1
    end = min(end,1749)
    
    env.change_and_reset(start = start, end = end)
    for action in range(21):
        port = [0 for _ in range(29)]
        env.reset()
        c1 = action * 0.05
        c3 = 1 - c1
        
        for day in range(start,end):
            ssignals = list(valispolicy[day-400,:])
            lsignals = list(valilpolicy[day-400,:])
            
            signals,port = decoder(ssignals,lsignals,port)
                
            is_Friday = day_names_signals[day] == 1
            
            if is_Friday:                  
                sigma = sharpe(env.values,env.timestep,30)
                try:
                    port = quadratic(sigma,port,signals,c1,-c3)
                    port = [i if abs(i) > 1e-3 else 0 for i in port]
                except: pass
                
                port = env.step(port)
                
            else:
                port = env.step(port)

        
    
        env.step([0 for _ in range(29)])
        reward = env.valuation - env.initial_cash
        
        Q = Q_values[action]
        Q_values[action] = Q*(1 - lr) + lr*reward
        
    action = np.argmax(Q_values)
    c1 = action*0.05
    c3 = 1-c1
    actions.append(action)
    
    stop = True
    while stop:
        day = env_test.timestep
        if day < 1750: 
        
            ssignals = list(valispolicy[day-400,:])
            lsignals = list(valilpolicy[day-400,:])
            
            signals,test_port = decoder(ssignals,lsignals,test_port)
            is_Friday = day_names_signals[day] == 1
            
            if is_Friday:                  
                sigma = sharpe(env_test.values,env_test.timestep,30)
                try:
                    test_port = quadratic(sigma,test_port,signals,c1,-c3)
                    test_port = [i if abs(i) > 1e-3 else 0 for i in test_port]
                except: pass
                
                test_port = env_test.step(test_port)
                    
            else:
                test_port = env_test.step(test_port)
           
            if day_names_signals[day+1] == 1:
                stop = False

    print(env_test.valuation,action)
    
    