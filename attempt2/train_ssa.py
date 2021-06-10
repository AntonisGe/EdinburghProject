from dqn_agent import *
from smart_p_convertor import *
from reward_finder_SSA import *
from risks import *
from cvx import *
from settings import *
from ssa_reward import *

def x2_finder(current_position,is_Friday):
    if current_position>0:
        i = 1
    elif current_position<0:
        i = -1
    else:
        i = 0
        
    x2 = [0 for _ in range(4)]
    x2[i+1] = 1
    x2[3] = int(is_Friday)
    
    return x2

def train_ssa(env,agent,no_itterations):
    
    assert day_names_signals[env.start+1] == 1

    for itteration in range(no_itterations):

        treward = 0

        port = [0.0 for _ in range(len(stocks))]
        old_position = [0 for _ in range(len(stocks))]
        past_entries = [-1 for _ in range(len(stocks))]
        c1,c3 = (0,0)
        for day in range(env.start,env.end):
            
            done = False
            if day == env.end -1: done = True
            
            is_Friday = day_names_signals[env.timestep+1] == 1
            is_Friday_tmr = day_names_signals[env.timestep+2] == 1
            
            # find new percentages of today
            port = smart_p_covert(env.portofolio,list(env.values[:,env.timestep]),env.valuation)
            
            #SSA
            signals = [0 for _ in range(len(stocks))]
            for stock_no,stock in enumerate(stocks):
            
                current_position = old_position[stock_no]
            
                x1 = env.SSA_matrix(stock)
                x2 = x2_finder(current_position,is_Friday)
                
                action = agent.choose_action(x1,x2) - 1
                reward,entered_trade,new_current = reward_SSA(env,stock,old_position[stock_no],action,is_Friday,past_entries[stock_no])
                
                if entered_trade: past_entries[stock_no] == env.timestep
                
                x1_next = env.SSA_matrix(stock,env.timestep+1)
                x2_next = x2_finder(new_current,is_Friday_tmr)
                
                agent.store_transition(x1,x2,action+1,reward,x1_next,x2_next,done)
                agent.learn()

                signals[stock_no] = action
                treward += reward
     
            port,signals,old_position = signal_and_port_convertor(port,signals,is_Friday)
     
            if is_Friday: # We do this step only on Fridays
            
                # find the covariance matrix
                cov = sharpe(env.values,env.timestep,lookback)
                
                c1 = 0.5
                c3 = -0.5
                long_signals,short_signals = cvx_helper(port,signals,len(port))

                long_signals_exist = 1 in long_signals
                short_signals_exist = 1 in short_signals
                
                if not long_signals_exist:
                    c1 = 0.0
                if not short_signals_exist:
                    c3 = 0.0
                
                if long_signals_exist or short_signals_exist:
                    try:
                        for _ in range(500): # ccx sometimes just bugs. I pass the same parameters, but sometimes just bugs and others works fine.
                            try:
                                port = quadratic(cov,port,signals,c1,c3)
                                break
                            except:
                                pass
                    except:
                        return cov,port,signals,c1,c3
                
            #take step
            env.step(port)
                       
        #reset_enviroment
        env.reset()
        print(f'Total reward is {treward}')

if __name__ == '__main__':
    from enviroment import *
    
    start =23
    end = 503
    
    env = Enviroment(start = start, end = end, stocks = stocks,variance = 0)
    SSA = DQN_Agent(lr = 0.0000001,C=30,gamma =0.8,batch_size = 5,eps_min=0.01,epsilon = 0.5,eps_dec =5e-5,wd = 0.1)
    
    train_ssa(env,SSA,200)