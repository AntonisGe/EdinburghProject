from utils import *
from smart_p_convertor import *
from reward_finder_SSA import *
from risks import *
from cvx import *
from settings import *

def train(env,SSA,CAA,itteration_no):

    assert day_names_signals[env.start+1] == 1

    for itteration in range(itteration_no):

        port = [0.0 for _ in range(len(stocks))]
        old_position = [0 for _ in range(len(stocks))]
        c1,c3 = (0,0)
        for _ in range(env.start,env.end):
            
            
            is_Friday = day_names_signals[env.timestep+1] == 1
            
            # find new percentages of today
            port = smart_p_covert(env.portofolio,list(env.values[:,env.timestep]),env.valuation)
            
            #SSA
            signals = []
            log_probs = []
            states = []
            for stock_no,stock in enumerate(stocks):
            
                current_position = old_position[stock_no]
            
                obs = env.SSA_matrix(stock)
                states.append(obs)
                
                signal,add = SSA.choose_action(obs,current_position,is_Friday,explore = True)
                signal = signal - 1
                signals.append(signal)
                log_probs.append(add)
                
            SSA.store_memory(log_prob = log_probs,action=signals,state=states)
     
            port,signals,old_position = signal_and_port_convertor(port,signals,is_Friday)
     
            if is_Friday: # We do this step only on Fridays
            
                # find the covariance matrix
                cov = sharpe(env.values,env.timestep,lookback)
                
                #CAA
                z1 = 1 if 1 in signals else 0
                z3 = 1 if -1 in signals else 0     
                obs = env.CAA_matrix()
                output_CAA,log_prob,entropy = CAA.choose_action(obs,z1,z3,port,explore = True)
                CAA.store_memory(log_prob=log_prob,entropy=entropy)
                c1 = float(output_CAA[0])
                c3 = - float(output_CAA[2]) 

                #make a decision
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
            
        # CAA update
        CAA_rewards = []
        
        last_one = env.start
        for day in range(env.start+1,env.end):
        
            if day_names_signals[day] == 1:
                reward = sum(env.CAA_reward(last_one,day))
                CAA_rewards.append(reward)
                
                last_one = day
        
        CAA.update(CAA_rewards,CAA.entropy_memory,CAA.log_prob_memory)
        CAA.schedule_hyper()
        CAA.clean_memory()
        
        # SSA update
        signals = np.array(SSA.action_memory)
        #log_probs = np.array(SSA.log_prob_memory)
        states = np.array(SSA.state_memory)
        for stock_no,stock in enumerate(stocks):
            stock_signals = list(signals[:,stock_no])
            log_prob = [SSA.log_prob_memory[i][stock_no] for i in range(len(SSA.log_prob_memory))]
            state = list(states[:,stock_no])

            rewards = reward_finder_ssa(env,stock_signals,log_prob,stock)
            
            SSA.update(rewards,state,stock_signals,log_prob)
            if stock_no == 0:
                print(sum(rewards)/len(rewards))
            
        SSA.schedule_hyper()
        SSA.clean_memory()
                
        #reset_enviroment
        #print(env.valuation)
        env.reset()