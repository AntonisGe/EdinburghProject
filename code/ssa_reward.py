from utils import *

def reward_SSA(env,stock,old_position,action,is_Friday,past_day):
    '''
    Returns the reward, if the agent entered_a_trade, and next position
    '''
    
    if old_position == action and old_position != 0:
        return -0.2,False,old_position
        
    elif old_position ==0 and action == 0:
        return 0,False,old_position
        
    elif old_position ==0 and action != 0:
        return 0,True,action
        
    else:
        rewards = env.SSA_reward(past_day,env.timestep,stock)
        
        return np.sign(sum(rewards)),False,0
        
        
    
    
    