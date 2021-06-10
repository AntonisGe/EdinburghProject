from utils import *

def reward_finder_ssa(env,signals,adds,stock):
    rewards = []
    # for day in range(env.start,env.end):
    
    for action,add in zip(signals,adds):
        
        add = add.cpu().detach().numpy()[0]
                
        is_Friday = bool(add[3])  
        current_position = list(add[:3])
        current_position = current_position.index(1) - 1
        
        if action == 1:
            reward = 1
        else:
            reward = 0
        
        # if position != 0:
            # if action == position: # in case agent gives the same
                # reward = -1
            # elif action == 0: # agent does not do anything
                # reward = 0
            # else: # agent gets out of the trade
                # rewards_env = env.SSA_reward(last_day,day,stock)
                # reward = np.sign(sum(rewards_env))
                # position = 0
        # else:
            # if is_Friday:
                # reward = 0
                # if action != 0:
                    # last_day = day
                    # position = action
            # else:
                # if action != 0:
                    # reward = -1
                # else:
                    # reward = 0
                
    
        rewards.append(reward)

    return rewards