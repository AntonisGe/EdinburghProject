# finds long validation policy if not already found
# Writes on the \Policy directory

from long import *
from dqn_agent_adapted import *
from enviroment import *
from normalizer import *
from numpy import save,load
import os
 
def find_policy_long(file_path):

    c = cluster_class(n_features = 8, n_clusters = 4)

    agent_name = 'long'
     
    path_to_save1 = dir_path.replace('Enviroment','Policies') + f'\\{file_path}'
    path_to_save2 = path_to_save1 + f'\\{agent_name}.npy'

    continue_ = True

    if not os.path.isdir(path_to_save1):
        os.mkdir(path_to_save1)
        
    if not os.path.isfile(path_to_save2):

        print(f'Starting to find policy for {agent_name} agent.')

        start = warm_up
        end = warm_up + switch

        agent = DQN_Agent_Adapted(path_to_save,gamma,max_mem_size,C,batch_size,epsilon, eps_min, eps_dec, num_layers,hidden_size,added_size,fc1_dims,fc2_dims,lr,wd,lookback,x2_size,scheduler_max_value,scheduler_initial,mc)

        policy = np.zeros((last-warm_up+switch,len(stocks)))

        while end <= last+switch:
            
            file_start = 50 if start == warm_up else start - 200
            file_end = start
            
            c.train(file_start,file_end)
            
            env = Enviroment(start = start, end = end, variance = 0, lookback = lookback,cluster = c,p_finder = shares)
            agent.load_weights(name = f'{file_start}_{file_end}')
            
            running_counts = [0 for _ in range(len(stocks))]
            for day in range(start,end):
                day_policy = [0 for _ in range(len(stocks))]
                
                for number,stock in enumerate(stocks):
                    running_count = running_counts[number]
                    x1 = env.feature_matrix(stock)
                    x2 = [running_count] + env.GOAT_features(stock)
                
                    agent.Q_eval.eval()
                    with T.no_grad():
                        action = agent.choose_action(x1,x2)
                    agent.Q_eval.train()
                    
                    if action == 2: running_counts[number] = running_counts[number] + 1
                    if action == 0: running_counts[number] = 0
                    
                    day_policy[number] = action
                    
                policy[day - warm_up,:] = day_policy
                env.step([i for i in running_counts])
                
                print(day)
            
            start += switch
            end += switch
            
        save(path_to_save2,policy)
        
    return load(path_to_save2)
    
if __name__ == '__main__':
    path = 'final'
    te = find_policy_long(path)