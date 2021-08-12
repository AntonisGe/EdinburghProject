import sys
from utils import *

sys.path.append(dir_path.replace('Enviroment','Agents'))
sys.path.append(dir_path.replace('\\Enviroment',''))

from cluster import *

from dqn_agent_adapted import *
from short import *
short = DQN_Agent_Adapted(path_to_save,gamma,max_mem_size,C,batch_size,epsilon, eps_min, eps_dec, num_layers,hidden_size,added_size,fc1_dims,fc2_dims,lr,wd,lookback,x2_size,scheduler_max_value,scheduler_initial,mc)
short.load_weights(name = '1500_1700')
short.load_buffer()

from long import *
long_agent = DQN_Agent_Adapted(path_to_save,gamma,max_mem_size,C,batch_size,epsilon, eps_min, eps_dec, num_layers,hidden_size,added_size,fc1_dims,fc2_dims,lr,wd,lookback,x2_size,scheduler_max_value,scheduler_initial,mc)
long_agent.load_weights(name = '1500_1700')
long_agent.load_buffer()
print('started_training')
c = cluster_class(n_features = 8, n_clusters = 4)
c.train(50,400)
end = 400
while end <= 1700:
    end += 50
    start = end - 200
    c.train(start,end)
    
# from settings_CAA import *
# CAA = DQN(directory = path_to_save,num_output = 20, input_size = caa_lookback*6 + 2,gamma = gamma,C = C, batch_size = batch_size, epsilon = epsilon,eps_min = eps_min,eps_dec = eps_dec, lr = lr, wd = wd)
# CAA.load_weights(name = '1650_1700')
# CAA.load_buffer()

