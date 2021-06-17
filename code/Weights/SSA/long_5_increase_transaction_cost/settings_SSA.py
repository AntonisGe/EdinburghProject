from Enviroment.utils import *
import torch
import os,sys

np.random.seed(5642134)
torch.manual_seed(13453)

folder_name = 'long_5_increase_transaction_cost' 

lookback = 3
variance = 0.1
transaction_cost = 0.5

gamma=0.95
max_mem_size = 40000
C=100
batch_size=30
epsilon=0.20
eps_min = 0.02
eps_dec = 5e-5
num_layers=1
hidden_size=120
added_size=15
fc1_dims=100
fc2_dims=40
lr=0.0001#0.000007
wd=0.001
x2_size = 6

warm_up = 400
switch = 20
last = 1700

itterations = 100
warm_up_itterations = 200

scheduler_max_value = itterations*switch
scheduler_initial = warm_up_itterations*warm_up

# reward 0.56

path_to_save = dir_path.replace('Enviroment','Weights\\SSA\\{}'.format(folder_name))
if not os.path.isdir(path_to_save):
    os.mkdir(path_to_save)
    
sys.path.append(dir_path)
sys.path.append(dir_path.replace('Enviroment','Agents'))