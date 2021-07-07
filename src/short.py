from Enviroment.utils import *
import torch
import os,sys

np.random.seed(561237)
torch.manual_seed(7893123)

folder_name = 'short_baseline'

lookback = 10
variance = 0.1
transaction_cost = 0.1
mc = 0.8

gamma=0.83
max_mem_size = 40000
C=100
batch_size=30
epsilon=0.20
eps_min = 0.005
eps_dec = 5e-5
num_layers=1
hidden_size=20
added_size=15
fc1_dims=200
fc2_dims=100
lr=0.0001#0.000007
wd=0
x2_size = 6

warm_up = 400
switch = 50
last = 1700

itterations = 100
warm_up_itterations = 100

scheduler_max_value = itterations*switch*4
scheduler_initial = warm_up_itterations*warm_up

path_to_save = dir_path.replace('Enviroment','Weights\\SSA\\{}'.format(folder_name))
if not os.path.isdir(path_to_save):
    os.mkdir(path_to_save)
    
sys.path.append(dir_path)
sys.path.append(dir_path.replace('Enviroment','Agents'))