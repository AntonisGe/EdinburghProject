# this file is here as the hyperparamters of a short Stock Selection Agent.

from Enviroment.utils import *
import torch
import os,sys

np.random.seed(6248)#561237)
torch.manual_seed(23518)#7893123)

folder_name = 'short_final'

lookback = 10
variance = 0.03
transaction_cost = 0.2
mc = 0.4

gamma=0.74
max_mem_size = 40000
C=100
batch_size=30
epsilon=0.50
eps_min = 0.01
eps_dec = 1e-5
num_layers=2
hidden_size=200
added_size=300
fc1_dims=200
fc2_dims=100
lr=0.001#0.000007
wd=0.0003
x2_size = 9

warm_up = 400
switch = 50
last = 1700

itterations = 20
warm_up_itterations = 200#200

scheduler_max_value = itterations*switch*4
scheduler_initial = warm_up_itterations*warm_up

path_to_save = dir_path.replace('Enviroment','Weights\\SSA\\{}'.format(folder_name))
if not os.path.isdir(path_to_save):
    os.mkdir(path_to_save)
    
sys.path.append(dir_path)
sys.path.append(dir_path.replace('Enviroment','Agents'))