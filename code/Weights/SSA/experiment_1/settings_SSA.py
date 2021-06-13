from Enviroment.utils import *
import torch
import os,sys

np.random.seed(5642134)
torch.manual_seed(13453)

folder_name = 'delete_this_file' 

lookback = 20
variance = 0.5

gamma=0.95
max_mem_size = 10000
C=30
batch_size=30
epsilon=0.1
eps_min = 0.007
eps_dec = 5e-5
num_layers=2
hidden_size=120
added_size=15
fc1_dims=100
fc2_dims=40
lr=0.0001
wd=0
x2_size = 1

warm_up = 1000
switch = 50
last = 1700

itterations = 40
warm_up_itterations = 500

path_to_save = dir_path.replace('Enviroment','Weights\\SSA\\{}'.format(folder_name))
if not os.path.isdir(path_to_save):
    os.mkdir(path_to_save)
    
sys.path.append(dir_path)
sys.path.append(dir_path.replace('Enviroment','Agents'))