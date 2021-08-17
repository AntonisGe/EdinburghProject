# this file is here to know what settings the CA agent version 3 should be run on

from Enviroment.utils import *
import torch
import os,sys
from numpy import load

np.random.seed(561237)
torch.manual_seed(7893123)

policy_folder = 'final'

from find_policy_short import *
valispolicy = find_policy_short(policy_folder)

from find_policy_long import *
valilpolicy = find_policy_long(policy_folder)

folder_name = 'experiment_05'
short_name = 'short_final'
long_name = 'long_final'

variance = 0.05
transaction_cost = 0.1

caa_lookback = 15

gamma = 0.99
lr = 1e-6
# epsilon = 0.3
# eps_dec = 5e-5
# eps_min = 0.01
max_mem_size = 10000
wd = 0.03
batch_size = 30
C = 20
lookback = 15


warm_up = 400
switch = 50
last = 1700

itterations = 200#300
warm_up_itterations = 750#300


spolicy = load(dir_path.replace('Enviroment','Weights')+ f'\\SSA\\{short_name}\\policy.npy')
lpolicy = load(dir_path.replace('Enviroment','Weights')+ f'\\SSA\\{long_name}\\policy.npy')

path_to_save = dir_path.replace('Enviroment','Weights\\CAA\\{}'.format(folder_name))
if not os.path.isdir(path_to_save):
    os.mkdir(path_to_save)
    
sys.path.append(dir_path)
sys.path.append(dir_path.replace('Enviroment','Agents'))