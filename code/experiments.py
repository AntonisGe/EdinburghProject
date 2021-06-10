from enviroment import *
from agents import *
from caa_agent import *
from agents import *
from utils import *
from train import *
from settings import *

start = 23
last = start+train_on

env = Enviroment(start = start,end=last,lookback=20,stocks = stocks)
SSA = SSA_Agent(hidden,num_layers,learning_rate = learning_rate,gamma = gamma)
CAA = CAA_Agent(hidden,num_layers,learning_rate = learning_rate,gamma = gamma, variance = variance,n_stocks = len(stocks))

itteration_no = 100
train(env,SSA,CAA,itteration_no)
