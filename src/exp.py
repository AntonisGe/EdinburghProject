from settings_SSA import *
from enviroment import *
from boring_agent import *
from helper import *
from normalizer import *

env = Enviroment(stocks = ['AAPL'],start = 25, end = 50)
agent = random_agent('bla')
norm = normalizer_function(env)

test_reward = test(env,agent,short,norm)

