# I tested simpler alogrithms out of curiousity. Did not mention it in the report.

from utils import *
from enviroment import *
import numpy as np

stop_loss = 0.05

folder_path = 'test'
path_l = dir_path.replace('Enviroment','Policies') + f'\\{folder_path}\\long.npy'
path_s = dir_path.replace('Enviroment','Policies') + f'\\{folder_path}\\short.npy'

env = Enviroment(start= 400, end = 1750, variance = 0)

l_policy = np.load(path_l)
s_policy = np.load(path_s)


bought_at = [0 for _ in range(len(env.stocks))]
port = [0 for _ in range(len(env.stocks))]

for day in range(env.start-env.start,env.end-env.start):
    long = l_policy[day,:]
    short = s_policy[day,:]
    
    continue_ =  sum([abs(i) for i in port]) < 0.99
    entry = [0 for _ in range(len(env.stocks))]
    # which stocks we exit
    for stock_no,l,s,b in zip(range(29),long,short,bought_at):
        
        close = env.values[stock_no,day+env.start]
        
        if b > 0:
            if l == 0 or s == 2:
                port[stock_no] = 0.0
                continue_ = True
                
            if (close - b)/b < - stop_loss:
                port[stock_no] = 0.0
                continue_ = True
                
        elif b < 0:
            if l == 2 or s == 0:
                port[stock_no] = 0.0
                continue_ = True
            
            if (b-close)/b < - stop_loss:
                port[stock_no] = 0.0
                continue_ = True
                
        elif b == 0:
            if l == 2 and s != 2:
                entry[stock_no] = 1
                
            if s == 2 and l != 2:
                entry[stock_no] = -1
            
    if continue_:
        total_port = 1 - sum([abs(i) for i in port])
        total_stocks = sum([i != 0 for i in entry])
        
        for stock_no,i in enumerate(entry):
            if i != 0:
                port[stock_no] = i*total_port/total_stocks
                
    port = env.step(port)

print(f'Total valuation is {round(env.valuation,2)}')    