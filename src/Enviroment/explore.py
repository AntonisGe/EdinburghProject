from utils import *
from enviroment import *
import numpy as np
import matplotlib.pyplot as plt

folder_path = 'test'
path = dir_path.replace('Enviroment','Policies') + f'\\{folder_path}\\long.npy'

env = Enviroment(start= 400, end = 1750, variance = 0)

policys = np.load(path)

closes = env.values[:,400:1750]

def funct(stock):
    
    stock_no = stocks.index(stock)
    
    bought_at = []
    day_at = []
    close = closes[stock_no,:]
    policy = policys[:,stock_no]
    profits = []
    time = []
    
    for p,c,day in zip(policy,close,range(close.shape[0])):
        if p == 2:
            bought_at.append(c)
            day_at.append(day)
        if p == 0:
            for b,d in zip(bought_at,day_at):
                profit = (c - b)/b
                profits.append(profit)
                time.append(day-d)
                
            bought_at = []
            day_at = []
                
                
    return profits,time


p = []
t = []

for stock in stocks:
    a,b = funct(stock)
    for i,j in zip(a,b):
        p.append(i)
        t.append(j)
        
plt.hist(p,bins = 400)
plt.show()
