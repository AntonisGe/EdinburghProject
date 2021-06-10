from utils import *

'''
Imagine we have 50 stocks of AAPL that at timestep t are worth 0.3 of our capitalize

but at timestep t+1 the portoflio goes down. 50 stocks will have more weight than 0.3
'''

def smart_p_covert(port,value_of_stocks,valuation):
    '''
    port: list of total,stocks,owned
    valuation: total valuation of the port and cash
    
    output: new percentages, supposed to be passed through cvx
    '''
    
    
    percentage_port = []
    for i,j in zip(port,value_of_stocks):
        value = i*j
        percenrage = value/valuation
    
        percentage_port.append(float(percenrage))
        
    return percentage_port