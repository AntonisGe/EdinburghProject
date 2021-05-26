from utils import *

def weight(env,x:list):
    '''Takes a weight vector of portofolios and converts into actual number of stocks to buy'''

    assert sum([abs(i) for i in x]) <1.00001
    
    cash = env.valuation  #env.value_of_port(env.portofolio) + env.cash
   
    portofolio = [int(i*cash/env.values[stock,env.timestep]) for stock,i in enumerate(x)]

    return feasibility_checker(env,portofolio)

def feasibility_checker(env,portofolio):
    
    cash = env.valuation #env.value_of_port(env.portofolio) + env.cash

    value_of_new_port = env.value_of_port(portofolio)    
    transaction_costs = env.tcost(portofolio)
    transaction_cost = sum(transaction_costs)
    
    cash_left = cash - transaction_cost - value_of_new_port
    while cash_left < 0:
        remove = np.random.choice(range(env.num_stocks))
        
        temp = portofolio[remove]
        portofolio[remove] = temp - abs(temp)/temp
        
        value_of_new_port = env.value_of_port(portofolio)
        transaction_costs = env.tcost(portofolio)
        transaction_cost = sum(transaction_costs)
        cash_left = cash - transaction_cost - value_of_new_port
    
    return portofolio,transaction_costs,value_of_new_port,cash_left