# this is a bunch of small helper functions

from utils import *

def weight(env,x:list,old_p:list):
    '''Takes a weight vector of portofolios and converts into actual number of stocks to buy'''
    #assert sum([abs(i) for i in x]) <1.00001
    
    cash = env.valuation  #env.value_of_port(env.portofolio) + env.cash
   
    portofolio = [int(i*cash/env.values[stock,env.timestep]) for stock,i in enumerate(x)]
    return feasibility_checker(env,portofolio,old_p)
    
def shares(env,x:list,old_p:list):
    
    portofolio = x
    
    return feasibility_checker(env,portofolio,old_p)

def feasibility_checker(env,portofolio,old_p):
    
    cash = env.valuation #env.value_of_port(env.portofolio) + env.cash
    
    value_of_new_port = env.value_of_port(portofolio)
    transaction_costs = env.tcost(portofolio)
    transaction_cost = sum(transaction_costs)
    
    cash_left = cash - transaction_cost - value_of_new_port
    while cash_left < 0:
        
        while True:
            remove = np.random.choice(range(env.num_stocks))
            if old_p[remove] != portofolio[remove]:
                break # we want to remove stocks that have changed in our port compared to the old one
        
        temp = portofolio[remove]
        old_temp = old_p[remove]
        direction = old_temp - temp
        
        portofolio[remove] = temp + abs(direction)/direction
        
        value_of_new_port = env.value_of_port(portofolio)
        transaction_costs = env.tcost(portofolio)
        transaction_cost = sum(transaction_costs)
        cash_left = cash - transaction_cost - value_of_new_port
    
    return portofolio,transaction_costs,cash_left
def cost_of_stocks(env,new_por,old_por):    
    cost = 0
    for new,old,stock_no in zip(new_por,old_por,range(env.num_stocks)):
    
        if old>=0:
            
            if new < 0:
                cost += (abs(new) - old)*env.closes[stock_no,env.timestep]
            else:
                cost += abs(abs(old)-new)*env.closes[stock_no,env.timestep]
            
        else:
            old = -min(old,0)
            new_p = new
            new = -min(new,0)
            
            if new>=old:
                cost += (new-old)*env.closes[stock_no,env.timestep]
            elif old>new:
                fake_cash = env.fake_cash[stock_no]
                cash_earned = ((old-new)/old)*fake_cash
                cost -= cash_earned
                
            cost += max(new_p,0)*env.closes[stock_no,env.timestep]
        
    return cost
    
def feasibility_checker_nope(env,portofolio,old_p):
    
    cash = env.cash #env.value_of_port(env.portofolio) + env.cash
    
    cost = cost_of_stocks(env,portofolio,old_p)
    transaction_costs = env.tcost(portofolio)
    transaction_cost = sum(transaction_costs)
    
    cash_left = cash - transaction_cost - cost
    while cash_left < 0:
        
        while True:
            remove = np.random.choice(range(env.num_stocks))
            if old_p[remove] != portofolio[remove]:
                break # we want to remove stocks that have changed in our port compared to the old one
        
        temp = portofolio[remove]
        old_temp = old_p[remove]
        direction = old_temp - temp
        
        portofolio[remove] = temp + abs(direction)/direction
        
        cost = cost_of_stocks(env,portofolio,old_p)
        transaction_costs = env.tcost(portofolio)
        transaction_cost = sum(transaction_costs)
        cash_left = cash - transaction_cost - cost
    
    return portofolio,transaction_costs,cash_left
    
if __name__ == '__main__':
    from enviroment import *
    stocks = ['AAPL']
    
    env = Enviroment(stocks = stocks)
    
    port = env.step([0.1])
    old_p = env.portofolio
    
    new_p_test = [10000]
    new_p_actual = shares(env,new_p_test,old_p)