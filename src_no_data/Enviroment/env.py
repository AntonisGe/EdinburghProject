# actual enviroment that works

from utils import *
from p_finders import *
from cluster import *

class Enviroment:
    def __init__(self,start=0,end=2871,cash=10000,tc = 0.1, weights = False, stocks = stocks):
    
        self.stocks = stocks
        self.num_stocks = len(stocks)
        
        self.start = start
        self.end = end
        self.initial_cash = cash
        self.tc = tc
        self.weights = weights
        
        self.reset()
        self.__data = [pd.read_csv( dir_path + f"\\Data\\{stock}.csv") for stock in stocks]
        self.closes = np.zeros([self.num_stocks,total_timesteps])
        for number,stock in enumerate(self.__data):
            self.closes[number] = list(stock.Close)
        self.values = self.closes
        
        
    def reset(self):
        
        self.cash = self.initial_cash
        self.cash_reserved = 0
        self.money_owed = [0 for _ in range(self.num_stocks)]
        
        
        self.valuation = self.initial_cash
        self.timestep = self.start
        self.port = [0 for _ in range(self.num_stocks)]
        self.record = np.zeros(total_timesteps)
        self.cashes = np.zeros(total_timesteps)
        self.stonks = np.zeros([self.num_stocks,total_timesteps])
        self.total_tc = 0
                
        
    def change_and_reset(self,start=None,end=None,stocks = None):
              
        if start != None: self.start = start
        if end != None: self.end = end
        if start != None or end != None: self.total = self.start - self.end
        
        self.reset()
        
    def tra_costs(self,port):
        
        tc = np.abs(port - np.array(self.port))
        tc = (self.tc/100)*tc*self.closes[:,self.timestep]
        tc = np.sum(tc)
        
        return tc

    def available_cash(self,port):
        '''port is np array'''
        
        if not (port>=0).all():
            cash_reserved = np.sum(port[port<0]*self.closes[:,self.timestep][port<0])
        else:
            cash_reserved = 0

        cash = self.cash
        money_owed,cash = self.find_short_needs(port,cash)
        cash = self.find_long_needs(port,cash)
        tc = self.tra_costs(port)
        cash = cash - tc
        
        continue_ = cash + float(cash_reserved) > 0

        return continue_,cash,cash_reserved,money_owed,tc
        
    def find_long_needs(self,port,cash):
        
        for old,new,stock_no in zip(self.port,list(port),range(self.num_stocks)):
            if new >= 0:
                if new!=0 : assert old >= 0
                if not old<0:
                    cash += (old - new)*self.closes[stock_no,self.timestep]
                     
        return cash
                    
        
    def find_short_needs(self,port,cash):
        '''port is np array'''
        money_owed = self.money_owed.copy()
        for old,new,stock_no in zip(self.port,list(port),range(self.num_stocks)):
            if new <=0:
                if new != 0: assert old <= 0
                
                if old <=0:
                    new = - new
                    old = - old
                    mo = money_owed[stock_no]
                    
                    if new >= old:
                        mo += (new - old)*self.closes[stock_no,self.timestep]
                        money_owed[stock_no] = mo
                        
                    else:
                        cash_earned = ((old-new)/old)*mo
                        cost_to_buy = (old-new)*self.closes[stock_no,self.timestep]
                        cash += cash_earned - cost_to_buy
                        money_owed[stock_no] = mo - cash_earned
                 
        return money_owed,cash
            
    
    def available_port(self,port):
        '''port is a np array'''
        
        portofolio = list(port)
        continue_,cash,cash_reserved,money_owed,tc = self.available_cash(port)
       
        while not continue_:
            
            break_ = False
            while not break_:
                remove = np.random.choice(self.num_stocks)
                
                if portofolio[remove] < 0:
                    portofolio[remove] += 1
                    break_ = True
                elif portofolio[remove] > self.port[remove]:
                    portofolio[remove] -= 1
                    break_ = True
    
            continue_,cash,cash_reserved,money_owed,tc = self.available_cash(np.array(portofolio))

        return np.array(portofolio),cash,cash_reserved,money_owed,tc
        
    def find_new_port_percentage(self):
         
        if self.timestep < self.end:
            values = self.closes[:,self.timestep]
            valuation = self.valuation
            new_por = np.array(self.port)
            
            new_percentage_port = values*new_por/valuation
            
        else:
            new_percentage_port = [0 for _ in range(self.num_stocks)]
        
        return new_percentage_port
        
        
    def step(self,port,prit = False):
        
        if self.weights:
            port = [int(i*self.valuation/self.closes[stock,self.timestep]) for stock,i in enumerate(port)]
            
        port,cash,cash_reserved,money_owed,tc = self.available_port(np.array(port))
        
        if not (port <0).all():
            long_worth = np.sum(port[port>=0]*self.closes[:,self.timestep][port>=0])
        else: long_worth=0
        
        if prit:
            price = self.closes[0,self.timestep]
            print(f'Bought at {price} {port[0] - self.port[0]} stocks.')
        
        self.total_tc += tc
        self.cash = cash
        self.money_owed = money_owed
        self.cash_reserved = cash_reserved
        self.valuation = self.cash + cash_reserved + sum(money_owed) + float(long_worth)
        self.port = list(port)
        self.record[self.timestep] = self.valuation
        self.stonks[:,self.timestep] = self.port
        self.cashes[self.timestep] = self.cash
        self.timestep +=1
        
        return self.find_new_port_percentage()
        

if __name__ == '__main__':
    env = Enviroment(start = 50, end = 500,stocks = ['BA'],tc =0)
    env.step([-1],prit= True)
    env.step([0],prit= True)
