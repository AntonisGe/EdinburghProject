from utils import *
from p_finders import *
from settings import *

class Enviroment:
    def __init__(self,start=20,end=total_timesteps,stocks = stocks,cash = 10000,transaction_cost = 0.1,p_finder = weight
                    ,variance=0.4,lookback = 20):
                    
        assert p_finder in [weight]
        assert lookback <= start
        
        self.start = start
        self.end = end
        self.total = end - start
        self.stocks = stocks
        self.num_stocks = len(stocks)
        self.variance = variance
        self.initial_cash = cash
        self.transaction_cost = transaction_cost
        self.portofolio_finder = p_finder
        self.lookback = lookback
    
        self.__data = [pd.read_csv( dir_path + f'\\Data\\{stock}.csv') for stock in stocks]
        self.__GLD = pd.read_csv(dir_path + f'\\Data\\GLD.csv').Close
        self.__SPY = pd.read_csv(dir_path + f'\\Data\\SPY.csv').Close
        
        self.reset()
        self.record = Record(values = self.values,num_stocks = self.num_stocks, cash = cash)

        self.step(list(self.portofolio))
        
        
    def step(self,w: list):
        '''w is the new portofolio'''
        
        assert len(w) == self.num_stocks
        
        if self.timestep == self.end:
            print('Cannt take step at final step. Reset enviroment.')
            return

        new_por,tcosts,cash_left = self.portofolio_finder(self,w,self.portofolio)
        # ADD CODE BELOW HERE
        
        # ADD CODE ABOVE HERE 
        self.timestep += 1
        value_of_new_port = self.value_of_port(new_por)
        self.record.store(self.timestep,new_por,tcosts,value_of_new_port,cash_left)    
        
        self.portofolio = new_por
        self.cash = cash_left
        self.valuation = value_of_new_port + cash_left
        
        return tcosts
                
    def SSA_matrix(self,stock,timestep = None):
    
        if timestep == None: timestep = self.timestep
        
        stock_index = self.stocks.index(stock)
        
        matrix = np.zeros((7,self.lookback))
        stock_number = self.stocks.index(stock)
        lookback_close = self.closes[stock_number,timestep-self.lookback]
        
        matrix[:4,:] = self.data[:,stock_number,timestep-self.lookback:timestep] / lookback_close
        matrix[4:] = self.volumes[stock_index,timestep-self.lookback:timestep]
        matrix[5,:] = self.SPY[timestep-self.lookback:timestep]
        matrix[6,:] = self.GLD[timestep-self.lookback:timestep]
        
        return np.transpose(matrix)
    def CAA_matrix(self):
    
        matrix = self.closes[:,self.timestep-self.lookback:self.timestep]
        
        return np.transpose(matrix)
                        
    def tcost(self,new_por):
    
        old_por = self.portofolio
        
        difference = np.abs(np.array(new_por) - old_por)
        transaction_costs = self.transaction_cost*difference*self.values[:,self.timestep]/100
        
        return transaction_costs
    
    def value_of_port(self,new_por):
        new_por = [abs(i) for i in new_por]
        vs = np.array(new_por) * self.values[:,self.timestep]
        value = sum(vs)
        
        return value
        
    def change_and_reset(self,start,end):
        self.start = start
        self.end = end
        self.total = start - end
        
        self.reset()
        
    def reset(self):  
        self.timestep = self.start - 1
        self.cash = self.initial_cash
        self.valuation = self.initial_cash
        self.portofolio = np.zeros(self.num_stocks)
        
        self.generate_data()
        
        try: 
            self.record.reset()
            self.step(list(self.portofolio))
        except AttributeError:
            pass
        
        
    def CAA_reward(self,first_day,last_day):
        temp = self.record.valuation[first_day:last_day]
        temp2 = self.record.valuation[first_day+1:last_day+1]
        
        rewards = temp2 - temp
        rewards = list(rewards)
        
        return rewards
        
    def SSA_reward(self,first_day,last_day,stock):
    
        stock_index = self.stocks.index(stock)
            
        rewards = []
        for day in range(first_day,last_day):
        
            day += 1
            
            stocks_owned = self.record.portofolios[day,stock_index]
            tcost = self.record.tcosts[day,stock_index]
            value = self.values [stock_index,day-1]
            value_next = self.values[stock_index,day]
            
            profit = (value_next - value)*stocks_owned - tcost
            rewards.append(profit)
            
        return rewards
    
    def generate_data(self):
    
        generate_noise = lambda:np.random.normal(1,self.variance/100,(self.num_stocks,total_timesteps))
        generate_noise_one_column = lambda: np.random.normal(1,self.variance/100,total_timesteps)
        
        # Genearate Volumes
        self.volumes = np.zeros([self.num_stocks,total_timesteps])
        for number,stock in enumerate(self.__data):
            days_volume = list(stock.Volume)
            one_day_before_volume = list(stock.Volume)[:-1]
            one_day_before_volume.insert(0,days_volume[0])
            
            self.volumes[number] = [i/j for i,j in zip(days_volume,one_day_before_volume)]
            
        # Generate Values
        self.values = np.zeros([self.num_stocks,total_timesteps])
        for number,stock in enumerate(self.__data):
            self.values[number] = list(stock.Close) # Change to Value if doing hourly
  
        white_values = generate_noise()
        self.values = self.values * white_values
        
        # Generate Closes
        self.closes = np.zeros([self.num_stocks,total_timesteps])
        for number,stock in enumerate(self.__data):
            self.closes[number] = list(stock.Close)            
        self.closes = self.closes*white_values # needs to have the same white noise with value
        
        #Generate the Rest:
        self.opens, self.lows, self.highs = (np.zeros([self.num_stocks,total_timesteps]) for _ in range(3))
        for i,j in zip([self.opens,self.lows,self.highs],['Open','Low','High']):
            for number,stock in enumerate(self.__data):
                i[number] = list(stock[j])
                
            #white_values = generate_noise()
            i *= white_values
            
        self.data = np.array([self.opens,self.highs,self.lows,self.closes])
        
        #Generate GLD,SPY
        white = generate_noise_one_column()
        self.SPY = self.__SPY*white
        white = generate_noise_one_column()
        self.GLD = self.__GLD*white
    
    
class Record:
    def __init__(self,values,num_stocks = len(stocks),cash = 10000):
        self.values = values
        self.initial_cash = cash
        self.num_stocks = num_stocks
        
        self.reset()
       
    def store(self,timestep,portofolio,tcosts,value_of_new_port,cash_left):
        
        self.portofolios[timestep,:] = portofolio
        self.tcosts[timestep,:] = tcosts
        self.portofolio_values[timestep] = value_of_new_port
        self.cash[timestep] = cash_left
        self.valuation[timestep] = cash_left + value_of_new_port
        
              
    def reset(self):
        self.cash = np.zeros(total_timesteps)
        self.valuation = np.zeros(total_timesteps)
        self.portofolios = np.zeros([total_timesteps,self.num_stocks])
        self.tcosts = np.zeros([total_timesteps,self.num_stocks])
        self.portofolio_values = np.zeros(total_timesteps)
        
      
if __name__ == '__main__':

    start = 21
    end = 24
    env = Enviroment(start = start, end = end, transaction_cost=0.5,variance = 0.4)
    
    
    env.step([0.1,0,0])
    env.step([0.1,0,0])
    
    test = env.SSA_reward(21,23,'AAPL')
    test2 = env.CAA_reward(21,22)