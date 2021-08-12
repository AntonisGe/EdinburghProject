from utils import *
from p_finders import *
from cluster import *

class Enviroment:
    def __init__(self,start=50,end=total_timesteps-1,stocks = stocks,cash = 10000,transaction_cost = 0.1,p_finder = weight
                    ,variance=0.4,lookback = 20,n = 20,m = 5, caa_lookback = 3,cluster = None):
                    
        assert p_finder in [weight,shares]
        assert lookback <= start
        assert end < total_timesteps
        
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
        self.n = n
        self.m = m
        self.cluster = cluster
        
        self.caa_lookback = caa_lookback
    
        self.__data = [pd.read_csv( dir_path + f'\\Data\\{stock}.csv') for stock in stocks]
        self.__GLD = pd.read_csv(dir_path + f'\\Data\\GLD.csv').Close
        self.__SPY = pd.read_csv(dir_path + f'\\Data\\SPY.csv').Close
        
        self.reset()
        self.record = Record(values = self.values,num_stocks = self.num_stocks, cash = cash)

        self.step(list(self.portofolio))
        
        
    def step(self,w: list):
        '''
        Takes a step into the next timestep
        
        inputs:
        w -> a list with number or shares or percentages of port of each stock
        output:  the new percentage portofolio of each stock in the next timestep
        '''
        
        assert len(w) == self.num_stocks
        
        if self.timestep == self.end:
            assert sum([abs(i) for i in w]) == 0
        
        if self.timestep == self.end+1:
            print('Reset enviroment.')
            return

        new_por,tcosts,cash_left = self.portofolio_finder(self,w,self.portofolio)
        # ADD CODE BELOW HERE
        old_por = self.portofolio
        for new,old,stock_no in zip(new_por,old_por,range(self.num_stocks)):
            old = -min(old,0)
            new = -min(new,0)
            
            if new>=old:
                self.fake_cash[stock_no] += (new-old)*self.closes[stock_no,self.timestep]
            elif old>new:
                fake_cash = self.fake_cash[stock_no]
                
                cash_earned = ((old-new)/old)*fake_cash
                fake_cash = fake_cash - cash_earned
                
                self.fake_cash[stock_no] = fake_cash
                #cash_left += cash_earned
        # ADD CODE ABOVE HERE 
        self.timestep += 1
        value_of_new_port = self.value_of_port(new_por)
        self.record.store(self.timestep,new_por,tcosts,value_of_new_port,cash_left)    
        
        self.portofolio = new_por
        self.cash = cash_left
        self.valuation = value_of_new_port + cash_left
        
        # if self.timestep == self.end:
            # self.step([0 for _ in range(self.num_stocks)])
         
        return self.find_new_port_percentage() # return new percentage of portofolio

                
    def SSA_matrix(self,stock,timestep = None):
    
        if timestep == None: timestep = self.timestep
        
        #timestep += 1
        
        stock_index = self.stocks.index(stock)
        
        matrix = np.zeros((4,self.lookback))
        stock_number = self.stocks.index(stock)
        lookback_close = self.closes[stock_number,timestep-self.lookback]#-self.n]
        
        matrix[:3,:] = self.data[1:,stock_number,timestep-self.lookback:timestep] / lookback_close # does not include open 
        
        # mean = np.mean(self.volumes[stock_index,timestep-self.lookback-self.n:timestep])
        # std = np.std(self.volumes[stock_index,timestep-self.lookback-self.n:timestep])
        mean = 0
        std = 1
        matrix[3:] = (self.volumes[stock_index,timestep-self.lookback:timestep] - mean)/std
        #matrix[5,:] = self.SPY[timestep-self.lookback:timestep]
        #matrix[6,:] = self.GLD[timestep-self.lookback:timestep]
        
        return np.transpose(matrix)
        
    def feature_matrix(self,stock,timestep = None):
        
        if timestep == None: timestep = self.timestep
        stock_index = self.stocks.index(stock)
        
        return self.features[stock_index,timestep-self.lookback+1:timestep+1,:]
                
    # def train_cluster(self):
        
        # num_features = len(self.GOAT_features(self.stocks[0]))

        # if self.cluster != None:
            
            # how_much_back = self.start-self.lookback
            # if self.end - how_much_back <200 and how_much_back >250:
                # how_much_back = how_much_back - 200
            
            # self.cluster.train(how_much_back,self.end)

    def find_features(self):
        
        num_features = len(self.GOAT_features(self.stocks[0]))
        
        if self.cluster != None:
            
            array = np.zeros((self.num_stocks,total_timesteps,num_features))
            
            for stock in self.stocks:
                stock_index = self.stocks.index(stock)
                for day in range(self.start-self.lookback,self.end):
                    features = self.GOAT_features(stock,timestep = day)
                    
                    array[stock_index,day,:] = features
                    
            self.features = self.cluster.find_features(array)
            self.array = array
            
    def CAA_find_features(self):
        
        num_features = len(self.CAA_f())
        
        array = np.zeros((total_timesteps,num_features))
        for day in range(self.start-self.caa_lookback,self.end):
            features = self.CAA_f(timestep = day)
            
            array[day,:] = features
            
        self.caa_features = array
            
            
    def CAA_f(self,timestep = None):
        if timestep ==None: timestep = self.timestep
        
        
        closes_spy = np.array(self.SPY)[timestep+1 - self.n: timestep+1]
        closes_gld = np.array(self.GLD)[timestep+1 - self.n: timestep+1]
        
        close_spy = np.array(self.SPY)[timestep]
        close_gld = np.array(self.GLD)[timestep]
        
        features = []
        
        for closes,close_ in zip([closes_spy,closes_gld],[close_spy,close_gld]):
        
            mean_close = np.mean(closes)
            std_close = np.std(closes)
            
            z_score_close = (close_ - mean_close)/std_close
            features.append(z_score_close)
            mag_change = close_/mean_close - 1
            features.append(mag_change)
            
            gains = closes[1:] - closes[:-1]

            temp = gains[gains > 0]
            positive_ave = np.mean(temp) if len(temp) != 0 else 0
            temp = gains[gains < 0]
            negative_ave = np.mean(temp) if len(temp) != 0 else 0
            #print(gains)
            if negative_ave == 0: rsi = 100
            elif positive_ave == negative_ave: rsi = 50
            else: rsi = 100 - (100/(1-positive_ave/negative_ave))
            features.append(rsi)
            
            gains = closes[1:] - closes[:-1]
            
        return features
        
    def GOAT_features(self,stock,timestep = None):
        
        if timestep == None: timestep = self.timestep
        stock_index = self.stocks.index(stock)
        features = []
        
        opens = self.opens[stock_index,timestep+1-self.n:timestep+1] 
        highs = self.highs[stock_index,timestep+1-self.n:timestep+1] 
        lows = self.lows[stock_index,timestep+1-self.n:timestep+1] 
        closes = self.closes[stock_index,timestep+1-self.n:timestep+1] 
        volumes = self.volumes[stock_index,timestep+1-self.n:timestep+1]    
        
        open_ = self.opens[stock_index,timestep] 
        high_ = self.highs[stock_index,timestep] 
        low_ = self.lows[stock_index,timestep] 
        close_ = self.closes[stock_index,timestep] 
        volume_ = self.volumes[stock_index,timestep]
        
        mean_close = np.mean(closes)
        std_close = np.std(closes)
        
        z_score_close = (close_ - mean_close)/std_close
        features.append(z_score_close)
        mag_change = close_/mean_close - 1
        features.append(mag_change)
        
        gains = closes[1:] - closes[:-1]
        
        temp = gains[gains > 0]
        positive_ave = np.mean(temp) if len(temp) != 0 else 0
        temp = gains[gains < 0]
        negative_ave = np.mean(temp) if len(temp) != 0 else 0

        if negative_ave == 0: rsi = 100
        elif positive_ave == negative_ave: rsi = 50
        else: rsi = 100 - (100/(1-positive_ave/negative_ave))
        features.append(rsi)
        
        past_26 = self.closes[stock_index,timestep - 25:timestep+1]
        #print(past_26.shape,past_26)
        theta = 0.2
        ema26 = np.ma.average(past_26,weights = [(1-theta)**i for i in np.arange(25,-1,-1)])/mean_close
        
        past_12 = self.closes[stock_index,timestep - 11:timestep+1]
        ema12 = np.ma.average(past_12,weights = [(1-theta)**i for i in np.arange(11,-1,-1)])/mean_close
        
        past_9 = self.closes[stock_index,timestep - 8:timestep+1]
        ema9 = np.ma.average(past_9,weights = [(1-theta)**i for i in np.arange(8,-1,-1)])/mean_close
        
        macd = ema12 - ema26
        features.append(macd)
        signal_macd = ema9 - ema12
        features.append(signal_macd)
        
        v = volumes[1:]
        obv = np.sum(np.sign(gains) * v)
        obv = obv/np.mean(volumes)
        features.append(obv)
        
        new_highs = 0
        max_ = -1
        for h in highs:
            if h > max_:
                max_ = h
                new_highs += 1
                
        new_lows = 0
        low_ = 999999
        for l in lows:
            if l<low_:
                low_ = l
                new_lows += 1
                
        per_high = new_highs/(new_highs+new_lows)
        per_low = new_lows/(new_highs+new_lows)
        
        features.append(per_high)
        features.append(per_low)

        return features
    
    def CAA_matrix(self):
        
        m = self.caa_features[self.timestep - self.caa_lookback+1: self.timestep+1,:]
        return m.reshape(self.caa_lookback*6)
        
    def find_new_port_percentage(self):
         
        if self.timestep < self.end:
            values = self.values[:,self.timestep]
            valuation = self.valuation
            new_por = np.array(self.portofolio)
            
            new_percentage_port = values*new_por/valuation
            
        else:
            new_percentage_port = [0 for _ in range(self.num_stocks)]
        
        return new_percentage_port
    
    def ema_calculator(self,stock,t):
        
        stock_index = self.stocks.index(stock)
        
        sequence = self.values[stock_index,t+1-self.n:t+1]
        price = self.values[stock_index,t]
        
        summation = np.sum(sequence)
    
        return 2*price/(self.n+1) + (summation/self.n)*(100-2/(self.n+1))
    def paper_1_data(self,stock,t=None):
    
        stock_index = self.stocks.index(stock)
    
        if t == None: t = self.timestep
    
        sequence = self.values[stock_index,t+1-self.n:t+1]
        #print(sequence)
        
        mean = np.mean(sequence)
        std = np.std(sequence)
        price = self.values[stock_index,t]
        
        z_score = (price - mean)/std
        price_change = price/mean - 1
        
        volume_sequence = self.volumes[stock_index,t+1-self.n:t+1]
        
        volume_mean = np.mean(volume_sequence)
        volume_std = np.std(volume_sequence)
        volume = self.volumes[stock_index,t]
        
        z_score_volume = (volume - volume_mean)/volume_std
        volume_change = volume/volume_mean - 1
        
        ema_at_t = self.ema_calculator(stock,t)
        ema_at_t_minus_m = self.ema_calculator(stock,t-self.m)
        volatility = (ema_at_t - ema_at_t_minus_m)/ema_at_t_minus_m
        
        return [float(i) for i in [z_score,price_change,z_score_volume,volume_change,volatility]]
                        
    def tcost(self,new_por):
    
        old_por = self.portofolio
        
        difference = np.abs(np.array(new_por) - old_por)
        transaction_costs = self.transaction_cost*difference*self.values[:,self.timestep]/100
        
        return transaction_costs
    
    def value_of_port(self,new_por):
        new_por1 = [abs(i) for i in new_por]
        
        
        if sum(new_por1) != 0:
            vs = np.array(new_por) * self.values[:,self.timestep]
            value = sum(vs)
        else:
            value = 0
        
        return value + 2*sum(self.fake_cash)
        
    def value_of_port2(self,new_por):
        new_por = [abs(i) for i in new_por]
        
        
        if sum(new_por) != 0:
            vs = np.array(new_por) * self.values[:,self.timestep]
            value = sum(vs)
        else:
            value = 0
        
        return value
        
    def change_and_reset(self,start=None,end=None,stocks = None):
        
        if stocks != None:
            assert len(stocks) == self.num_stocks
            self.stocks = stocks
            self.__data = [pd.read_csv( dir_path + f'\\Data\\{stock}.csv') for stock in stocks]
        
        if start != None: self.start = start
        if end != None: self.end = end
        if start != None or end != None: self.total = self.start - self.end
        
        self.reset()
        
    def reset(self):  
        self.timestep = self.start - 1
        self.cash = self.initial_cash
        self.valuation = self.initial_cash
        self.portofolio = np.zeros(self.num_stocks)
        self.fake_cash = [0 for _ in range(self.num_stocks)]
        
        self.generate_data()
        
        try: 
            self.record.reset()
            self.step(list(self.portofolio))
        except AttributeError:
            pass
            
        self.find_features()
        self.CAA_find_features()
        
        
    def CAA_reward(self,first_day,last_day):
        temp = self.record.valuation[first_day:last_day]
        temp2 = self.record.valuation[first_day+1:last_day+1]
        
        rewards = temp2 - temp
        rewards = list(rewards)
        
        return rewards
    
    def SSA_reward(self,first_day,last_day,stock):
        
        stock_index = self.stocks.index(stock)
        
        stocks_owned = self.record.portofolios[first_day,stock_index]
        tcost = self.record.tcosts[first_day,stock_index]
        
        
        
    def SSA_reward_old(self,first_day,last_day,stock):
    
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
        self.volumes_ = np.zeros([self.num_stocks,total_timesteps])
        for number,stock in enumerate(self.__data):
            days_volume = list(stock.Volume)
            one_day_before_volume = list(stock.Volume)[:-1]
            one_day_before_volume.insert(0,days_volume[0])
            
            self.volumes_[number] = [i/j for i,j in zip(days_volume,one_day_before_volume)]
            
        # Generate_Volumes_
        self.volumes = np.zeros([self.num_stocks,total_timesteps])
        for number,stock in enumerate(self.__data):
            days_volume = list(stock.Volume)
            
            self.volumes[number] = days_volume
                
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
        self.cash = np.zeros(total_timesteps+1)
        self.valuation = np.zeros(total_timesteps+1)
        self.portofolios = np.zeros([total_timesteps+1,self.num_stocks])
        self.tcosts = np.zeros([total_timesteps+1,self.num_stocks])
        self.portofolio_values = np.zeros(total_timesteps+1)
        
      
if __name__ == '__main__':

    start = 50
    end = 500
    
    env = Enviroment(stocks = ['AAPL','BA'], start = start, end = end, transaction_cost=0,variance = 0,p_finder=shares,lookback = 20,cash = 1000)
    env.closes[0,:] = [100 for _ in range(env.closes.shape[1])]
    env.closes[1,:] = [100 for _ in range(env.closes.shape[1])]
    env.values[0,:] = [100 for _ in range(env.closes.shape[1])]
    env.values[1,:] = [100 for _ in range(env.closes.shape[1])]
    env.step([2,0])
    env.step([-1,0])
    #env.step([-3])
    #env.step([0,50])
    #port = env.step([-1,0])
    
    
    
    