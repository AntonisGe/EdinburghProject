# finally did not use it . Just used a sklearn library isntead

from utils import *

def normalizer_function(env,first_x_to_normalize = 4):
    array = []

    for day in range(env.start,env.end):
        for stock in env.stocks:
            array.append(env.SSA_matrix(stock)[:,:first_x_to_normalize])
            

    array = np.array(array)
    mean = np.mean(array, axis = 0)
    std = np.std(array, axis = 0) + 1e-20

    three = np.zeros_like(mean) + 3
    mthree = np.zeros_like(mean) - 3

    def func(x):
        
        x_small = x[:,:first_x_to_normalize]
        
        x_small = np.maximum(np.minimum((x_small-mean)/std,three),mthree)
        
        value = np.zeros_like(x)
        value[:,:first_x_to_normalize] = x_small
        value[:,first_x_to_normalize:] = x[:,first_x_to_normalize:]
        
        return value
     
    env.reset()
    
    return func
    

if __name__ == '__main__':
    from enviroment import *

    env = Enviroment(start = 25, end =400,lookback = 5)
    x = env.SSA_matrix('AAPL')
    
    normal = normalizer(env) # this is a function!
    y = normal(x)