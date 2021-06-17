from utils import *


def normalizer(env):
    array = []

    for day in range(env.start,env.end):
        for stock in env.stocks:
            array.append(env.SSA_matrix(stock))
            

    array = np.array(array)
    mean = np.mean(array, axis = 0)
    std = np.std(array, axis = 0) + 1e-20

    three = np.zeros_like(mean) + 3
    mthree = np.zeros_like(mean) - 3

    func = lambda x: np.minimum(np.maximum((x-mean)/std,mthree),three)
     
    env.reset()
    
    return func
    

if __name__ == '__main__':
    from enviroment import *

    env = Enviroment(start = 25, end =55)
    x = env.SSA_matrix('AAPL')
    
    normal = normalizer(env) # this is a function!
    y = normal(x)