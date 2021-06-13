from utils import *

def return_convert(values,timestep,lookback):

    assert timestep > lookback

    current_values = values[:,timestep-lookback+1:timestep]
    one_behind = values[:,timestep-lookback:timestep-1]
    
    returns = current_values/one_behind
    
    return returns
    
def sharpe(values,timestep,lookback):
    
    returns = return_convert(values,timestep,lookback)
    covariance = np.cov(returns)
    
    try:
        covariance.shape[0]
    except:
        covariance = np.array([float(covariance)])
        
    return covariance
    
def cvar(values,timestep,lookback,q=0.2):

    def cvar_helper_long(data):
        quantile = np.quantile(data,q)
        values_below = data[data<quantile]
        
        return np.mean(values_below)

    def cvar_helper_short(data):
        quantile = np.quantile(data,1-q)
        values_below = data[data>quantile]
        
        return np.mean(values_below)

    returns = return_convert(values,timestep,lookback)
    
    cvar_long = np.apply_along_axis(cvar_helper_long,1,returns)
    cvar_short = np.apply_along_axis(cvar_helper_short,1,returns)
    
    return cvar_long,cvar_short


if __name__ == '__main__':
    from enviroment import *
    
    env = Enviroment()
    sigma = sharpe(env.values,60,20)