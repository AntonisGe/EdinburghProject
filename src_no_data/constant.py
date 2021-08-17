# CA version 1

from settings_CAA import *
from utils import *
from risks import *
from cvx import *
from decoders import *
from numpy import arange
from env import *
# val = []
# for c1 in arange(0,1.0001,0.05):
    # c3 = 1 - c1

    # env = Enviroment(start = 400, end = 1750,weights=True)

    # port = [0 for _ in range(29)]
    # for day in range(400,1750):

        # is_Friday = day_names_signals[day] == 1
                
        # ssignals = list(valispolicy[day-400,:])
        # lsignals = list(valilpolicy[day-400,:])

        # signals,port = decoder(ssignals,lsignals,port)

        # if is_Friday:                    
            # sigma = sharpe(env.values,env.timestep,30)
            # try:
                # port = quadratic(sigma,port,signals,c1,-c3)
            # except: pass
            # port = env.step(port)               
        # else:
            # port = env.step(port)
            
    # env.step([0 for _ in range(29)])

    # val.append(env.valuation)

# r = arange(0,1.0001,0.05)
# from numpy import mean
# smoothed = [mean(val[max(i-3,0):i]) for i in range(1,22)] 
# import matplotlib.pyplot as plt
# plt.plot(r,[i-10000 for i in val],label = 'Profit')
# plt.axhline(y=0.5, color='grey', linestyle='--',label='Break Even')
# plt.legend()
# plt.xlabel('c1')
# plt.ylabel('Profit ($)')
# plt.title('Profit on Const. CAM for c1 ($10K start) Valid. Time')
# plt.show()
 
c1 = 1
c3 = 1 - c1

env = Enviroment(start = 400, end = 1750,weights = True)

port = [0 for _ in range(29)]
for day in range(400,1750):

    is_Friday = day_names_signals[day] == 1
            
    ssignals = list(valispolicy[day-400,:])
    lsignals = list(valilpolicy[day-400,:])

    old_p = port
    signals,port = decoder(ssignals,lsignals,port)

    if is_Friday:                    
        sigma = sharpe(env.values,env.timestep,30)
        try:
            port = quadratic(sigma,port,signals,c1,-c3)
            port = [i if abs(i) > 1e-3 else 0 for i in port]
        except: pass
        port = env.step(port)
    else:
        port = env.step(port)
    print(env.valuation,env.port)
env.step([0 for _ in range(29)])
