import pandas as pd
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

stocks = ['AXP','AMGN','AAPL','BA','CAT','CSCO','CVX','GS','HD','HON','IBM','INTC','KO','JPM','MCD',
'MMM','MRK','MSFT','NKE','PG','TRV','UNH','CRM','VZ','V','WBA','WMT','DIS','JNJ']

times = ['09:00','10:00','11:00','12:00','13:00','14:00','15:00']

dates = np.unique(list(pd.read_csv(dir_path + r'\Data\AAPL.csv').Date))

### DAY NAMES
day_names = list(pd.read_csv(dir_path + r'\Data\AAPL.csv').Day)
day_names_unique = ['Monday','Tuesday','Wednesday','Thursday','Friday']

day_names_b = []
for d in day_names:
    value = day_names_unique.index(d)
    day_names_b.append(value)

day_names_signals = []
d_old = 4
for d in day_names_b:
    if d<d_old:
        signal = 1        
    else:
        signal = 0
        
    day_names_signals.append(signal)
    d_old = d
total_weeks = sum(day_names_signals)
Friday_timesteps = []
for d in range(len(day_names_signals[:-1])):
    if day_names_signals[d+1] == 1:
        Friday_timesteps.append(d)

####################################


total_timesteps = pd.read_csv(dir_path + r'\Data\AAPL.csv').shape[0]

def decode_timestep(timestep):
    
    date_counter = timestep // 7
    time_counter = timestep % 7
    
    date = dates[date_counter]
    time = times[time_counter]
    
    return date,time
