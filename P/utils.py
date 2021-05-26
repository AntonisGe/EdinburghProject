import pandas as pd
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

stocks = ['AXP','AMGN','AAPL','BA','CAT','CSCO','CVX','GS','HD','HON','IBM','INTC','KO','JPM','MCD',
'MMM','MRK','MSFT','NKE','PG','TRV','UNH','CRM','VZ','V','WBA','WMT','DIS','JNJ']

times = ['09:00','10:00','11:00','12:00','13:00','14:00','15:00']

dates = np.unique(list(pd.read_csv(dir_path + r'\Data\AAPL.csv').Date))

total_timesteps = pd.read_csv(dir_path + r'\Data\AAPL.csv').shape[0]

def decode_timestep(timestep):
    
    date_counter = timestep // 7
    time_counter = timestep % 7
    
    date = dates[date_counter]
    time = times[time_counter]
    
    return date,time