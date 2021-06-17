import yfinance as yf
from utils import *
from multiprocessing import Pool

'''
This script downloads the data from yahoo finance
then manipulates it and stores in the directory 
'''

path_to_save = dir_path + '\Data\{}.csv'
start = '2010-01-01'
end = '2021-06-01'

def run_stock_individually(stock):
    df = yf.download(stock,start=start,end=end)
    
    save = pd.DataFrame(columns = ['Date','Open','High','Low','Volume','Close','Time','Day'])
    counter = 0
    for index in df.index:
    
        temp = df.loc[index]
    
        day = str(index)[:10]
    
        save.loc[counter] = [day,temp['Open'],temp['High'],temp['Low'],temp['Volume'],temp['Close'],'09:00',pd.Timestamp(day).day_name()]
        counter += 1
        
    save.to_csv(path_to_save.format(stock),index=False)
    

if __name__ == '__main__':

    extended_stocks = stocks + ['GLD','SPY']

    with Pool(10) as p:
        p.map(run_stock_individually,extended_stocks)
        p.close()

if True:
    check = []    
    for stock in stocks:
        df = pd.read_csv(path_to_save.format(stock))
        check.append(list(df.Date)[2500])