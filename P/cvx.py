from utils import *
from risks import *
from cvxopt import solvers,matrix

def cvx_helper(old_p,signals,num_of_stocks):

    assert len(old_p) == len(signals)
    assert len(signals) == num_of_stocks

    long_signals = [0.0 for _ in range(num_of_stocks)]
    short_signals = [0.0 for _ in range(num_of_stocks)]
    constant_signals = [0.0 for _ in range(num_of_stocks)]

    for stock_number,p,signal in zip(range(num_of_stocks),old_p,signals):
        if p > 0: long_signals[stock_number] = 1.0
        elif p<0: short_signals[stock_number] = 1.0
        else:
            if signal ==1: long_signals[stock_number] = 1.0
            elif signal == -1: short_signals[stock_number] = 1.0
        if signal == 0: constant_signals[stock_number] = 1.0

    return long_signals,short_signals,constant_signals

def quadratic(covariance,old_p,signals,c1,c3,max_iters = 1000):

    for i in old_p: assert type(i) == float

    num_of_stocks = covariance.shape[0]

    long_signals,short_signals,constant_signals = cvx_helper(old_p,signals,num_of_stocks)

    P = matrix(covariance)
    q = matrix([0.0 for _ in range(num_of_stocks)])
    
    fill_A = []
    fill_b = []
    
    for i,j in zip([long_signals,short_signals],[c1,c3]):
        if sum(abs(k) for k in i) !=0 : 
            fill_A.append(i)
            fill_b.append(j)
        else:
            if j != 0:
                return 'Not Solvable, Error 1'
    
    for p,number,signal in zip(old_p,range(num_of_stocks),constant_signals):
        if signal == 1.0:
            filler = [0.0 for _ in range(num_of_stocks)]
            filler[number] = 1.0
            
            rank_check_passed = True
            for n,check in enumerate(fill_A):
                if check == filler:
                    rank_check_passed = False
                    if fill_b[n] != p:
                        return 'Not Solvable','Error 2'
            
            if rank_check_passed:
                fill_A.append(filler)
                fill_b.append(p)
        
    fill_G = []
    fill_h = []

    for number,signal,p in zip(range(num_of_stocks),signals,old_p):
        if signal != 0:
            appention = [0.0 for _ in range(num_of_stocks)]
            appention[number] = -float(signal)
            
            fill_G.append(appention)
            fill_h.append(-signal*p)
    
    A = matrix(fill_A).trans()
    b = matrix(fill_b)
    G = matrix(fill_G).trans()
    h = matrix(fill_h)

    solvers.options['show_progress'] = False # supress tilting output
    solvers.options['maxiters'] = max_iters
    sol = solvers.qp(P,q,G,h,A,b)

    return [round(sol['x'][i],4) for i in range(num_of_stocks)]


if __name__ == '__main__':
    from risks import *
    from enviroment import *
    env = Enviroment()
    risk = sharpe(env.values,21,20)
    
    old_p = [0.3 for _ in range(3)]
    signals = [1,0,-1]
    c1,c3 = (0.35,-0.1)
    
    test = quadratic(risk,old_p,signals,c1,c3)
    
    
