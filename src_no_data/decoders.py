# this file is here for the decoder on line 64. This is a mini version of the
# decoder. In the paper I explained the algorithm in one step
# in my code I do in two steps. The other step is in the \Enviroment\cvx.py

from utils import *

def helper(ssignals,lsignals,port):
    signals = []
    for s,l,p, n in zip(ssignals,lsignals,port,range(len(stocks))):
        if p == 0:
            if s ==2 and l == 2:
                signals.append(0)
            elif s == 2 and l != 2:
                signals.append(-1)
            elif s != 2 and l == 2:
                signals.append(1)
            else:
                signals.append(0)
        elif p > 0:
            if s == 2 or l == 0:
                signals.append(0)                 
                port[n] = 0.0
            else:
                signals.append(0)
                
        elif p < 0:
            if s == 0 or l == 2:
                signals.append(0)
                port[n] = 0.0
            else:
                signals.append(0)                   
        else:
            print(p)
        
    port = [float(i) for i in port]
    
    return signals,port
    
def helper(ssignals,lsignals,port):
    signals = []
    for s,l,p, n in zip(ssignals,lsignals,port,range(len(stocks))):
        if p == 0:
            if l == 2:
                signals.append(1)
            else:
                signals.append(0)
        elif p > 0:
            if l == 0:
                signals.append(0)                 
                port[n] = 0.0
            elif l == 2:
                signals.append(1)
            else:
                signals.append(0)      
        else:
            print(p)
        
    port = [float(i) for i in port]
    
    return signals,port

# the decoder here is different than the one on the paper
# I have done in two steps there is a second algorithm on the cvx files
# that when merging the decoder here with the algorihtm on the cvx file 
# it gives the same decoder like the one in the paper.
def decoder(ssignals,lsignals,port):
    signals = []
    for s,l,p, n in zip(ssignals,lsignals,port,range(len(stocks))):
        if p == 0:
            if l == 2:
                signals.append(1)
            elif s == 2:
                signals.append(-1)
            else:
                signals.append(0)
        elif p > 0:
            if l == 0:
                signals.append(0)                 
                port[n] = 0.0
            elif l == 2:
                signals.append(0)
            else:
                signals.append(0)

        elif p<0:
            if s == 0:
                signals.append(0)
                port[n] = 0.0
            elif s == 2:
                signals.append(0)
            else:
                signals.append(0)
        else:
            print(p)
        
    port = [float(i) for i in port]
    
    return signals,port