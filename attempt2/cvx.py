from utils import *
from risks import *
from cvxopt import solvers,matrix

def cvx_helper(old_p,signals,num_of_stocks):

    assert len(old_p) == len(signals)
    num_of_stocks = len(old_p)

    long_signals = [0.0 for _ in range(num_of_stocks)]
    short_signals = [0.0 for _ in range(num_of_stocks)]

    for stock_number,p,signal in zip(range(num_of_stocks),old_p,signals):
        if p > 0: long_signals[stock_number] = 1.0
        elif p<0: short_signals[stock_number] = 1.0
        else:
            if signal ==1: long_signals[stock_number] = 1.0
            elif signal == -1: short_signals[stock_number] = 1.0

    return long_signals,short_signals

def reduce_finder(old_p,c1,c3):
    mins = []
    
    c1_current = 0
    c3_current = 0
    
    for c in old_p:
        if c>0:c1_current += c
        elif c<0: c3_current += c

    c1_diff = max(0,c1_current - c1)
    c3_diff = -max(0,abs(c3_current) - abs(c3))
    
    for c in old_p:
        if c>0:
            value = c - c1_diff
        elif c<0:
            value = c - c3_diff
        else:
            value = 0.0
            
        mins.append(value)

    return mins

def quadratic(covariance,old_p,signals,c1,c3,max_iters = 1000):

    for i in old_p: assert type(i) == float

    num_of_stocks = covariance.shape[0]

    long_signals,short_signals = cvx_helper(old_p,signals,num_of_stocks)

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
    
    for p,number,signal in zip(old_p,range(num_of_stocks),signals):
        # if signal == 1.0:
            # filler = [0.0 for _ in range(num_of_stocks)]
            # filler[number] = 1.0
            
            # rank_check_passed = True
            # for n,check in enumerate(fill_A):
                # if check == filler:
                    # rank_check_passed = False
                    # if fill_b[n] != p:
                        # return 'Not Solvable','Error 2'
            
            # if rank_check_passed:
                # fill_A.append(filler)
                # fill_b.append(p)
                
        if signal == 0.0 and p == 0.0:
            filler = [0.0 for _ in range(num_of_stocks)]
            filler[number] = 1.0
            
            fill_A.append(filler)
            fill_b.append(0.0)
        
    mins = reduce_finder(old_p,c1,c3)
        
    fill_G = []
    fill_h = []

    for number,long_signal,short_signal,p in zip(range(num_of_stocks),long_signals,short_signals,mins):
        if long_signal == 1.0:
            appention = [0.0 for _ in range(num_of_stocks)]
            appention[number] = -1.0
            
            fill_G.append(appention)
            fill_h.append(-p)
            
            if short_signal == 1.0:
                print('YOU HAVE DONE SOMETHING WRONG!!!! 1')
                return
                
        if short_signal == 1.0:
        
            appention = [0.0 for _ in range(num_of_stocks)]
            appention[number] = 1.0
            
            fill_G.append(appention)
            fill_h.append(p)
   
    A = matrix(fill_A).trans()
    b = matrix(fill_b)
    G = matrix(fill_G).trans()
    h = matrix(fill_h)

    solvers.options['show_progress'] = False # supress tilting output
    solvers.options['maxiters'] = max_iters
    sol = solvers.qp(P,q,G,h,A,b)

    return [round(sol['x'][i],4) for i in range(num_of_stocks)]

def signal_and_port_convertor(port,signals,is_Friday):
    
    new_port = port.copy()
    new_signals = signals.copy()
    old_position = [0 for _ in range(len(port))]

    for n,p,s in zip(range(len(port)),port,signals):
        if np.sign(p) == (-1)*s:
            new_port[n] = 0.0
            new_signals[n] = 0
        elif np.sign(p) == s:
            new_signals[n] = 0

        old_position[n] = int(np.sign(new_port[n]))
        
    return new_port,new_signals,old_position


if __name__ == '__main__':
    from risks import *
    from enviroment import *
    from utils import *
    env = Enviroment(stocks=stocks)
    risk = sharpe(env.values,66,20)
    
    risk = np.array([[ 1.21175992e-03,  4.33390741e-04,  9.14965516e-04,
         9.64478166e-04,  9.44623946e-04,  8.46045184e-04,
         5.98306882e-04,  5.47996513e-04,  5.56327338e-04,
         8.03458768e-04,  4.40837005e-04,  8.05583798e-04,
         4.01987669e-04,  7.23264148e-04,  4.39742789e-04,
         5.35978025e-04,  5.81722168e-04,  5.75560693e-04,
         8.71659973e-04,  3.29953126e-04,  4.17497320e-04,
         4.13968417e-04,  1.14872991e-03,  4.50656281e-04,
         7.03858986e-04,  4.33770892e-04,  2.11656871e-04,
         8.10297386e-04,  3.41346613e-04],
       [ 4.33390741e-04,  3.06491667e-04,  3.46942798e-04,
         3.05319375e-04,  3.48397222e-04,  3.65692366e-04,
         2.70783577e-04,  2.48169607e-04,  2.38520914e-04,
         2.70269283e-04,  2.15322676e-04,  3.28700210e-04,
         2.25029276e-04,  1.84264533e-04,  2.09785426e-04,
         1.99924219e-04,  1.63825502e-04,  2.87305583e-04,
         3.21611770e-04,  1.73733349e-04,  2.03440290e-04,
         1.95191080e-04,  3.27441182e-04,  2.07581678e-04,
         1.68232093e-04,  2.95456553e-04,  1.30248851e-04,
         1.97536260e-04,  1.96347303e-04],
       [ 9.14965516e-04,  3.46942798e-04,  9.62095244e-04,
         8.54178859e-04,  8.80095591e-04,  7.40406643e-04,
         5.11220042e-04,  2.73562201e-04,  5.17790433e-04,
         7.68172632e-04,  4.31476533e-04,  6.55931516e-04,
         3.94880672e-04,  5.21829633e-04,  3.50434331e-04,
         4.40586836e-04,  4.26802859e-04,  5.00538651e-04,
         7.90550517e-04,  2.76595079e-04,  3.27478340e-04,
         4.54553207e-04,  1.05144849e-03,  3.10832817e-04,
         4.68684072e-04,  3.91983340e-04,  2.13982099e-04,
         7.36588916e-04,  3.09122161e-04],
       [ 9.64478166e-04,  3.05319375e-04,  8.54178859e-04,
         9.71737634e-04,  8.55049804e-04,  6.96488499e-04,
         4.98776009e-04,  4.20076197e-04,  5.84609495e-04,
         7.92350759e-04,  4.28965045e-04,  6.97777312e-04,
         3.62884145e-04,  6.35181216e-04,  3.81363904e-04,
         4.86623572e-04,  4.80658416e-04,  4.61449191e-04,
         8.25625820e-04,  3.03236923e-04,  2.85816537e-04,
         4.19420804e-04,  1.13468459e-03,  3.59639166e-04,
         5.66467926e-04,  4.03662674e-04,  1.79751810e-04,
         8.12030114e-04,  3.36472548e-04],
       [ 9.44623946e-04,  3.48397222e-04,  8.80095591e-04,
         8.55049804e-04,  9.45950689e-04,  8.00657000e-04,
         5.30563932e-04,  3.92162111e-04,  5.39567306e-04,
         7.73951493e-04,  4.39915411e-04,  6.89611896e-04,
         3.70997454e-04,  6.05800074e-04,  3.53055977e-04,
         5.01582683e-04,  4.11697110e-04,  4.96871245e-04,
         7.77344391e-04,  3.09719798e-04,  4.06272302e-04,
         4.55879476e-04,  1.04779320e-03,  3.53476291e-04,
         4.66589754e-04,  4.30395766e-04,  1.88646125e-04,
         7.28229103e-04,  3.34303540e-04],
       [ 8.46045184e-04,  3.65692366e-04,  7.40406643e-04,
         6.96488499e-04,  8.00657000e-04,  7.73393849e-04,
         4.61893061e-04,  4.08736438e-04,  4.77928931e-04,
         6.29759750e-04,  3.91901708e-04,  6.45394903e-04,
         3.60012709e-04,  5.08808492e-04,  3.42337942e-04,
         4.38710512e-04,  3.82491845e-04,  4.61219930e-04,
         6.84119622e-04,  2.99611055e-04,  3.83818787e-04,
         4.04139950e-04,  8.85397706e-04,  3.17428458e-04,
         4.05103046e-04,  4.25037662e-04,  1.93850108e-04,
         5.05171760e-04,  3.16128871e-04],
       [ 5.98306882e-04,  2.70783577e-04,  5.11220042e-04,
         4.98776009e-04,  5.30563932e-04,  4.61893061e-04,
         4.00633638e-04,  2.79809839e-04,  2.94724781e-04,
         4.42739436e-04,  2.70900335e-04,  4.19859321e-04,
         2.77100035e-04,  3.87783409e-04,  2.15674859e-04,
         3.00585637e-04,  2.83892625e-04,  3.45168476e-04,
         4.44837931e-04,  2.03935660e-04,  2.60129636e-04,
         2.76109492e-04,  5.54774430e-04,  2.88535107e-04,
         2.99374446e-04,  3.21027085e-04,  1.38844069e-04,
         4.32531454e-04,  2.41351610e-04],
       [ 5.47996513e-04,  2.48169607e-04,  2.73562201e-04,
         4.20076197e-04,  3.92162111e-04,  4.08736438e-04,
         2.79809839e-04,  7.46206820e-04,  3.57362407e-04,
         3.41281675e-04,  2.77814526e-04,  5.07400351e-04,
         2.03794590e-04,  4.68316557e-04,  2.21587677e-04,
         3.41975419e-04,  3.81167154e-04,  2.38700036e-04,
         3.81649284e-04,  1.85496501e-04,  2.65104191e-04,
         1.65501853e-04,  4.70559006e-04,  2.83277342e-04,
         3.91758487e-04,  3.69421481e-04,  3.74717129e-05,
         2.78143560e-04,  2.45359416e-04],
       [ 5.56327338e-04,  2.38520914e-04,  5.17790433e-04,
         5.84609495e-04,  5.39567306e-04,  4.77928931e-04,
         2.94724781e-04,  3.57362407e-04,  4.82777902e-04,
         4.67235883e-04,  2.93783704e-04,  4.53846734e-04,
         2.40678446e-04,  3.56931834e-04,  2.67621845e-04,
         3.22344220e-04,  3.05738205e-04,  2.82057815e-04,
         5.42559729e-04,  2.21339516e-04,  1.82591037e-04,
         2.94217993e-04,  7.19245279e-04,  2.02941352e-04,
         2.54588816e-04,  3.24735408e-04,  1.55379486e-04,
         4.62354148e-04,  2.59961654e-04],
       [ 8.03458768e-04,  2.70269283e-04,  7.68172632e-04,
         7.92350759e-04,  7.73951493e-04,  6.29759750e-04,
         4.42739436e-04,  3.41281675e-04,  4.67235883e-04,
         7.28998020e-04,  3.83657814e-04,  6.20014991e-04,
         3.39263669e-04,  5.48409616e-04,  3.14615473e-04,
         4.37754291e-04,  3.83989244e-04,  4.21577007e-04,
         6.68167527e-04,  2.63536563e-04,  3.16770465e-04,
         3.99245392e-04,  9.24159129e-04,  3.30379665e-04,
         3.63197719e-04,  3.85519564e-04,  1.52966661e-04,
         6.87711683e-04,  2.82582353e-04],
       [ 4.40837005e-04,  2.15322676e-04,  4.31476533e-04,
         4.28965045e-04,  4.39915411e-04,  3.91901708e-04,
         2.70900335e-04,  2.77814526e-04,  2.93783704e-04,
         3.83657814e-04,  2.99714497e-04,  3.59608580e-04,
         2.17481424e-04,  2.79481189e-04,  1.86849637e-04,
         2.67563702e-04,  2.10084378e-04,  2.74134353e-04,
         3.88187819e-04,  1.79883182e-04,  1.88687172e-04,
         2.37483548e-04,  5.15909683e-04,  1.78130926e-04,
         1.69718302e-04,  2.70833996e-04,  1.34087367e-04,
         2.98687438e-04,  1.96031200e-04],
       [ 8.05583798e-04,  3.28700210e-04,  6.55931516e-04,
         6.97777312e-04,  6.89611896e-04,  6.45394903e-04,
         4.19859321e-04,  5.07400351e-04,  4.53846734e-04,
         6.20014991e-04,  3.59608580e-04,  7.17288704e-04,
         3.62227562e-04,  5.27948636e-04,  3.26680308e-04,
         4.28342978e-04,  4.62990297e-04,  4.43166839e-04,
         6.33035961e-04,  2.64889980e-04,  3.50070041e-04,
         3.65498297e-04,  7.93718638e-04,  3.39963230e-04,
         5.29984281e-04,  3.99508183e-04,  1.17778596e-04,
         5.32248919e-04,  2.66317692e-04],
       [ 4.01987669e-04,  2.25029276e-04,  3.94880672e-04,
         3.62884145e-04,  3.70997454e-04,  3.60012709e-04,
         2.77100035e-04,  2.03794590e-04,  2.40678446e-04,
         3.39263669e-04,  2.17481424e-04,  3.62227562e-04,
         2.91571479e-04,  2.71025500e-04,  1.88737968e-04,
         2.27281568e-04,  2.52379492e-04,  2.75369772e-04,
         3.35042492e-04,  1.80682359e-04,  2.09530960e-04,
         2.48441187e-04,  4.10225763e-04,  1.89654273e-04,
         2.27447899e-04,  2.81160098e-04,  1.41586528e-04,
         2.54278730e-04,  2.08940112e-04],
       [ 7.23264148e-04,  1.84264533e-04,  5.21829633e-04,
         6.35181216e-04,  6.05800074e-04,  5.08808492e-04,
         3.87783409e-04,  4.68316557e-04,  3.56931834e-04,
         5.48409616e-04,  2.79481189e-04,  5.27948636e-04,
         2.71025500e-04,  6.34650087e-04,  2.50932409e-04,
         4.10454446e-04,  4.23160710e-04,  3.09174825e-04,
         5.07719449e-04,  2.29859938e-04,  3.16898173e-04,
         2.72388667e-04,  7.63815513e-04,  3.10993047e-04,
         4.81673065e-04,  3.14342292e-04,  9.86093343e-05,
         5.82890985e-04,  2.65258327e-04],
       [ 4.39742789e-04,  2.09785426e-04,  3.50434331e-04,
         3.81363904e-04,  3.53055977e-04,  3.42337942e-04,
         2.15674859e-04,  2.21587677e-04,  2.67621845e-04,
         3.14615473e-04,  1.86849637e-04,  3.26680308e-04,
         1.88737968e-04,  2.50932409e-04,  2.36739298e-04,
         2.24560657e-04,  2.07240266e-04,  2.60239787e-04,
         3.35398786e-04,  1.63812027e-04,  1.92511079e-04,
         2.01634825e-04,  4.50301618e-04,  1.74298874e-04,
         1.96081047e-04,  2.31615881e-04,  1.20044817e-04,
         2.77645622e-04,  1.87747115e-04],
       [ 5.35978025e-04,  1.99924219e-04,  4.40586836e-04,
         4.86623572e-04,  5.01582683e-04,  4.38710512e-04,
         3.00585637e-04,  3.41975419e-04,  3.22344220e-04,
         4.37754291e-04,  2.67563702e-04,  4.28342978e-04,
         2.27281568e-04,  4.10454446e-04,  2.24560657e-04,
         3.34739268e-04,  2.87034171e-04,  2.97897161e-04,
         4.36184658e-04,  2.04711299e-04,  2.59750285e-04,
         2.64508621e-04,  5.97436694e-04,  2.28721716e-04,
         2.40664734e-04,  2.67110797e-04,  1.17947833e-04,
         4.11334796e-04,  2.11018377e-04],
       [ 5.81722168e-04,  1.63825502e-04,  4.26802859e-04,
         4.80658416e-04,  4.11697110e-04,  3.82491845e-04,
         2.83892625e-04,  3.81167154e-04,  3.05738205e-04,
         3.83989244e-04,  2.10084378e-04,  4.62990297e-04,
         2.52379492e-04,  4.23160710e-04,  2.07240266e-04,
         2.87034171e-04,  4.99449891e-04,  3.06595528e-04,
         4.18490190e-04,  1.58118124e-04,  2.12354348e-04,
         2.33402238e-04,  5.91402793e-04,  2.04535763e-04,
         4.48099552e-04,  1.73064303e-04,  1.33946774e-04,
         3.99348789e-04,  1.62373273e-04],
       [ 5.75560693e-04,  2.87305583e-04,  5.00538651e-04,
         4.61449191e-04,  4.96871245e-04,  4.61219930e-04,
         3.45168476e-04,  2.38700036e-04,  2.82057815e-04,
         4.21577007e-04,  2.74134353e-04,  4.43166839e-04,
         2.75369772e-04,  3.09174825e-04,  2.60239787e-04,
         2.97897161e-04,  3.06595528e-04,  4.35596799e-04,
         3.93936215e-04,  2.07843561e-04,  3.04299335e-04,
         2.99791731e-04,  5.21532452e-04,  2.53167007e-04,
         2.14026120e-04,  2.85386804e-04,  1.79791104e-04,
         3.70338627e-04,  2.02653515e-04],
       [ 8.71659973e-04,  3.21611770e-04,  7.90550517e-04,
         8.25625820e-04,  7.77344391e-04,  6.84119622e-04,
         4.44837931e-04,  3.81649284e-04,  5.42559729e-04,
         6.68167527e-04,  3.88187819e-04,  6.33035961e-04,
         3.35042492e-04,  5.07719449e-04,  3.35398786e-04,
         4.36184658e-04,  4.18490190e-04,  3.93936215e-04,
         9.16510583e-04,  2.72875861e-04,  1.99292055e-04,
         3.81205980e-04,  1.05029680e-03,  2.83843409e-04,
         4.81661060e-04,  3.01131071e-04,  1.64776645e-04,
         6.58499182e-04,  2.67828982e-04],
       [ 3.29953126e-04,  1.73733349e-04,  2.76595079e-04,
         3.03236923e-04,  3.09719798e-04,  2.99611055e-04,
         2.03935660e-04,  1.85496501e-04,  2.21339516e-04,
         2.63536563e-04,  1.79883182e-04,  2.64889980e-04,
         1.80682359e-04,  2.29859938e-04,  1.63812027e-04,
         2.04711299e-04,  1.58118124e-04,  2.07843561e-04,
         2.72875861e-04,  1.73449458e-04,  1.65299435e-04,
         1.95033654e-04,  3.67111840e-04,  1.52682591e-04,
         1.28446671e-04,  2.09062955e-04,  1.11909883e-04,
         2.23261965e-04,  1.74540789e-04],
       [ 4.17497320e-04,  2.03440290e-04,  3.27478340e-04,
         2.85816537e-04,  4.06272302e-04,  3.83818787e-04,
         2.60129636e-04,  2.65104191e-04,  1.82591037e-04,
         3.16770465e-04,  1.88687172e-04,  3.50070041e-04,
         2.09530960e-04,  3.16898173e-04,  1.92511079e-04,
         2.59750285e-04,  2.12354348e-04,  3.04299335e-04,
         1.99292055e-04,  1.65299435e-04,  3.47446313e-04,
         2.27484736e-04,  3.28072124e-04,  2.13085197e-04,
         1.79898114e-04,  2.72775475e-04,  9.70605260e-05,
         2.41042647e-04,  1.91047468e-04],
       [ 4.13968417e-04,  1.95191080e-04,  4.54553207e-04,
         4.19420804e-04,  4.55879476e-04,  4.04139950e-04,
         2.76109492e-04,  1.65501853e-04,  2.94217993e-04,
         3.99245392e-04,  2.37483548e-04,  3.65498297e-04,
         2.48441187e-04,  2.72388667e-04,  2.01634825e-04,
         2.64508621e-04,  2.33402238e-04,  2.99791731e-04,
         3.81205980e-04,  1.95033654e-04,  2.27484736e-04,
         3.15199909e-04,  5.08070467e-04,  1.84731586e-04,
         1.62439242e-04,  2.35837220e-04,  1.34624455e-04,
         3.27818006e-04,  2.06111230e-04],
       [ 1.14872991e-03,  3.27441182e-04,  1.05144849e-03,
         1.13468459e-03,  1.04779320e-03,  8.85397706e-04,
         5.54774430e-04,  4.70559006e-04,  7.19245279e-04,
         9.24159129e-04,  5.15909683e-04,  7.93718638e-04,
         4.10225763e-04,  7.63815513e-04,  4.50301618e-04,
         5.97436694e-04,  5.91402793e-04,  5.21532452e-04,
         1.05029680e-03,  3.67111840e-04,  3.28072124e-04,
         5.08070467e-04,  1.46264635e-03,  3.46173926e-04,
         6.24623241e-04,  3.99930748e-04,  2.60680956e-04,
         9.45065111e-04,  3.82021177e-04],
       [ 4.50656281e-04,  2.07581678e-04,  3.10832817e-04,
         3.59639166e-04,  3.53476291e-04,  3.17428458e-04,
         2.88535107e-04,  2.83277342e-04,  2.02941352e-04,
         3.30379665e-04,  1.78130926e-04,  3.39963230e-04,
         1.89654273e-04,  3.10993047e-04,  1.74298874e-04,
         2.28721716e-04,  2.04535763e-04,  2.53167007e-04,
         2.83843409e-04,  1.52682591e-04,  2.13085197e-04,
         1.84731586e-04,  3.46173926e-04,  2.70264097e-04,
         1.99598193e-04,  2.75746481e-04,  6.07331470e-05,
         3.15080166e-04,  1.79414205e-04],
       [ 7.03858986e-04,  1.68232093e-04,  4.68684072e-04,
         5.66467926e-04,  4.66589754e-04,  4.05103046e-04,
         2.99374446e-04,  3.91758487e-04,  2.54588816e-04,
         3.63197719e-04,  1.69718302e-04,  5.29984281e-04,
         2.27447899e-04,  4.81673065e-04,  1.96081047e-04,
         2.40664734e-04,  4.48099552e-04,  2.14026120e-04,
         4.81661060e-04,  1.28446671e-04,  1.79898114e-04,
         1.62439242e-04,  6.24623241e-04,  1.99598193e-04,
         1.12258038e-03,  1.17158643e-04, -5.80827355e-05,
         4.11333066e-04,  1.79615375e-04],
       [ 4.33770892e-04,  2.95456553e-04,  3.91983340e-04,
         4.03662674e-04,  4.30395766e-04,  4.25037662e-04,
         3.21027085e-04,  3.69421481e-04,  3.24735408e-04,
         3.85519564e-04,  2.70833996e-04,  3.99508183e-04,
         2.81160098e-04,  3.14342292e-04,  2.31615881e-04,
         2.67110797e-04,  1.73064303e-04,  2.85386804e-04,
         3.01131071e-04,  2.09062955e-04,  2.72775475e-04,
         2.35837220e-04,  3.99930748e-04,  2.75746481e-04,
         1.17158643e-04,  5.01509005e-04,  1.26889897e-04,
         2.84217425e-04,  3.01682663e-04],
       [ 2.11656871e-04,  1.30248851e-04,  2.13982099e-04,
         1.79751810e-04,  1.88646125e-04,  1.93850108e-04,
         1.38844069e-04,  3.74717129e-05,  1.55379486e-04,
         1.52966661e-04,  1.34087367e-04,  1.17778596e-04,
         1.41586528e-04,  9.86093343e-05,  1.20044817e-04,
         1.17947833e-04,  1.33946774e-04,  1.79791104e-04,
         1.64776645e-04,  1.11909883e-04,  9.70605260e-05,
         1.34624455e-04,  2.60680956e-04,  6.07331470e-05,
        -5.80827355e-05,  1.26889897e-04,  1.88990427e-04,
         1.30239180e-04,  1.10228198e-04],
       [ 8.10297386e-04,  1.97536260e-04,  7.36588916e-04,
         8.12030114e-04,  7.28229103e-04,  5.05171760e-04,
         4.32531454e-04,  2.78143560e-04,  4.62354148e-04,
         6.87711683e-04,  2.98687438e-04,  5.32248919e-04,
         2.54278730e-04,  5.82890985e-04,  2.77645622e-04,
         4.11334796e-04,  3.99348789e-04,  3.70338627e-04,
         6.58499182e-04,  2.23261965e-04,  2.41042647e-04,
         3.27818006e-04,  9.45065111e-04,  3.15080166e-04,
         4.11333066e-04,  2.84217425e-04,  1.30239180e-04,
         8.84104911e-04,  2.31676278e-04],
       [ 3.41346613e-04,  1.96347303e-04,  3.09122161e-04,
         3.36472548e-04,  3.34303540e-04,  3.16128871e-04,
         2.41351610e-04,  2.45359416e-04,  2.59961654e-04,
         2.82582353e-04,  1.96031200e-04,  2.66317692e-04,
         2.08940112e-04,  2.65258327e-04,  1.87747115e-04,
         2.11018377e-04,  1.62373273e-04,  2.02653515e-04,
         2.67828982e-04,  1.74540789e-04,  1.91047468e-04,
         2.06111230e-04,  3.82021177e-04,  1.79414205e-04,
         1.79615375e-04,  3.01682663e-04,  1.10228198e-04,
         2.31676278e-04,  2.53349257e-04]])
    
    old_p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.04432643045004631, -0.0695574866661632, 0.0, 0.0, 0.0, -0.03187899569448171, 0.0, 0.07181671353291447, 0.0, 0.0, -0.0036452959329534595, 0.0, 0.0, 0.0, 0.0, 0.10136539404342554, 0.0, 0.0, 0.0, 0.0, 0.0]
    signals =  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, -1, 0, 1, 0, 0, 0, 0]
    c1,c3 = (0.27412232756614685,-0.14457650482654572)
    
    test = quadratic(risk,old_p,signals,c1,c3)
    
    