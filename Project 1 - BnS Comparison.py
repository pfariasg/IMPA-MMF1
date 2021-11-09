import datetime
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy import interpolate
import numpy as np
import pandas as pd
import sys

S0 = 9   # Stock price at t=0
K = 10   # Strike
N = 3    # Maturity (years)
r = 0.06 # Annual risk-free rate
T_options = [10 + 10*i for i in range(40)] # number of steps
std = .3

call_bns = 2.120093831410867
put_bns = 1.472795945523587

bns = {'call': call_bns, 'put': put_bns}

plt.rcParams['figure.figsize'] = [16, 9]
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.titlesize'] = '18'
plt.rcParams['axes.facecolor'] = '#262626'
plt.rcParams['figure.facecolor'] = '#262626'
plt.rcParams['figure.edgecolor'] = 'white'
plt.rcParams['patch.edgecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = 'white'
plt.rcParams['grid.alpha'] = .2

def random_variable(x, i):
    '''
    multiplica por u ou d 
    dependendo da amostra 
    '''
    if x['index'][i] == 'H':
        return x.get(i)*u
    return x.get(i)*d

def payoff(K, X, contract_type):
    if contract_type == 'put':
        return max(K - X, 0)
    elif contract_type == 'call':
        return max(X - K, 0)
    elif contract_type == 'future':
        return X - K
    else:
        sys.exit(f'invalid contract type: {contract_type}')

results = pd.DataFrame(columns=['call', 'put'])

for contract_type in ['call', 'put']:
    for T in T_options:
        print(contract_type+': '+str(T))
        delta_t = N/T

        u = np.exp( std*np.sqrt(delta_t)+(r-0.5*pow(std,2))*delta_t)
        d = np.exp(-std*np.sqrt(delta_t)+(r-0.5*pow(std,2))*delta_t)

        p_tilde = (np.exp(r*delta_t)-d)/(u-d) 
        q_tilde = 1 - p_tilde   

        # creates a path with all H, gradually
        # changes the last H for a T to reach 
        # all final values

        # w = {H, T}
        header = []
        path = ['H' for s in range(T)]

        i = len(path)-1
        while i >= -1:
            header.append(path.copy())
            path[i] = 'T'
            i -= 1

        header = tuple(header)
        price_tree = pd.DataFrame(columns=header)

        S0_k = [S0 for s in price_tree.columns]
        price_tree.loc[0, :] = S0_k

        # make a price tree from t=0 until t=T, containing each path
        for i in range(T):
            last_line = price_tree.iloc[-1, :]

            new_line = pd.Series(last_line.reset_index().apply(lambda x: random_variable(x, i), axis=1).values, index=last_line.index)
            price_tree = price_tree.append(new_line, ignore_index=True)

        # gets the last values from the tree (t=T) and calculates option payoff
        payoff_tree = price_tree.iloc[-1, :].to_frame()
        payoff_tree = payoff_tree.applymap(lambda x: payoff(K, x, contract_type))

        # from those payoffs, prices the option at t=0
        payoff_tree = pd.DataFrame(data=payoff_tree).T

        for i in range(T):
            payoff_tree.loc[T-i-1] = np.nan
            # for each pair of values, calculate the last node
            for col_pair in range(len(payoff_tree.columns)-1):
                alternatives = payoff_tree.iloc[i, col_pair:col_pair+2]
                Vn1H = alternatives.iloc[0]
                Vn1T = alternatives.iloc[1]

                Vn = np.exp(-r*delta_t)*(p_tilde*Vn1H + q_tilde*Vn1T)

                payoff_tree.iloc[i+1, col_pair+1] = Vn

        payoff_tree.columns = pd.MultiIndex.from_tuples(payoff_tree.columns)

        option_price = payoff_tree.iloc[-1, :].dropna().iloc[0]
        results.loc[T, contract_type] = option_price

    results[contract_type] = results[contract_type]


fig = plt.figure() # (call, put)
gs = fig.add_gridspec(2, 3)

ax1 = fig.add_subplot(gs[0, 0:2])
ax2 = fig.add_subplot(gs[1, 0:2])

ax1.get_shared_x_axes().join(ax1, ax2)

f = interpolate.interp1d(results.index, results['call'], kind='quadratic')
x = np.linspace(results.index.min(), results.index.max(), num=5000)

call_error = abs(f(x)-bns['call'])
ax1.scatter(x, f(x), s=2, c=cm.autumn(call_error/call_error.max()), label='Call price (Binomial)')

ax1.set_title('Call')

f = interpolate.interp1d(results.index, results['put'], kind='quadratic')
x = np.linspace(results.index.min(), results.index.max(), num=5000)

put_error = abs(f(x)-bns['put'])
ax2.scatter(x, f(x), s=2, c=cm.autumn(put_error/put_error.max()), label='Put price (Binomial)')

ax2.set_title('Put')

ax1.axhline(bns['call'], color='#2AC408', lw=2, ls='-', label='Call price (BnS)')
ax2.axhline(bns['put'], color='#2AC408', lw=2, ls='-', label='Put price (BnS)')

ax1.legend(loc='right')
ax1.set_ylabel('Call price')
ax2.legend(loc='right')
ax2.set_ylabel('Put Price')
ax2.set_xlabel('Number of Steps')

ax3 = fig.add_subplot(gs[0, 2])

call_error = pd.DataFrame(call_error)
call_error.index = call_error.index*results.index.max()/5000

put_error = pd.DataFrame(put_error)
put_error.index = put_error.index*results.index.max()/5000

ax3.scatter(call_error.index, call_error.iloc[:, 0], s=2, c=cm.autumn(call_error.iloc[:, 0]/call_error.iloc[:, 0].max()))
ax3.set_title('Call Error')
ax3.set_ylabel('Call Error')

ax4 = fig.add_subplot(gs[1, 2])
ax4.scatter(put_error.index, put_error.iloc[:, 0], s=2, c=cm.autumn(put_error.iloc[:, 0]/put_error.iloc[:, 0].max()))
ax4.set_title('Put Error')
ax4.set_xlabel('Number of Steps')
ax4.set_ylabel('Put Error')

plt.show()

call_error['call_constant'] = 1/call_error.index 
put_error['put_constant'] = 1/put_error.index 

call_error = call_error.iloc[100:]
put_error = put_error.iloc[100:]

fig = plt.figure() # (call, put)
gs = fig.add_gridspec(2, 2)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

ax1.get_shared_x_axes().join(ax1, ax3)
ax2.get_shared_x_axes().join(ax2, ax4)

ax1.scatter(call_error.index, call_error.iloc[:, 0], s=2, c=cm.autumn(call_error.iloc[:, 0]/call_error.iloc[:, 0].max()), label='Call Error')
ax1.set_title('Call Error')
ax1.set_ylabel('Call Error')

ax3.scatter(put_error.index, put_error.iloc[:, 0], s=2, c=cm.autumn(put_error.iloc[:, 0]/put_error.iloc[:, 0].max()), label='Put Error')
ax3.scatter(put_error.index, put_error.iloc[:, 0], s=2, c=cm.autumn(put_error.iloc[:, 0]/put_error.iloc[:, 0].max()))
ax3.set_title('Put Error')
ax3.set_xlabel('Number of Steps')
ax3.set_ylabel('Put Error')

ax2.scatter(call_error.index, call_error.iloc[:, 0], s=2, c=cm.autumn(call_error.iloc[:, 0]/call_error.iloc[:, 0].max()), label='Call Error')
ax2.scatter(call_error.index, call_error['call_constant'], s=2, c=cm.winter(call_error.iloc[:, 0]/call_error.iloc[:, 0].max()), label='1 / # of Steps')
ax2.set_title('Call Error vs. 1 / # of Steps')
ax2.set_ylabel('Call Error')

ax4.scatter(put_error.index, put_error.iloc[:, 0], s=2, c=cm.autumn(put_error.iloc[:, 0]/put_error.iloc[:, 0].max()), label='Put Error')
ax4.scatter(put_error.index, put_error['put_constant'], s=2, c=cm.winter(put_error.iloc[:, 0]/put_error.iloc[:, 0].max()), label='1 / # of Steps')
ax4.set_title('Put Error vs. 1 / # of Steps')
ax4.set_xlabel('Number of Steps')
ax4.set_ylabel('Put Error')

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
plt.show()