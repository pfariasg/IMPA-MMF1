import datetime
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import sys

contract_type = 'call'
S0 = 9   # stock price at t=0
K = 10   # strike
N = 3    # maturity (years)
r = 0.06 # annual risk-free rate
T = 15   # number of steps
std = .3 # annual standard deviation

delta_t = N/T

# up, down and risk-neutral probabilities
u = np.exp( std*np.sqrt(delta_t)+(r-.5*pow(std, 2))*delta_t)
d = np.exp(-std*np.sqrt(delta_t)+(r-.5*pow(std, 2))*delta_t)

p_tilde = (np.exp(r*delta_t)-d)/(u-d)
q_tilde = 1 - p_tilde

# default parameters to plot
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#262626'
# plt.rcParams['axes.grid'] = True
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['axes.titlesize'] = '18'
# plt.rcParams['axes.xmargin'] = 0
plt.rcParams['figure.figsize'] = [30, 14]
plt.rcParams['figure.facecolor'] = '#262626'
plt.rcParams['figure.edgecolor'] = 'white'
plt.rcParams['grid.alpha'] = .2
plt.rcParams['grid.color'] = 'white'
plt.rcParams['patch.edgecolor'] = 'white'
plt.rcParams['text.color'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

def random_variable(x, i):
    '''
    if H, multiply x by u
    it T, multiply x by d 
    '''
    if x['index'][i] == 'H':
        return x.get(i)*u
    return x.get(i)*d

def payoff(K, X, contract_type):
    '''
    returns a payoff given
    the underlying price, 
    strike and contract type
    '''
    if contract_type == 'put':
        return max(K - X, 0)
    elif contract_type == 'call':
        return max(X - K, 0)
    elif contract_type == 'future':
        return X - K
    else:
        sys.exit(f'invalid contract type: {contract_type}')

# initialize a path with Heads only and
# incrementally switch the end for Tails
# until there are only tails

# w = {H, T}
header = []
path = ['H' for w in range(T)]
i = len(path)-1

while i >= -1:
    header.append(path.copy())
    path[i] = 'T'
    i -= 1

header = tuple(header)

price_tree = pd.DataFrame(columns=header)

S0 = [S0 for _ in price_tree.columns]
price_tree.loc[0, :] = S0

# make a price tree from t=0 until t=T, containing each path
for i in range(T):
    last_line = price_tree.iloc[-1, :]

    new_line = pd.Series(last_line.reset_index().apply(lambda x: random_variable(x, i), axis=1).values, index=last_line.index)
    price_tree = price_tree.append(new_line, ignore_index=True)

print('\ninicia um dataframe com os caminhos e preços relevantes:')
print(price_tree)

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

print('\nárvore binomial completa do preço da opção:')
print(payoff_tree)
payoff_tree.to_clipboard()

# plots the results
stock_values_per_lvl = []
for _, row in price_tree.iterrows():
    row = row.dropna().to_list()
    row = list(dict.fromkeys(row))
    stock_values_per_lvl.append(row)

option_values_per_lvl = []
for _, row in payoff_tree.iloc[::-1].iterrows():
    option_values_per_lvl.append(row.dropna().to_list())

plt.rcParams['figure.figsize'] = [16, 9]
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, sharey='none')
fig.tight_layout(rect=(0.02,0.02,0.98,0.98))
fig.suptitle(f'Binomial Tree - {contract_type.upper()[0]+contract_type[1:]}', size=20, y=0.95)

ax1.axhline(K, color='white', lw=1.5, ls='-')
ax2.axhline(K, color='white', lw=1.5, ls='-')

color_map = cm.get_cmap('viridis')
x = [1]
for i in range(T):
    x.append(0)
    x.append(1)
    plot_x = (np.array(x) + i)*delta_t

    stitch_stocks = []
    for k, pos in enumerate(stock_values_per_lvl[i+1]):
        stitch_stocks.append(pos)
        if k == len(stock_values_per_lvl[i]):
            break
        stitch_stocks.append(stock_values_per_lvl[i][k])

    stitch_options = []
    for k, pos in enumerate(option_values_per_lvl[i+1]):
        stitch_options.append(pos)
        if k == len(option_values_per_lvl[i]):
            break
        stitch_options.append(option_values_per_lvl[i][k])

    stitch_options = [round(price,4) for price in stitch_options]

    ax1.plot(plot_x, stitch_stocks, 'bo-', color=color_map(i/(T-1)))
    ax2.plot(plot_x, stitch_stocks, 'bo-', color=color_map(i/(T-1)))

    for d, l, r in zip(plot_x, stitch_stocks, stitch_options):
        for ax in (ax1, ax2):
            ax.annotate(r, xy=(d, l),
                xytext=(-100, -80), textcoords='offset pixels')

ax1.set_ylabel('S - Standard Scale')
ax2.set_ylabel('S - Log Scale')
ax2.set_xlabel('Time')

ax2.set_yscale('log')
plt.show()
