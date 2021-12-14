# this script replicates the Longstaff-Schwartz method of option pricing 
# described at this paper: 
# https://people.math.ethz.ch/~hjfurrer/teaching/LongstaffSchwartzAmericanOptionsLeastSquareMonteCarlo.pdf

import numpy as np
import pandas as pd
import sys
import warnings
warnings.simplefilter('ignore', np.RankWarning)

##########################################################################
# inputs

contract_type = 'put'
K = 1.10   # strike
r = .06    # interest rate per period

input_dict = {
#   t: [path 1, path 2, path 3,   ..., path 7,  path 8],    
    0: [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    1: [1.09, 1.16, 1.22, 0.93, 1.11, 0.76, 0.92, 0.88],
    2: [1.08, 1.26, 1.07, 0.97, 1.56, 0.77, 0.84, 1.01],
    3: [1.34, 1.54, 1.03, 0.92, 1.52, 0.90, 1.01, 1.34]
}

##########################################################################

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

def exercise(row, contract_type):
    if row['payoffs_dm1'] > row['continuation']:
        return 1
    return 0

##########################################################################

# create a dataframe for stock prices from the input, 
# with each t on the column row
stock_prices = pd.DataFrame(input_dict)
stock_prices.index += 1

# extract american option exercise values at any moment
payoffs = stock_prices.applymap(lambda x: payoff(K, x, contract_type))

funcs = []

# discount rate
dis = np.exp(-r) 

for i in range(len(payoffs.columns)-1):
    # get stock prices at second last column
    df = pd.DataFrame(stock_prices.iloc[:, -(i+2)].astype(float))
    df.columns = ['stock_prices']
    
    # discount option payoffs of the last column and get payoffs from second last column
    df['payoffs'] = payoffs.iloc[:, -(i+1)] * dis
    df['payoffs_dm1'] = payoffs.iloc[:, -(i+2)]

    # select when option can be exercised with a profit
    if contract_type == 'call':
        df = df.loc[df['stock_prices'] > K]
    elif contract_type == 'put':
        df = df.loc[df['stock_prices'] < K]

    # regression between stock prices (t-1) and payoffs (t, discounted to t-1)
    y = np.poly1d(np.polyfit(df['stock_prices'], df['payoffs'], 2))
    
    # apply regression to stock prices (t-1) to calculate intrinsic value
    df['continuation'] = df['stock_prices'].apply(y)

    # check if option was exercised or not
    df['exercise'] = df.apply(lambda row: exercise(row, contract_type), axis=1)
    df = df.loc[df['exercise']==1]

    # find value of each path at t-1 until t=1
    payoffs.loc[df.index, len(payoffs.columns)-1-i] = df['payoffs_dm1']
    payoffs.loc[:, len(payoffs.columns)-2-i] = payoffs.loc[:, len(payoffs.columns)-1-i]

# now at t=1, discount the value of each path and calculate the mean to find the american option value
value = (payoffs.iloc[:, 0] * dis).mean()

print(value)