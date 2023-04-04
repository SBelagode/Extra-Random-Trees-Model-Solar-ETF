#!/usr/bin/env python
# coding: utf-8

# Paper: https://www.sciencedirect.com/science/article/pii/S1062940822000572
# This project is based off the above research paper.
# 
# Paper Main Idea: Using decision trees to forecast upward direction in (TAN) solar stock ETF that tracks major 
# solar companies. The prediction time horizon they tested was in 1,2,3,…,20 days. The best prediction was for 15 days. The predictor space includes relative strength indicator (RSI), moving average cross-over divergence (MACD), price rate of change (ROC), on balance volume (OBV), the 50-day moving average, the 200-day moving average, and Williams accumulation and distribution (WAD). The three additional features are oil price volatility, silver prices, and silver price volatility. 
# 
# Intuition: There have been previous studies on the effectiveness of these technical indicators. Solar energy is also one of the hottest growing electricity generation sources. As countries are looking for renewable energy sources, wind has become one of the most widely used forms. However, solar power appears to be poised on the path to overtake wind, at least based on the superior growth rate of solar power. Furthermore, there has been extensive research on the relationship between movements in oil prices/technology stock prices and clean energy stock prices. However, there is a lot of skepticism over predicting prices, so this paper uses a classification system instead. It is also important to note that silver is used in the manufacturing of solar panels. As a result, silver price volatility may affect the cost of solar panels, potentially creating more chain reactions from there. The solar industry’s silver demand accounted for 10% approximately of total silver demand in 2019. 
# 
# My testing: I essentially extended their logic by including additional predictors such as the semiconductor ETF SOXX, as well as obtaining volatility values for everyday based on the Yang-Zhang formula. 
# 
# Furthermore, I optimized over various forecasting periods and configurations of an Extra Random Trees model to obtain a robust model to trade off of. I then tested this model on the out of sample period. 
# 

# In[33]:


# importing various models and loading the data I pulled from Bloomberg Terminal to create a data set using the prices 
# and other features such as open, close, volatility, high, low for each of the ETF's. 

import numpy as np

import pandas as pd
import yfinance as yf
import datetime


import yfinance as yf




# Import the plotting library
import matplotlib.pyplot as plt
  


df_oil_volatility = pd.read_csv("/Users/sohumbelagode/Desktop/Solar Forecasting/oil_full.csv")
df_oil_volatility = df_oil_volatility[::-1]
df_oil_volatility = df_oil_volatility.reset_index(drop = True)
df_oil_volatility = df_oil_volatility.set_index('Date', drop = False)


df_slv_open_close = pd.read_csv("/Users/sohumbelagode/Desktop/Solar Forecasting/slv_open_close.csv")
df_slv_open_close = df_slv_open_close[7:]
df_slv_open_close.columns = ['Date', 'Open', 'Close']
df_slv_open_close = df_slv_open_close[::-1]
df_slv_open_close = df_slv_open_close.reset_index(drop = True)
df_slv_open_close = df_slv_open_close.set_index('Date', drop = False)
df_slv_open_close = df_slv_open_close.dropna()




df_slv_high_low = pd.read_csv("/Users/sohumbelagode/Desktop/Solar Forecasting/slv_high_low.csv")
df_slv_high_low = df_slv_high_low[7:]
df_slv_high_low.columns = ['Date', 'High', 'Low']
df_slv_high_low = df_slv_high_low[::-1]
df_slv_high_low = df_slv_high_low.reset_index(drop = True)
df_slv_high_low = df_slv_high_low.set_index('Date', drop = False)
df_slv_high_low = df_slv_high_low.dropna()




df_tan_price_volume = pd.read_csv("/Users/sohumbelagode/Desktop/Solar Forecasting/tan_full.csv")

df_tan_price_volume.columns = ['Date', 'Price', 'Volume']
df_tan_price_volume = df_tan_price_volume[::-1]
df_tan_price_volume = df_tan_price_volume.reset_index(drop = True)
df_tan_price_volume = df_tan_price_volume.set_index('Date', drop = False)
df_tan_price_volume = df_tan_price_volume.dropna()




df_tan_price_open_close = pd.read_csv("/Users/sohumbelagode/Desktop/Solar Forecasting/tan_open_close.csv")
df_tan_price_open_close = df_tan_price_open_close[7:]
df_tan_price_open_close.columns = ['Date', 'Open', 'Close']
df_tan_price_open_close = df_tan_price_open_close[::-1]
df_tan_price_open_close = df_tan_price_open_close.reset_index(drop = True)
df_tan_price_open_close = df_tan_price_open_close.set_index('Date', drop = False)
df_tan_price_open_close = df_tan_price_open_close.dropna()


df_tan_price_high_low = pd.read_csv("/Users/sohumbelagode/Desktop/Solar Forecasting/tan_high_low.csv")
df_tan_price_high_low = df_tan_price_high_low[7:]
df_tan_price_high_low.columns = ['Date', 'High', 'Low']
df_tan_price_high_low = df_tan_price_high_low[::-1]
df_tan_price_high_low = df_tan_price_high_low.reset_index(drop = True)
df_tan_price_high_low = df_tan_price_high_low.set_index('Date', drop = False)
df_tan_price_high_low = df_tan_price_high_low.dropna()





df_soxx_open_close = pd.read_csv("/Users/sohumbelagode/Desktop/Solar Forecasting/soxx_open_close.csv")
df_soxx_open_close = df_soxx_open_close[7:]
df_soxx_open_close.columns = ['Date', 'Open', 'Close']
df_soxx_open_close = df_soxx_open_close[::-1]
df_soxx_open_close = df_soxx_open_close.reset_index(drop = True)
df_soxx_open_close = df_soxx_open_close.set_index('Date', drop = False)
df_soxx_open_close = df_soxx_open_close.dropna()




df_soxx_high_low = pd.read_csv("/Users/sohumbelagode/Desktop/Solar Forecasting/soxx_high_low.csv")
df_soxx_high_low = df_soxx_high_low[7:]
df_soxx_high_low.columns = ['Date', 'High', 'Low']
df_soxx_high_low = df_soxx_high_low[::-1]
df_soxx_high_low = df_soxx_high_low.reset_index(drop = True)
df_soxx_high_low = df_soxx_high_low.set_index('Date', drop = False)
df_soxx_high_low = df_soxx_high_low.dropna()











#print(df_soxx_volatility.head())
#print(df_soxx_volatility.loc[0, 'OVX_Vol'])
'''
print(df_tan_price.loc['2/25/11': '3/3/11', 'Tan_price'])
#print(df_tan_price['Tan_price'][0:5])
print(df_tan_price.head())
print(df_soxx_volatility.head())
#print(df_tan_price.index)

'''


tan_price = []
soxx_price = []
silver_price = []
oil_vol = []

tan_open = []
soxx_open = []
silver_open = []

soxx_high = []
silver_high = []
tan_high = []

soxx_low = []
silver_low = []
tan_low = []

tan_volume = []


date = []

for i in df_tan_price_open_close.index:
    try: 
        date.append(i)
        tan_price.append(float(df_tan_price_open_close.loc[i, 'Close']))
        soxx_price.append(float(df_soxx_open_close.loc[i,'Close']))
        silver_price.append(float(df_slv_open_close.loc[i, 'Close']))
        oil_vol.append(float(df_oil_volatility.loc[i, 'OVX_Vol' ]))
        
        tan_open.append(float(df_tan_price_open_close.loc[i, 'Open']))
        soxx_open.append(float(df_soxx_open_close.loc[i,'Open']))
        silver_open.append(float(df_slv_open_close.loc[i, 'Open']))
        
        tan_high.append(float(df_tan_price_high_low.loc[i, 'High']))
        soxx_high.append(float(df_soxx_high_low.loc[i,'High']))
        silver_high.append(float(df_slv_high_low.loc[i, 'High']))
        
        tan_low.append(float(df_tan_price_high_low.loc[i, 'Low']))
        soxx_low.append(float(df_soxx_high_low.loc[i,'Low']))
        silver_low.append(float(df_slv_high_low.loc[i, 'Low']))
        
        tan_volume.append(float(df_tan_price_volume.loc[i, 'Volume']))
        
        
    except: 
     
        pass
        

        
#making sure my columns are all the same length
print(len(date))
print(len(tan_price))
print(len(soxx_price))
print(len(silver_price))
print(len(oil_vol))

print(len(tan_open))
print(len(soxx_open))
print(len(silver_open))

print(len(tan_high))
print(len(soxx_high))
print(len(silver_high))


print(len(tan_low))
print(len(soxx_low))
print(len(silver_low))

print(len(tan_volume))



#creating data frame 
new_data = {'date': date, 'Tan Close': tan_price, 'Silver Close': (silver_price), 'Soxx Close': (soxx_price),\
            'Oil Vol': (oil_vol), 'Tan Open': (tan_open), 'Soxx Open': (soxx_open), 'Silver Open': (silver_open), \
           'Tan High': (tan_high), 'Silver High': (silver_high), 'Soxx High': (soxx_high), 'Tan Low': (tan_low), \
           'Soxx Low': (soxx_low), 'Silver Low': (silver_low), 'Tan Volume': (tan_volume)}
new_data = pd.DataFrame(data=new_data)
print(new_data.head())



# In[24]:


#making sure moving average list is calculating properly. 

# this list calculates moving average including current day for the past window. 

window_size = 50
  
i = 0

tan_50_ma = []
  

while i < len(tan_price):
    try: 
  
        #average
        window_average = round(np.sum(new_data['Tan Close'][
          i-window_size:i]) / window_size, 2)

        tan_50_ma.append(window_average)
    except: 
        #if there are not enough days, I still want a value there so I know which rows to drop
        tan_50_ma.append(float("NaN"))
    
    i += 1
    
    
# now repeating process for 200 day MA
    
window_size = 200
  
i = 0

tan_200_ma = []
  

while i < len(tan_price):
    try: 
  
        #average
        window_average = round(np.sum(new_data['Tan Close'][
          i-window_size:i]) / window_size, 2)

        tan_200_ma.append(window_average)
    except: 
        #if there are not enough days, I still want a value there so I know which rows to drop
        tan_200_ma.append(float("NaN"))
    
    i += 1
    
#print(len(tan_200_ma))



## now to get rsi 

#rsi function I found online that I slightly altered to accept lists 
def rsi(df, periods = 14, ema = True):
    
    close_delta = df['Tan Close'][14:].diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    if ema == True:
        ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window = periods, adjust=False).mean()
        ma_down = down.rolling(window = periods, adjust=False).mean()
        
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi

#print(rsi(new_data, periods = 14, ema = True).head(20))

rsi_tan = rsi(new_data, periods = 14, ema = True).tolist()

for i in range(len(rsi_tan)): 
    try: 
        rsi_tan[i] = round(rsi_tan[i])
    except: 
        rsi_tan[i] = rsi_tan[i]
rsi_pre = [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')\
          , float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')]
rsi_tan = rsi_pre + rsi_tan
print(len(rsi_tan))

#wad function I found online that I adapted to work on my data

def williams_ad(datacopy, high_col='<HIGH>', low_col='<LOW>', close_col='<CLOSE>'):
    data = datacopy.copy()
    data['williams_ad'] = 0.
    
    for index,row in data.iterrows():
        if index > 0:
            prev_value = data.at[index-1, 'williams_ad']
            prev_close = data.at[index-1, close_col]
            if row[close_col] > prev_close:
                ad = row[close_col] - min(prev_close, row[low_col])
            elif row[close_col] < prev_close:
                ad = row[close_col] - max(prev_close, row[high_col])
            else:
                ad = 0.
                                                                                                        
            data.at[index, 'williams_ad'] = (ad+prev_value)
        
    return data
# I first recreate a data frame so that the function can easily iterate through the data

tan_high = new_data['Tan High'].tolist()
tan_low = new_data['Tan Low'].tolist()
tan_close = new_data['Tan Close'].tolist()
ad_dictionary = {'Close': tan_close, 'High': tan_high, 'Low': tan_low}
ad_df = pd.DataFrame(data=ad_dictionary)

tan_ad = williams_ad(ad_df, high_col='High', low_col='Low', close_col="Close")
tan_ad = tan_ad['williams_ad'].tolist()
#print(tan_ad)

#on balance volume code that I found online which I adapted to work on my data set
def add_obv(df):
    copy = df.copy()
    copy["OBV"] = (np.sign(copy["Tan Close"].diff()) * copy["Tan Volume"]).fillna(0).cumsum()
    return copy

tan_obv = add_obv(new_data)

tan_obv = tan_obv["OBV"].tolist()
#print(tan_obv)

#macd indicator code I found online which I adapted to work on my data set
def get_macd(price, slow, fast, smooth):
    exp1 = price.ewm(span = fast, adjust = False).mean()
    exp2 = price.ewm(span = slow, adjust = False).mean()
    macd = pd.DataFrame(exp1 - exp2).rename(columns = {'Tan Close':'macd'})
    signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns = {'macd':'signal'})
    hist = pd.DataFrame(macd['macd'] - signal['signal']).rename(columns = {0:'hist'})
    frames =  [macd, signal, hist]
    df = pd.concat(frames, join = 'inner', axis = 1)
    return df
tan_macd = get_macd(new_data['Tan Close'][26:], 26, 12, 9)
tan_macd = tan_macd['hist'].tolist()
pre = [float("nan")]*26
tan_macd = pre + tan_macd
print(len(tan_macd))










# In[25]:


#creating a column that represents the change in Soxx price 

roc_soxx = [float("nan")]
for i in range(1,len(soxx_price)): 
    try: 
        roc_soxx.append(soxx_price[i] - soxx_price[i-15])
    except: 
        print('not working')
print(len(roc_soxx))






# 

# In[26]:


## Yang-Zhang Volatility 

#First, I need the Rogers-Satchell volatility. Function returns list of calculated volatility based on that day's vol
# I created these functions based on the Yang-Zhang volatility forumla that I researched online. The Yang Zhang 
# formula also references the Rogers Satchell formula, Overnight volatility formula, and Close to close volatility
#formula, which is why I created those functions first to be able to easily call them and make my code more readable.


def Rogers_Satchell(days, dataframe, iteration, ticker):
    
    rs_sum = 0
    # getting sum 
    lookback = days
    for i in range(days):
        high_to_close = np.log(new_data[ticker + ' High'][iteration - lookback]/ \
                               new_data[ticker + ' Close'][iteration - lookback])
        high_to_open = np.log((new_data[ticker + ' High'][iteration - lookback])/ \
                               new_data[ticker + ' Open'][iteration - lookback])
        low_to_close = np.log(new_data[ticker + ' Low'][iteration - lookback]/ \
                              new_data[ticker + ' Close'][iteration - lookback])
        low_to_open = np.log(new_data[ticker + ' Low'][iteration - lookback]/ \
                             new_data[ticker + ' Open'][iteration - lookback])

        value_for_today = high_to_close * high_to_open + low_to_close * low_to_open
        
        rs_sum += value_for_today
        lookback -= 1
                        
        
    before_sqrt = rs_sum / days
    
    rs_value = np.sqrt(before_sqrt)
    return rs_value



def Overnight_vol(days, dataframe, iteration, ticker):
    o_sum = 0
    avg_c_to_o_list = []
    total_days = days
    lookback = days
    lookback1 = days
    for i in range(days): 
        avg_c_to_o_list.append(np.log(new_data[ticker + ' Open'][iteration - lookback]/ \
                                     new_data[ticker + ' Close'][iteration - lookback - 1]))
        lookback -= 1
    
    avg = sum(avg_c_to_o_list)/ len(avg_c_to_o_list)
    
    for i in range(lookback1): 
        o_sum += (np.log(new_data[ticker + ' Open'][iteration - lookback1]/ \
                                     new_data[ticker + ' Close'][iteration - lookback1 - 1]) - avg) ** 2
        lookback1 -= 1
    o_vol = o_sum / total_days
    return o_vol 
    
    
def open_to_close_vol(days, dataframe, iteration, ticker):
    oc_sum = 0
    avg_oc_list = []
    total_days = days
    lookback = days
    lookback1 = days
    for i in range(days): 
        avg_oc_list.append(np.log(new_data[ticker + ' Close'][iteration - lookback]/ \
                                     new_data[ticker + ' Open'][iteration - lookback]))
        lookback -= 1
    avg = sum(avg_oc_list) / len(avg_oc_list)
    
    for i in range(lookback1): 
        oc_sum += (np.log(new_data[ticker + ' Close'][iteration - lookback1]/ \
                                     new_data[ticker + ' Open'][iteration - lookback1]) - avg) ** 2
        lookback1 -= 1
    oc_vol = oc_sum / total_days
    return oc_vol 
    

def Yang_Zhang(days, dataframe, iteration, ticker):
    
    N = days
    k = (0.34) / (1.34 + ((N + 1)/(N - 1)))
    yz = np.sqrt(Overnight_vol(days, dataframe, iteration, ticker) + k * \
                 open_to_close_vol(days, dataframe, iteration, ticker) + \
                 Rogers_Satchell(days, dataframe, iteration, ticker) )
    return yz

                              
lst = [float("nan")] * 22                              
yz_tan = []                             
for i in range(22, len(new_data)): 
    yz_tan.append(Yang_Zhang(21, new_data, i, 'Tan') * 100)
yz_tan = lst + yz_tan

yz_soxx = []                             
for i in range(22, len(new_data)): 
    yz_soxx.append(Yang_Zhang(21, new_data, i, 'Soxx') * 100)
yz_soxx = lst + yz_soxx


yz_slv = []                             
for i in range(22, len(new_data)): 
    yz_slv.append(Yang_Zhang(21, new_data, i, 'Silver') * 100)
yz_slv = lst + yz_slv


print(len(yz_tan))
print(yz_tan[22:32])
print(yz_slv[22:32])
print(yz_soxx[22:32])


# In[27]:


#creating a list of silver price changes

silver_roc = [float("nan")]
for i in range(1,len(silver_price)): 
    try: 
        silver_roc.append(float(silver_price[i]) - float(silver_price[i-15]))
    except: 
        silver_roc.append(float("nan"))
   


# In[28]:


#creating list that checks if price went up for tan after 15 days. 1 if yes, 0 if no

tan_15 = []
for i in range(len(tan_price)): 
    try: 
        if tan_price[i + 15] > tan_price[i]: 
            tan_15.append('Up')
        else: 
            tan_15.append('Down')
    except: 
        tan_15.append(float("nan"))
print(len(tan_15))


# In[29]:


#making sure these columns have the same length (I want to create a data set with them)

print(len(date))
print(len(tan_price))
print(len(silver_roc))
print(len(oil_vol))
print(len(roc_soxx))


print('50', len(tan_50_ma))
print('200', len(tan_200_ma))
print('rsi', len(rsi_tan))
print('ad', len(tan_ad))
print('obv', len(tan_obv))
print('macd', len(tan_macd))
print('tan_15', len(tan_15))

print('silver', len(silver_price))


# In[30]:


## creating data set
new_data = {'date': date, 'tan': tan_price, 'silver_p': silver_price, 'oil_vol': oil_vol}
new_data = pd.DataFrame(data=new_data)

tan_15_days_out = {'date': date, 'tan_vol': yz_tan, 'silver_price': silver_price, 'silver_roc': silver_roc, 'oil_vol': oil_vol, 
                   'roc_soxx': \
                   roc_soxx, 'silver_vol': yz_slv, 'soxx_vol': yz_soxx,\
                   #'tan_50_ma':tan_50_ma, 'tan_200_ma': \
                   #tan_200_ma,\ 
                   'rsi_tan': rsi_tan, 'tan_ad':tan_ad, 'tan_obv': tan_obv, 
                   'tan_macd': tan_macd,\
                  'Up': tan_15,}
tan_15_days_out = pd.DataFrame(data = tan_15_days_out)
print(tan_15_days_out.head())

# taking out missing values
tan_15_days_out = tan_15_days_out.dropna()

#setting date as my index 

print(tan_15_days_out.loc[:, 'date'])
tan_15_days_out.set_index('date', inplace = True)


# In[9]:


# this was an additional feature I tested; however, I don't end up using it in my final model as it proved to not be a
# useful feature. However, I have left the code in here for future optimization purposes. 
for max_lag in [100]:
    
    def get_hurst_exponent(time_series, max_lag = 20):
        """Returns the Hurst Exponent of the time series"""

        lags = range(2, max_lag)
        
        # variances of the lagged differences
        tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]


        # calculate the slope of the log plot -> the Hurst Exponent
        reg = np.polyfit(np.log(lags), np.log(tau), 1)

        return reg[0]



    
    tan_hurst = []
    '''
    for i in range(len(tan_price)):
        try:

            tan_hurst.append(get_hurst_exponent(tan_price[0:i], max_lag))

        except: 

            tan_hurst.append(float("nan"))
    tan_hurst_q = [x for x in tan_hurst if str(x) != 'nan']

    print("lag value", max_lag, np.quantile(tan_hurst_q, [0.1, 0.5,0.75, 0.95]))
    '''


# From here, I looked for other variations of the Random Forest classifier (which I had previously tested but delted the code now) that would help reduce variance and decided to implement the Extra Trees Classifier. The Random Forest classifier was doing especially bad in short time frames, which meant that I would get less trades when implementing an optimized signal from the model. When testing the ETC, I found very consistent results at shorter time frames, and I decided to pick 8 days to trade. Though, I could use more optimization. 

# In[482]:


# after importing the appropriate libraries, I have created various for loops to try various configurations 
# of the classifier to see which has the best predictive accuracy. I only print out configurations that give me a 
# > 90 percent accuracy rate so I don't have hundreds of configurations to go through. 

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification

for estimators in [200, 500, 1500]: 
    print(estimators)
    for mD in np.arange(2,13): 
        for crit in ['gini']:
            for mss in np.arange(2, 100, 20): 
                for dec in np.arange(0, 0.1, 0.02): 
                    for days in range(3,10):
                        tan_15 = []
                        for i in range(len(tan_price)): 
                            try: 
                                if tan_price[i + days] > tan_price[i]: 
                                    tan_15.append('Up')
                                else: 
                                    tan_15.append('Down')
                            except: 
                                tan_15.append(float("nan"))
                        #print(tan_15)
                        ## creating data set


                        tan_15_days_out = {'date': date, #'tan_price': tan_price, 
                                           'tan_vol': yz_tan, 'silver_price': silver_price, 'silver_roc': silver_roc, 'oil_vol': oil_vol, 
                                       'roc_soxx': \
                                       roc_soxx, 'silver_vol': yz_slv, 'soxx_vol': yz_soxx, 
                                       'tan_50_ma':tan_50_ma, 'tan_200_ma': \
                                       tan_200_ma, 
                                       'rsi_tan': rsi_tan, 'tan_ad':tan_ad, 'tan_obv': tan_obv, 
                                       'tan_macd': tan_macd,\
                                      'Up': tan_15,}
                        tan_15_days_out = pd.DataFrame(data = tan_15_days_out)
                        #print(tan_15_days_out.head())

                        # to see if this makes a difference 
                        tan_15_days_out = tan_15_days_out.dropna()

                        #setting date as my index 

                        #print(tan_15_days_out.loc[:, 'date'])
                        tan_15_days_out.set_index('date', inplace = True)

                        train_tree_data_x = tan_15_days_out.loc['2/13/17':'7/14/20', :'tan_macd']
                        #print(train_tree_data_x.head())
                        decision_tree_data_y = tan_15_days_out.loc['2/13/17':'7/14/20','Up']
                        #print(decision_tree_data_y)

                        x_outsample = tan_15_days_out.loc['7/14/20':'12/5/22', :'tan_macd']
                        y_outsample = tan_15_days_out.loc['7/14/20':'12/5/22', 'Up']


                        #trying
                        from sklearn.model_selection import train_test_split

                        SEED = 42
                        X_train, X_test, y_train, y_test = train_test_split(train_tree_data_x, 
                                                                            decision_tree_data_y, test_size=0.2, random_state=SEED)


                        Etc = ExtraTreesClassifier(n_estimators=estimators, random_state=0, max_depth = mD, criterion = crit, 
                                                  min_samples_split = mss, min_impurity_decrease = dec )
                        Etc.fit(X_train,y_train)
                        ExtraTreesClassifier(random_state=0)
                        predictedETC = Etc.predict(X_test)
                        cm = confusion_matrix(y_test, predictedETC)

                        acc = (cm[1][1] + cm[0][0])/ (cm[0][1] + cm[1][1] + cm[0][0] + cm[1][0])


                        if acc > 0.89: 

                            print("Below. ")
                            print('Days: ', days, 'Estimators:', estimators, 'Max Depth', mD, "Criterion:", crit, 'minimum SS:', mss, "MID:", dec)
                            cm = confusion_matrix(y_test, predictedETC)


                            print((cm[1][1] + cm[0][0])/ (cm[0][1] + cm[1][1] + cm[0][0] + cm[1][0]))

                            print(confusion_matrix(y_test, predictedETC))


                        feature_names = ['tan_vol', 'silver_price', 'silver_roc', 'oil_vol', 
                                       'roc_soxx', 'silver_vol', 'soxx_vol', 'tan_50_ma', 'tan_200_ma',
                                       'rsi_tan', 'tan_ad', 'tan_obv', 'tan_macd']


                        '''
                        ## seeing if trained on outsample how the model would have done 
                        X_train, X_test, y_train, y_test = train_test_split(x_outsample, 
                                                                            y_outsample, test_size=0.2, random_state=SEED)

                        Etc.fit(X_train,y_train)
                        outsample_prediction = Etc.predict(X_test)
                        outcm = confusion_matrix(y_test, outsample_prediction)
                        print(confusion_matrix(y_test, outsample_prediction))
                        print(outcm[1][1]/ (outcm[0][1] + outcm[1][1]))

                        importances = Etc.feature_importances_
                        outsample_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)


                        plt.bar(feature_names, outsample_imp)
                        plt.xlabel('Feature Labels')
                        plt.xticks(rotation = 45)
                        plt.ylabel('Feature Importances')
                        plt.title('Comparison of different Feature Importances')
                        plt.show()

                        '''


# Here, I tested the same thing with XG Boosting, but please feel free to disregard the following optimization code as 
# the model didn't work well on out of sample data. So, I have not used it in my final backtest.

# In[425]:


#now doing the same thing but with XG Boosting 

for estimators in np.arange(200, 1201, 500):
    for days in range(3,10):
        for lR in np.arange(0.05, 1, 0.10):
            for mD in np.arange(4,11):

                tan_15 = []
                for i in range(len(tan_price)): 
                    try: 
                        if tan_price[i + days] > tan_price[i]: 
                            tan_15.append('Up')
                        else: 
                            tan_15.append('Down')
                    except: 
                        tan_15.append(float("nan"))
                #print(tan_15)
                ## creating data set


                tan_15_days_out = {'date': date, #'tan_price': tan_price, 
                                   'tan_vol': yz_tan, 'silver_price': silver_price, 'silver_roc': silver_roc, 'oil_vol': oil_vol, 
                               'roc_soxx': \
                               roc_soxx, 'silver_vol': yz_slv, 'soxx_vol': yz_soxx, 
                               'tan_50_ma':tan_50_ma, 'tan_200_ma': \
                               tan_200_ma, 
                               'rsi_tan': rsi_tan, 'tan_ad':tan_ad, 'tan_obv': tan_obv, 
                               'tan_macd': tan_macd,\
                              'Up': tan_15,}
                tan_15_days_out = pd.DataFrame(data = tan_15_days_out)
                #print(tan_15_days_out.head())

                # to see if this makes a difference 
                tan_15_days_out = tan_15_days_out.dropna()

                #setting date as my index 

                #print(tan_15_days_out.loc[:, 'date'])
                tan_15_days_out.set_index('date', inplace = True)

                train_tree_data_x = tan_15_days_out.loc['2/13/17':'7/14/20', :'tan_macd']
                #print(train_tree_data_x.head())
                decision_tree_data_y = tan_15_days_out.loc['2/13/17':'7/14/20','Up']
                #print(decision_tree_data_y)

                x_outsample = tan_15_days_out.loc['7/14/20':'12/5/22', :'tan_macd']
                y_outsample = tan_15_days_out.loc['7/14/20':'12/5/22', 'Up']


                #trying
                from sklearn.model_selection import train_test_split

                SEED = 42
                X_train, X_test, y_train, y_test = train_test_split(train_tree_data_x, 
                                                                    decision_tree_data_y, test_size=0.2, random_state=SEED)

                GBC = GradientBoostingClassifier(n_estimators=estimators, learning_rate=lR,
                                          max_depth=mD, random_state=0).fit(X_train, y_train)




                predictedGBC = GBC.predict(X_test)
                if accuracy_score(y_test, predictedGBC) > 0.91: 

                    print('Days:', days, "Max Depth:", mD, "Learning Rate:", lR, "estimators", estimators)

                    print("accuracy", accuracy_score(y_test, predictedGBC))








# Now, I am running a backtest by trading based on predictions from my optimized Extra Random Trees Model. 
# If an 'Up' day is predicted, I buy for 9 days. I sell for 9 days if a 'Down' day is predicted. 

# In[37]:


## back test 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification




## initializing model that I will use in trade 

tan_15 = []
for i in range(len(tan_price)): 
    try: 
        if tan_price[i + 9] > tan_price[i]: 
            tan_15.append('Up')
        else: 
            tan_15.append('Down')
    except: 
        tan_15.append(float("nan"))
#print(tan_15)
## creating data set


tan_15_days_out = {'date': date, #'tan_price': tan_price, 
                   'tan_vol': yz_tan, 'silver_price': silver_price, 'silver_roc': silver_roc, 'oil_vol': oil_vol, 
               'roc_soxx': \
               roc_soxx, 'silver_vol': yz_slv, 'soxx_vol': yz_soxx, 
               'tan_50_ma':tan_50_ma, 'tan_200_ma': \
               tan_200_ma, 
               'rsi_tan': rsi_tan, 'tan_ad':tan_ad, 'tan_obv': tan_obv, 
               'tan_macd': tan_macd,\
              'Up': tan_15,}
tan_15_days_out = pd.DataFrame(data = tan_15_days_out)
#print(tan_15_days_out.head())

# to see if this makes a difference 
tan_15_days_out = tan_15_days_out.dropna()

#setting date as my index 

#print(tan_15_days_out.loc[:, 'date'])
tan_15_days_out.set_index('date', inplace = True)

train_tree_data_x = tan_15_days_out.loc['2/13/17':'7/14/20', :'tan_macd']
#print(train_tree_data_x.head())
decision_tree_data_y = tan_15_days_out.loc['2/13/17':'7/14/20','Up']
#print(decision_tree_data_y)

x_outsample = tan_15_days_out.loc['7/14/20':'12/5/22', :'tan_macd']
y_outsample = tan_15_days_out.loc['7/14/20':'12/5/22', 'Up']


#recreating data set that I will be training my optimized model on and using to input predictors. 
from sklearn.model_selection import train_test_split

SEED = 42
X_train, X_test, y_train, y_test = train_test_split(train_tree_data_x, 
                                                    decision_tree_data_y, test_size=0.2, random_state=SEED)


Etc_bt = ExtraTreesClassifier(n_estimators=200, random_state=0, max_depth = 10, criterion = 'gini', 
                          min_samples_split = 2, min_impurity_decrease = 0.0 )
Etc_bt.fit(X_train,y_train)

new_data = new_data.set_index('date', drop = False)

trade_number = 0
trade_pnl = []
trade_dates = []
trade_current = 0
days_in = 0

testing_data_set = tan_15_days_out.copy()
testing_data_set = testing_data_set.loc['7/14/20':'12/5/22', :]





# this outputs feature impotances 
feature_names = ['tan_vol', 'silver_price', 'silver_roc', 'oil_vol', 
                                       'roc_soxx', 'silver_vol', 'soxx_vol', 'tan_50_ma', 'tan_200_ma',
                                       'rsi_tan', 'tan_ad', 'tan_obv', 'tan_macd']


importances = Etc_bt.feature_importances_
outsample_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)


plt.bar(feature_names, outsample_imp)
plt.xlabel('Feature Labels')
plt.xticks(rotation = 45)
plt.ylabel('Feature Importances')
plt.title('Comparison of different Feature Importances')
plt.show()


##now backtesting model 
no_trade = []


#this is my backtest, I loop through my test set and follow my previously mentioned trading strategy
for i in testing_data_set.index: 
    
    
    
    predictors = testing_data_set.loc[i, :'tan_macd']
    #print("now testing data")
    #print(predictors)
    reshape = [0]
    reshape[0] = predictors.tolist()
    predicted_class = Etc_bt.predict(reshape)
    #print(predicted_class[0], Etc_bt.predict_proba(reshape)[0][0])
    #print(predicted_class[0], Etc_bt.predict_proba(reshape)[0][1])
    
    #print(Etc_bt.predict_proba(reshape)[0])
    
    
    
    
        
    
    
    # setting up checker to see if trees predict Up, but only when we aren't already in a position 
    if trade_current == 0: 
        
        #the commented code using hurst exponent was overfit, so I have left it as a comment for 
        #potential future use. 
        #print('hust', get_hurst_exponent(new_data.loc[:i, 'Tan Close'], 100))
        '''
        hurst_data = [x for x in new_data.loc[:i, 'Tan Close'] if str(x) != 'nan']
        print(get_hurst_exponent(hurst_data, 100))
        if get_hurst_exponent(hurst_data, 100) > 0.60:
            no_trade.append(i)
            print('no trade', i)
            trade_pnl.append(0)
            trade_dates.append(i)
            continue
        '''
        
        if predicted_class[0] == 'Up': 
            
            trade_number += 1
            trade_dates.append(i)
            trade_current = 1
            
            
            tan_buy_price = new_data.loc[i, 'Tan Close']
            trade_pnl.append(0)
            continue
            
        elif predicted_class[0] == 'Down': 
            
            trade_number += 1
            trade_dates.append(i)
            trade_current = -1
            
            
            tan_buy_price = new_data.loc[i, 'Tan Close']
            trade_pnl.append(0)
            continue
        
        
        else: 
            trade_pnl.append(0)
            trade_dates.append(i)
            
        
        
        
    
        
            
    #keeping track of pnl and days to see when to exit if I am currently in a trade     
    if trade_current == 1: 
        days_in += 1
        trade_dates.append(i)
        
        #daily pnl recalculation from trade 
        tan_sell_price = new_data['Tan Close'][i]
        pnl = (tan_sell_price - tan_buy_price)/tan_buy_price
        trade_pnl.append(pnl)
        tan_buy_price = new_data['Tan Close'][i]
        
        if days_in == 9:
            

            
            #resetting values for the next trade 
            
            
            trade_current = 0
            tan_buy_price = 0
            
            tan_sell_price = 0
            days_in = 0
        
            
    if trade_current == -1: 
        days_in += 1
        trade_dates.append(i)
        
        #daily pnl recalculation from trade 
        tan_sell_price = new_data['Tan Close'][i]
        pnl = (tan_buy_price - tan_sell_price)/tan_buy_price
        trade_pnl.append(pnl)
        tan_buy_price = new_data['Tan Close'][i]
        
        if days_in == 9:
            

            
            #resetting values for the next trade 
            
            
            trade_current = 0
            tan_buy_price = 0
            
            tan_sell_price = 0
            days_in = 0
        
        
            


            


        
        
            
    
        
        
   
    


# In[35]:


print(len(testing_data_set.index.tolist()))
print(min(trade_pnl))
print(len(trade_dates))
print(no_trade)


# In[36]:


#now visualizing trade results

import math
avg_pnl = np.mean(trade_pnl)
        #sharpe 
sharpe_before_adj = avg_pnl/np.std(trade_pnl)
sharpe = sharpe_before_adj * math.sqrt(252)
sharpe = sharpe - (0.028/(np.std(trade_pnl)*math.sqrt(252)))
print('sharpe:', sharpe) 
print('days in trade:', 9)
print('number of trades:', trade_number)
print('average pnl:', avg_pnl)
print('max gain:', max(trade_pnl))
print('max loss:', min(trade_pnl))



df_cum_returns = {'Dates': testing_data_set.index.tolist(), 'returns': (np.add(1,trade_pnl)).cumprod()}
df_final_returns = pd.DataFrame(df_cum_returns, columns=['Dates','returns'])

df_final_returns.plot(title="Solar Forecasting over Two Years Hurst",x='Dates', y='returns')

plt.xticks(rotation = 45)
plt.show()



# The above code is in reference to my final model that I picked and showed was statistically significant. 
# 
# Below, I have left other models that I explored and tried. However, please feel free to disregard, as these methodologies did not prove to have much predictive power. I have left them in though, as they may prove useful for future endeavors. 

# In[498]:


### rolling extra random trees

## back test 
new_data = new_data.set_index('date', drop = False)

trade_number = 0
trade_pnl = []
trade_dates = []
trade_current = 0
days_in = 0

testing_data_set = tan_15_days_out.copy()
testing_data_set = testing_data_set.loc['7/14/20':'12/5/22', :]

lagged = tan_15_days_out.copy()
lagged = lagged.loc['6/29/20':'12/5/22', :]

list_of_index = list(lagged.index.values)
index_of_lagged = 0


## initializing model that I will use in trade 

tan_15 = []
for i in range(len(tan_price)): 
    try: 
        if tan_price[i + 9] > tan_price[i]: 
            tan_15.append('Up')
        else: 
            tan_15.append('Down')
    except: 
        tan_15.append(float("nan"))
#print(tan_15)
## creating data set


tan_15_days_out = {'date': date, #'tan_price': tan_price, 
                   'tan_vol': yz_tan, 'silver_price': silver_price, 'silver_roc': silver_roc, 'oil_vol': oil_vol, 
               'roc_soxx': \
               roc_soxx, 'silver_vol': yz_slv, 'soxx_vol': yz_soxx, 
               'tan_50_ma':tan_50_ma, 'tan_200_ma': \
               tan_200_ma, 
               'rsi_tan': rsi_tan, 'tan_ad':tan_ad, 'tan_obv': tan_obv, 
               'tan_macd': tan_macd,\
              'Up': tan_15,}
tan_15_days_out = pd.DataFrame(data = tan_15_days_out)
#print(tan_15_days_out.head())

# to see if this makes a difference 
tan_15_days_out = tan_15_days_out.dropna()

#setting date as my index 

#print(tan_15_days_out.loc[:, 'date'])
tan_15_days_out.set_index('date', inplace = True)

train_tree_data_x = tan_15_days_out.loc['2/13/17':'7/14/20', :'tan_macd']
#print(train_tree_data_x.head())
decision_tree_data_y = tan_15_days_out.loc['2/13/17':'7/14/20','Up']
#print(decision_tree_data_y)

x_outsample = tan_15_days_out.loc['7/14/20':'12/5/22', :'tan_macd']
y_outsample = tan_15_days_out.loc['7/14/20':'12/5/22', 'Up']


#trying
from sklearn.model_selection import train_test_split

SEED = 42
X_train, X_test, y_train, y_test = train_test_split(train_tree_data_x, 
                                                    decision_tree_data_y, test_size=0.2, random_state=SEED)


Etc_rolling = ExtraTreesClassifier(n_estimators=200, random_state=0, max_depth = 10, criterion = 'gini', 
                          min_samples_split = 2, min_impurity_decrease = 0.0 )
Etc_rolling.fit(X_train,y_train)


##now backtesting model 


for i in testing_data_set.index: 
    
    #Etc = ExtraTreesClassifier(n_estimators=500, random_state=0)
    
    retrainX = tan_15_days_out.loc['2/13/17': list_of_index[index_of_lagged], :'tan_macd']
    
    retrainY = tan_15_days_out.loc['2/13/17': list_of_index[index_of_lagged], 'Up']
    
    
    
    Etc_rolling.fit(retrainX,retrainY)
    
    #print(list_of_index[index_of_lagged])
    #print(retrainX)
    
        
    index_of_lagged += 1
    train_delay += 1
    
    
    
    
    predictors = testing_data_set.loc[i, :'tan_macd']
    #print("now testing data")
    #print(predictors)
    reshape = [0]
    reshape[0] = predictors.tolist()
    predicted_class = Etc_rolling.predict(reshape)
    #print(predicted_class[0], Etc_bt.predict_proba(reshape)[0][0])
    #print(predicted_class[0], Etc_bt.predict_proba(reshape)[0][1])
    
    #print(Etc_bt.predict_proba(reshape)[0])
    
    
    
    
        
    
    
    # setting up checker to see if trees predict Up, but only when we aren't already in a position 
    if trade_current == 0: 
        #print('hust', get_hurst_exponent(new_data.loc[:i, 'Tan Close'], 100))
        
        if get_hurst_exponent(new_data.loc[:i, 'Tan Close'], 100) > 0.57:
            continue
        
        if predicted_class[0] == 'Up' and Etc_rolling.predict_proba(reshape)[0][1] > 0.68: 
            
            trade_number += 1
            trade_dates.append(i)
            trade_current = 1
            
            
            tan_buy_price = new_data.loc[i, 'Tan Close']
            trade_pnl.append(0)
            continue
            
        elif predicted_class[0] == 'Down' and Etc_rolling.predict_proba(reshape)[0][0] > 0.68: 
            
            trade_number += 1
            trade_dates.append(i)
            trade_current = -1
            
            
            tan_buy_price = new_data.loc[i, 'Tan Close']
            trade_pnl.append(0)
            continue
        
        
        else: 
            trade_pnl.append(0)
            trade_dates.append(i)
        
        
        
    
        
            
    #keeping track of pnl and days to see when to exit        
    if trade_current == 1: 
        days_in += 1
        trade_dates.append(i)
        
        #pnl from trade 
        tan_sell_price = new_data['Tan Close'][i]
        pnl = (tan_sell_price - tan_buy_price)/tan_buy_price
        trade_pnl.append(pnl)
        tan_buy_price = new_data['Tan Close'][i]
        
        if days_in == 9:
            

            
            #resetting values for the next trade 
            
            
            trade_current = 0
            tan_buy_price = 0
            
            tan_sell_price = 0
            days_in = 0
        
            
    if trade_current == -1: 
        days_in += 1
        trade_dates.append(i)
        
        #pnl from trade 
        tan_sell_price = new_data['Tan Close'][i]
        pnl = (tan_buy_price - tan_sell_price)/tan_buy_price
        trade_pnl.append(pnl)
        tan_buy_price = new_data['Tan Close'][i]
        
        if days_in == 9:
            

            
            #resetting values for the next trade 
            
            
            trade_current = 0
            tan_buy_price = 0
            
            tan_sell_price = 0
            days_in = 0
        
        
            


avg_pnl = np.mean(trade_pnl)
        #sharpe 
sharpe_before_adj = avg_pnl/np.std(trade_pnl)
sharpe = sharpe_before_adj * math.sqrt(252)
sharpe = sharpe - (0.028/(np.std(trade_pnl)*math.sqrt(252)))
print('sharpe:', sharpe) 
print('days in trade:', 8)
print('number of trades:', trade_number)
print('average pnl:', avg_pnl)
print('max gain:', max(trade_pnl))
print('max loss:', min(trade_pnl))



df_cum_returns = {'Dates': testing_data_set.index.tolist(), 'returns': (np.add(1,trade_pnl)).cumprod()}
df_final_returns = pd.DataFrame(df_cum_returns, columns=['Dates','returns'])

df_final_returns.plot(title="Solar Forecasting over Two Years",x='Dates', y='returns')

plt.xticks(rotation = 45)
plt.show()
          


        
        
            
    
        
        
   
    


# In[429]:


## now trying backtest where I pre create the model with XG Boosting 
## back test 
new_data = new_data.set_index('date', drop = False)

trade_number = 0
trade_pnl = []
trade_dates = []
trade_current = 0
days_in = 0

testing_data_set = tan_15_days_out.copy()
testing_data_set = testing_data_set.loc['7/14/20':'12/5/22', :]

lagged = tan_15_days_out.copy()
lagged = lagged.loc['6/29/20':'12/5/22', :]

list_of_index = list(lagged.index.values)
index_of_lagged = 0


## initializing model that I will use in trade 

tan_15 = []
for i in range(len(tan_price)): 
    try: 
        if tan_price[i + 9] > tan_price[i]: 
            tan_15.append('Up')
        else: 
            tan_15.append('Down')
    except: 
        tan_15.append(float("nan"))

## creating data set


tan_15_days_out = {'date': date, #'tan_price': tan_price, 
                   'tan_vol': yz_tan, 'silver_price': silver_price, 'silver_roc': silver_roc, 'oil_vol': oil_vol, 
               'roc_soxx': \
               roc_soxx, 'silver_vol': yz_slv, 'soxx_vol': yz_soxx, 
               'tan_50_ma':tan_50_ma, 'tan_200_ma': \
               tan_200_ma, 
               'rsi_tan': rsi_tan, 'tan_ad':tan_ad, 'tan_obv': tan_obv, 
               'tan_macd': tan_macd,\
              'Up': tan_15,}
tan_15_days_out = pd.DataFrame(data = tan_15_days_out)
#print(tan_15_days_out.head())

# to see if this makes a difference 
tan_15_days_out = tan_15_days_out.dropna()

#setting date as my index 

#print(tan_15_days_out.loc[:, 'date'])
tan_15_days_out.set_index('date', inplace = True)

train_tree_data_x = tan_15_days_out.loc['2/13/17':'7/14/20', :'tan_macd']
#print(train_tree_data_x.head())
decision_tree_data_y = tan_15_days_out.loc['2/13/17':'7/14/20','Up']
#print(decision_tree_data_y)

x_outsample = tan_15_days_out.loc['7/14/20':'12/5/22', :'tan_macd']
y_outsample = tan_15_days_out.loc['7/14/20':'12/5/22', 'Up']


#trying
from sklearn.model_selection import train_test_split

SEED = 42
X_train, X_test, y_train, y_test = train_test_split(train_tree_data_x, 
                                                    decision_tree_data_y, test_size=0.2, random_state=SEED)


GBC_bt = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                          max_depth=5, random_state=0).fit(X_train, y_train)



##now backtesting model 


for i in testing_data_set.index: 
    
    
    
    
    predictors = testing_data_set.loc[i, :'tan_macd']
    #print("now testing data")
    #print(predictors)
    reshape = [0]
    reshape[0] = predictors.tolist()
    predicted_class = GBC_bt.predict(reshape)
    
   
    
    
    
    
        
    
    
    # setting up checker to see if trees predict Up, but only when we aren't already in a position 
    if trade_current == 0: 
        
        if predicted_class[0] == 'Up': 
            
            trade_number += 1
            trade_dates.append(i)
            trade_current = 1
            
            
            tan_buy_price = new_data.loc[i, 'Tan Close']
            trade_pnl.append(0)
            continue
            
        elif predicted_class[0] == 'Down': 
            
            trade_number += 1
            trade_dates.append(i)
            trade_current = -1
            
            
            tan_buy_price = new_data.loc[i, 'Tan Close']
            trade_pnl.append(0)
            continue
        
        
        else: 
            trade_pnl.append(0)
        
        
        
    
        
            
    #keeping track of pnl and days to see when to exit        
    if trade_current == 1: 
        days_in += 1
        trade_dates.append(i)
        
        #pnl from trade 
        tan_sell_price = new_data['Tan Close'][i]
        pnl = (tan_sell_price - tan_buy_price)/tan_buy_price
        trade_pnl.append(pnl)
        tan_buy_price = new_data['Tan Close'][i]
        
        if days_in == 9:
            

            
            #resetting values for the next trade 
            
            
            trade_current = 0
            tan_buy_price = 0
            
            tan_sell_price = 0
            days_in = 0
        
            
    if trade_current == -1: 
        days_in += 1
        trade_dates.append(i)
        
        #pnl from trade 
        tan_sell_price = new_data['Tan Close'][i]
        pnl = (tan_buy_price - tan_sell_price)/tan_buy_price
        trade_pnl.append(pnl)
        tan_buy_price = new_data['Tan Close'][i]
        
        if days_in == 9:
            

            
            #resetting values for the next trade 
            
            
            trade_current = 0
            tan_buy_price = 0
            
            tan_sell_price = 0
            days_in = 0
        

        
avg_pnl = np.mean(trade_pnl)
        #sharpe 
sharpe_before_adj = avg_pnl/np.std(trade_pnl)
sharpe = sharpe_before_adj * math.sqrt(252)
sharpe = sharpe - (0.028/(np.std(trade_pnl)*math.sqrt(252)))
print('sharpe:', sharpe) 
print('days in trade:', 8)
print('number of trades:', trade_number)
print('average pnl:', avg_pnl)
print('max gain:', max(trade_pnl))
print('max loss:', min(trade_pnl))



df_cum_returns = {'Dates': testing_data_set.index.tolist(), 'returns': (np.add(1,trade_pnl)).cumprod()}
df_final_returns = pd.DataFrame(df_cum_returns, columns=['Dates','returns'])

df_final_returns.plot(title="Solar Forecasting over Two Years",x='Dates', y='returns')

plt.xticks(rotation = 45)
plt.show()
            


            


        
        
            
    
        
        
   
    


# I also want to consider the role of the hurst exponent, and will be attempting to try different moving averages for the hurst exponent. 
# 
# 

# In[337]:


## optimizing learning period 
import warnings
warnings.filterwarnings('ignore')

daylist = []
learningperday = []
for days in range(1, 10): 
    
    # trying to optimize from 1 to 10 days what forecast period learns better
    
    tan_15 = []
    for i in range(len(tan_price)): 
        try: 
            if tan_price[i + days] > tan_price[i]: 
                tan_15.append('Up')
            else: 
                tan_15.append('Down')
        except: 
            tan_15.append(float("nan"))
   
    ## creating data set
    

    optimize_Meta_Set = {'date': date, 
                       'tan_vol': yz_tan, 'silver_price': silver_price, 'silver_roc': silver_roc, 'oil_vol': oil_vol, 
                   'roc_soxx': \
                   roc_soxx, 'silver_vol': yz_slv, 'soxx_vol': yz_soxx, 
                   'tan_50_ma':tan_50_ma, 'tan_200_ma': \
                   tan_200_ma, 
                   'rsi_tan': rsi_tan, 'tan_ad':tan_ad, 'tan_obv': tan_obv, 
                   'tan_macd': tan_macd,\
                  'Up': tan_15,}
    optimize_Meta_Set = pd.DataFrame(data = optimize_Meta_Set)
    #print(tan_15_days_out.head())

    # to see if this makes a difference 
    optimize_Meta_Set = optimize_Meta_Set.dropna()

    #setting date as my index 

    #print(tan_15_days_out.loc[:, 'date'])
    optimize_Meta_Set.set_index('date', inplace = True)

    
    
    optimize = optimize_Meta_Set.copy()
    optimize = optimize.loc['2/27/17':'7/14/20', :]

    lagged_O = optimize_Meta_Set.copy()
    lagged_O = lagged_O.loc['2/13/17':'7/14/20', :]

    list_of_index = list(lagged_O.index.values)
    index_of_lagged = 0

    predictionsCorrectList = []
    
    for i in optimize.index: 
    
        Etc = ExtraTreesClassifier(n_estimators=500, random_state=0)
        retrainX = optimize_Meta_Set.loc['2/13/17':list_of_index[index_of_lagged], :'tan_macd']
        retrainY = optimize_Meta_Set.loc['2/13/17':list_of_index[index_of_lagged], 'Up']

        Etc.fit(retrainX,retrainY)


        index_of_lagged += 1
        train_delay += 1



        predictors = optimize.loc[i, :'tan_macd']
        reshape = [0]
        reshape[0] = predictors.tolist()
        predicted_class = Etc.predict(reshape)
        
        if predicted_class == optimize.loc[i, 'Up']:
            predictionsCorrectList.append(1)
        else: 
            predictionsCorrectList.append(0)
            
            
            
    learningperday.append(sum(predictionsCorrectList)/len(predictionsCorrectList))

    daylist.append(days)
    
    
for i in range(len(learningperday)):
    print("Day: ", daylist[i], ". How accurate? ", learningperday[i])
    
        
        
    
    


# Now I will be considering XG Boosting to see if it has any additional predicitve power. Because traditional CV methods aren't robust when working with time series data, I'm using a method based on time series cross validation, where in each iteration, I retrain the model on all previous data and compare my prediction to the actual class for that day. If the predicted class matches the actual class (noting that the class definitions are 'Up' or 'Down' in a certain future period of days ('days' in the code)), then that day is assigned a 1. If the prediction and actual class do not match, that day gets a 0. I then average the indicator outcomes (the list of 1's and 0's for the entire time period) to guage how well my model performed. 
# 
# I got great results. However, I believe there is some statstical error, or overfitting, as on the test set, I'm getting extremely different results. So, I will now try a moving average-type optimization, to see if gradient boosting performs better on when using a moving subset of the data is used to train the model. 

# In[405]:


from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier







daylist = []
learningperday = []
learning_rate = []
max_depth = []
#min_samples_leaf = []
#min_samples_split = []
lookback_window = [] 



for lookback in np.arange(30,360, 60): 
    for days in np.arange(4,9): 
        for lR in np.arange(0.05, 0.3, 0.10):
            for mD in np.arange(4,9):


        # trying to optimize from 1 to 10 days what forecast period learns better

                tan_15 = []
                for i in range(len(tan_price)): 
                    try: 
                        if tan_price[i + days] > tan_price[i]: 
                            tan_15.append('Up')
                        else: 
                            tan_15.append('Down')
                    except: 
                        tan_15.append(float("nan"))

                ## creating data set


                optimize_Meta_Set = {'date': date, 
                                   'tan_vol': yz_tan, 'silver_price': silver_price, 'silver_roc': silver_roc, 'oil_vol': oil_vol, 
                               'roc_soxx': \
                               roc_soxx, 'silver_vol': yz_slv, 'soxx_vol': yz_soxx, 
                               'tan_50_ma':tan_50_ma, 'tan_200_ma': \
                               tan_200_ma, 
                               'rsi_tan': rsi_tan, 'tan_ad':tan_ad, 'tan_obv': tan_obv, 
                               'tan_macd': tan_macd,\
                              'Up': tan_15}
                optimize_Meta_Set = pd.DataFrame(data = optimize_Meta_Set)
                #print(tan_15_days_out.head())

                # to see if this makes a difference 
                optimize_Meta_Set = optimize_Meta_Set.dropna()

                #setting date as my index 

                #print(tan_15_days_out.loc[:, 'date'])
                optimize_Meta_Set.set_index('date', inplace = True)



                

                #setting a stopping point for the training set
                lagged_O = optimize_Meta_Set.copy()
                lagged_O = lagged_O.loc['2/21/17':'7/14/20', :]

                
                
                

                list_of_index = list(lagged_O.index.values)
                index_of_lagged = lookback
                starting_index_of_lagged = 0
                
                
                
                optimize_index = list_of_index[lookback + 10]
                optimize = optimize_Meta_Set.copy()
                
                
                optimize = optimize.loc[optimize_index:'7/14/20', :]


                predictionsCorrectList = []

                for i in optimize.index: 







                    retrainX = optimize_Meta_Set.loc[list_of_index[starting_index_of_lagged]:list_of_index[index_of_lagged], :'tan_macd']
                    retrainY = optimize_Meta_Set.loc[list_of_index[starting_index_of_lagged]:list_of_index[index_of_lagged], 'Up']
                    
                    starting_index_of_lagged += 1
                    index_of_lagged += 1
                    train_delay += 1

                    # sometimes the given window only has one class. So need exception handling 
                    try:
                        GBC = GradientBoostingClassifier(n_estimators=200, learning_rate=lR,
                              max_depth=mD, random_state=0).fit(retrainX, retrainY)
                    except: 
                        continue
                    #GBC.score(X_test, y_test)



                    



                    predictors = optimize.loc[i, :'tan_macd']
                    reshape = [0]
                    reshape[0] = predictors.tolist()
                    predicted_class = GBC.predict(reshape)

                    if predicted_class == optimize.loc[i, 'Up']:
                        predictionsCorrectList.append(1)
                    else: 
                        predictionsCorrectList.append(0)



                learningperday.append(sum(predictionsCorrectList)/len(predictionsCorrectList))

                daylist.append(days)
                learning_rate.append(lR)
                max_depth.append(mD)
                lookback_window.append(lookback)
                print("Day: ", days, ". How accurate? ", sum(predictionsCorrectList)/len(predictionsCorrectList), end = " ")
                print("learning:", lR, "max depth:", mD, "lookback:", lookback)



    

        
        
    
    


# In[367]:


GBresults = {'Number of Days' : daylist, 'Learning Rate':learning_rate, 'Max Depth': max_depth, 
              
             'Performance':learningperday }
GBresults = pd.DataFrame(data=GBresults)
GBresults = GBresults.sort_values("Performance", ascending = False)
print(GBresults.head(50))


# Here, I will apply the same trade and testing but with gradient boosting. 

# In[400]:


tan_15 = []
for i in range(len(tan_price)): 
    try: 
        if tan_price[i + 9] > tan_price[i]:                 
            tan_15.append('Up')
        else: 
             tan_15.append('Down')
    except: 
        tan_15.append(float("nan"))
    #print(tan_15)
    ## creating data set
    

tan_15_days_out = {'date': date, #'tan_price': tan_price, 
                       'tan_vol': yz_tan, 'silver_price': silver_price, 'silver_roc': silver_roc, 'oil_vol': oil_vol, 
                   'roc_soxx': \
                   roc_soxx, 'silver_vol': yz_slv, 'soxx_vol': yz_soxx, 
                   'tan_50_ma':tan_50_ma, 'tan_200_ma': \
                   tan_200_ma, 
                   'rsi_tan': rsi_tan, 'tan_ad':tan_ad, 'tan_obv': tan_obv, 
                   'tan_macd': tan_macd,\
                  'Up': tan_15}

tan_15_days_out = pd.DataFrame(data = tan_15_days_out)
    #print(tan_15_days_out.head())

    # to see if this makes a difference 
tan_15_days_out = tan_15_days_out.dropna()

    #setting date as my index 

    #print(tan_15_days_out.loc[:, 'date'])
tan_15_days_out.set_index('date', inplace = True)


new_data = new_data.set_index('date', drop = False)

trade_number = 0
trade_pnl = []
trade_dates = []
trade_current = 0
days_in = 0

testing_data_set = tan_15_days_out.copy()
testing_data_set = testing_data_set.loc['7/14/20':'12/5/22', :]

lagged = tan_15_days_out.copy()
lagged = lagged.loc['6/29/20':'12/5/22', :]

list_of_index = list(lagged.index.values)
index_of_lagged = 0


tscv = []
for i in testing_data_set.index: 
    
    
    retrainX = tan_15_days_out.loc['5/21/20':list_of_index[index_of_lagged], :'tan_macd']
    retrainY = tan_15_days_out.loc['5/21/20':list_of_index[index_of_lagged], 'Up']
    
    GBC = GradientBoostingClassifier(n_estimators=200, learning_rate=0.15,
                      max_depth=6, random_state=0).fit(retrainX, retrainY)
                #GBC.score(X_test, y_test)



    index_of_lagged += 1
    train_delay += 1



    predictors = testing_data_set.loc[i, :'tan_macd']
    reshape = [0]
    reshape[0] = predictors.tolist()
    predicted_class = GBC.predict(reshape)

    if predicted_class == testing_data_set.loc[i, 'Up']: 
        tscv.append(1)
    else:
        tscv.append(0)
    
    
    
    
    
        
    
    
    # setting up checker to see if trees predict Up, but only when we aren't already in a position 
    if trade_current == 0: 
        
        if predicted_class[0] == 'Up': 
            
            trade_number += 1
            trade_dates.append(i)
            trade_current = 1
            
            
            tan_buy_price = new_data.loc[i, 'Tan Close']
            trade_pnl.append(0)
            continue
           
        elif predicted_class[0] == 'Down': 
            
            trade_number += 1
            trade_dates.append(i)
            trade_current = -1
            
            
            tan_buy_price = new_data.loc[i, 'Tan Close']
            trade_pnl.append(0)
            continue
        
        
        else: 
            trade_pnl.append(0)
        
        
        
    
        
            
    #keeping track of pnl and days to see when to exit        
    if trade_current == 1: 
        days_in += 1
        trade_dates.append(i)
        
        #pnl from trade 
        tan_sell_price = new_data['Tan Close'][i]
        pnl = (tan_sell_price - tan_buy_price)/tan_buy_price
        trade_pnl.append(pnl)
        tan_buy_price = new_data['Tan Close'][i]
        
        if days_in == 9:
            

            
            #resetting values for the next trade 
            
            
            trade_current = 0
            tan_buy_price = 0
            
            tan_sell_price = 0
            days_in = 0
        
            
    if trade_current == -1: 
        days_in += 1
        trade_dates.append(i)
        
        #pnl from trade 
        tan_sell_price = new_data['Tan Close'][i]
        pnl = (tan_buy_price - tan_sell_price)/tan_buy_price
        trade_pnl.append(pnl)
        tan_buy_price = new_data['Tan Close'][i]
        
        if days_in == 9:
            

            
            #resetting values for the next trade 
            
            
            trade_current = 0
            tan_buy_price = 0
            
            tan_sell_price = 0
            days_in = 0
        
        
            
import math
avg_pnl = np.mean(trade_pnl)
        #sharpe 
sharpe_before_adj = avg_pnl/np.std(trade_pnl)
sharpe = sharpe_before_adj * math.sqrt(252)
sharpe = sharpe - (0.028/(np.std(trade_pnl)*math.sqrt(252)))
print('sharpe:', sharpe) 
print('days in trade:', 8)
print('number of trades:', trade_number)
print('average pnl:', avg_pnl)
print('max gain:', max(trade_pnl))
print('max loss:', min(trade_pnl))



df_cum_returns = {'Dates': testing_data_set.index.tolist(), 'returns': (np.add(1,trade_pnl)).cumprod()}
df_final_returns = pd.DataFrame(df_cum_returns, columns=['Dates','returns'])

df_final_returns.plot(title="Solar Forecasting over Two Years",x='Dates', y='returns')

plt.xticks(rotation = 45)
plt.show()



            
print("tscv:", sum(tscv)/len(tscv))

        
        
            
    
        
        
   
    


# In[398]:


# trying to plot how well the xg boosting model performs.

daylist = []
learningperday = []
learning_rate = []
max_depth = []
date_to_plot = []
#min_samples_leaf = []
#min_samples_split = []

            
    
    # trying to optimize from 1 to 10 days what forecast period learns better
    
tan_15 = []
for i in range(len(tan_price)): 
    try: 
        if tan_price[i + 9] > tan_price[i]: 
            tan_15.append('Up')
        else: 
            tan_15.append('Down')
    except: 
        tan_15.append(float("nan"))

## creating data set


optimize_Meta_Set = {'date': date, 
                   'tan_vol': yz_tan, 'silver_price': silver_price, 'silver_roc': silver_roc, 'oil_vol': oil_vol, 
               'roc_soxx': \
               roc_soxx, 'silver_vol': yz_slv, 'soxx_vol': yz_soxx, 
               'tan_50_ma':tan_50_ma, 'tan_200_ma': \
               tan_200_ma, 
               'rsi_tan': rsi_tan, 'tan_ad':tan_ad, 'tan_obv': tan_obv, 
               'tan_macd': tan_macd,\
              'Up': tan_15}


print(len(yz_tan))
print(len(silver_price))
print(len(silver_roc))
print(len(oil_vol))
print(len(yz_slv))
print(len(roc_soxx))
print(len(date))

print('50', len(tan_50_ma))
print('200', len(tan_200_ma))
print('rsi', len(rsi_tan))
print('ad', len(tan_ad))
print('obv', len(tan_obv))
print('macd', len(tan_macd))
print('tan_15', len(tan_15))

print('silver', len(silver_price))
optimize_Meta_Set = pd.DataFrame(data = optimize_Meta_Set)
#print(tan_15_days_out.head())

# to see if this makes a difference 
optimize_Meta_Set = optimize_Meta_Set.dropna()

#setting date as my index 

#print(tan_15_days_out.loc[:, 'date'])
optimize_Meta_Set.set_index('date', inplace = True)



optimize = optimize_Meta_Set.copy()
optimize = optimize.loc['3/3/17':'7/14/20', :]

lagged_O = optimize_Meta_Set.copy()
lagged_O = lagged_O.loc['2/21/17':'7/14/20', :]

list_of_index = list(lagged_O.index.values)
index_of_lagged = 5

predictionsCorrectList = []

for i in optimize.index: 







    retrainX = optimize_Meta_Set.loc['2/13/17':list_of_index[index_of_lagged], :'tan_macd']
    retrainY = optimize_Meta_Set.loc['2/13/17':list_of_index[index_of_lagged], 'Up']




    GBC = GradientBoostingClassifier(n_estimators=200, learning_rate=0.15,
                      max_depth=6, random_state=0).fit(retrainX, retrainY)
    #GBC.score(X_test, y_test)



    index_of_lagged += 1
    train_delay += 1



    predictors = optimize.loc[i, :'tan_macd']
    reshape = [0]
    reshape[0] = predictors.tolist()
    predicted_class = GBC.predict(reshape)

    if predicted_class == optimize.loc[i, 'Up']:
        predictionsCorrectList.append(1)
    else: 
        predictionsCorrectList.append(0)
    



    learningperday.append(sum(predictionsCorrectList)/len(predictionsCorrectList))

    date_to_plot.append(i)
    
    
print("Day: ", days, ". How accurate? ", sum(predictionsCorrectList)/len(predictionsCorrectList), end = "")
print("learning:", lR, "max depth:", mD)



#now plotting data
import numpy as np
from matplotlib.pyplot import step, show




xaxis = date_to_plot
yaxis = predictionsCorrectList
step(xaxis, yaxis)
show()



# In[399]:


xaxis = date_to_plot
yaxis = predictionsCorrectList
step(xaxis, yaxis)
show()


# In[402]:


print(optimize_Meta_Set.head())


# In[ ]:




