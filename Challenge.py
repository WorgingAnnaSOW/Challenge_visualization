# %%
import math as m
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from download import download
import seaborn as sns


#-------------CLEANING THE DATAFRAME AND CREATING NEWS--------------

# %%
## IMPORTATION
url= "https://docs.google.com/spreadsheets/d/e/2PACX-1vQVtdpXMHB4g9h75a0jw8CsrqSuQmP5eMIB2adpKR5hkRggwMwzFy5kB-AIThodhVHNLxlZYm8fuoWj/pub?gid=2105854808&single=true&output=csv"
#url="https://compteurs.velocite-montpellier.fr/communautaire/albert"
path_target= "./challenge.csv"
download(url, path_target, replace=True)
df_traffic_raw = pd.read_csv("challenge.csv")
df_traffic_raw.describe()
df_traffic_raw
#df_traffic_raw.dtypes

# %%
## NEW DATAFRAME WITHOUT USELESS COLUMS AND ROWS
df_traffic_raw.drop(['Unnamed: 4', 'Remarque'],axis=1, inplace=True)
df_traffic_raw.drop( [0,1],axis=0, inplace=True)
df_traffic_raw.rename(columns = {'Heure / Time' : 'Time' , 'Vélos ce jour / Today\'s total' : 'Today\'s_total'},inplace=True)
df_traffic_raw
#df_traffic_raw.columns
df_traffic_raw.describe()
df_traffic_raw

# %%
df_traffic_raw.tail(15)

# %%
## CREATE A NEW COLUMN AS date_time AND
## CONVERTING THE date_time COLUMN AS DATETIME TYPE
from datetime import datetime
#df_traffic_raw['date_time'] = df_traffic_raw['Date'].astype(str) + '-' +df_traffic_raw['Time']#.astype(str) 
time_improved = pd.to_datetime(df_traffic_raw['Date'] + ' ' + df_traffic_raw['Time'], format='%d/%m/%Y %H:%M:%S', infer_datetime_format=True)

# %%
df_traffic_raw['date_time']=time_improved
df_traffic_raw

# %%
#df_traffic_raw['Time']=df_traffic_raw['Time'].astype('str')
df_traffic_raw.iloc[195:205]

# %%
## DELETE NAN VALUES
df_traffic_raw=df_traffic_raw.dropna()
df_traffic_raw.iloc[393:410]

# %%
## CREATE A NEW DATAFRAME WITH A NEW COLUMN AS THE DAYS OF THE WEEK 
# It will be usefull to choose wich days we wan't to work with
df_traffic_raw['days'] = df_traffic_raw.date_time.dt.weekday
df_traffic_raw.dtypes

# %%
df_traffic_raw

# %%
## ADING THE HOUR COLUMN
# It will be usefull to choose wich hours we wan't to work with
df_traffic_raw["Hour"]=df_traffic_raw["date_time"].dt.hour
df_traffic_raw

# %%
# ADING THE MONTH COLUMN
# It will be usefull to choose wich months we wan't to work with
df_traffic_raw["Month"]=df_traffic_raw["date_time"].dt.month
df_traffic_raw.iloc[393:410]

# %%
df_traffic_raw

# %%
## NEW INDEX 
# usefull to plot graphs and see variation 
indexed_df = df_traffic_raw.set_index(['date_time']) # change the index of the dataframe as date_time
indexed_df

# %%  
# here is the evolution of bikes' number by months
plt.title("Evolution by months")
indexed_df['Today\'s_total'].plot(figsize=(12,5))
# %%








#--------------TRYING TO HAVE A COOLER DATAFRAME-------------------
#--------------WITH THE NECESSARY OF INFORMATION--------------------
#-------------WITHOUT THE WEEKEND BETWEEN 07 AM AND 9h59AM---------------
#-WE CAN TAKE VALUES TILL 09h59 CUZ WE DON'T WANNA LOSE MUCH INFORMATION----
#--------------WE'LL REMOVE MARCH,APRIL AND MAY 2020-------------
#------------------BECAUSE OF THE CONFINEMENT---------------------
# %%
## BETWEEN 00 AND 10 O'CLOCK
# Because we are making predictions between 00h and 09h am
df_hour0=df_traffic_raw[ df_traffic_raw.Hour <=9]
df_hour=df_hour0[df_traffic_raw.Hour >=7]
# %%
#df_hour0

# %%
df_hour

# %%
## DELETE WEEKENDS' RAW
df_week=df_hour[df_hour.days <=4.0]
df_week=df_week.loc[237: , ] 
#df_week.drop(1153)
# %%
df_week

# %%
#---------------A SMALLER DATAFRAME INDEXED-------------
#--------------Indexed the dataframe df_week-------------
# %%
indexed_df_week = df_week.set_index(['date_time']) # change the index of the dataframe as date_time

# %%
indexed_df_week # only between monday and friday
# %%
#indexed_df
# %%






#----------------------------FIGURES -------------------------------
##--------HELP TO SEE THE DIFFERENCE IN THE VARIATION OF THE--------
##---------------NUMBER OF BIKES BETWEEN MONTHS, DAYS,...---------------
# %%
# Choosing the colors
sns.set_palette("colorblind", n_colors=7)
indexed_df['days'] = indexed_df.index.weekday  # Monday=0, Sunday=6

Days = ['Monday', 'Tuesday', 'Wednesday',
        'Thursday', 'Friday', 'Saturday', 'Sunday']

traffic_week = indexed_df.groupby(['days', indexed_df.index.hour])[
    'Today\'s_total'].count().unstack(level=0) 

fig, axes = plt.subplots(1, 1, figsize=(7, 6))


traffic_week.plot(ax=axes)
axes.set_ylabel("Variation")
axes.set_xlabel("Time of day")
axes.set_title(
    "Variation of the number of bikes during the day in a week ")
axes.set_xticks(np.arange(0, 24))
axes.set_xticklabels(np.arange(0, 24), rotation=45)
axes.set_ylim(0, 35)
axes.legend(labels=Days, loc='lower left', bbox_to_anchor=(1, 0.1))

plt.tight_layout()


########################### COMMENTS ########################################
# We can notice that the variation is completly different 
# during the weekend, it's lower 
#####################################################################





#---------------CHANGEMENT OF THE DAY---------24/03/2021-----------------
# %% if i want to have month names in a column

import calendar
indexed_df['month_name'] = indexed_df.index.month  # Janvier=0, .... Decembre=11
indexed_df['month_name'] = indexed_df['month_name'].apply(lambda x: calendar.month_abbr[x])
indexed_df.tail()


# %%
indexed_df


# %%
variation = indexed_df.groupby(['month_name', indexed_df.index.hour])[
    'Today\'s_total'].mean().unstack(level=0)


sns.set_palette("colorblind", n_colors=12)

fig, axes = plt.subplots(1, 1, figsize=(7, 5), sharex=True)

variation.plot(ax=axes)

axes.set_ylabel("Number of bikes")
axes.set_xlabel("Hours of the day")
axes.set_title(
    "Evolution of bikes' number by hour monthly")
axes.set_xticks(np.arange(0, 24))
axes.set_xticklabels(np.arange(0, 24), rotation=45)
axes.set_ylim(0, 2300)

axes.legend().set_visible(True)
# ax.legend()
axes.legend(labels=calendar.month_name[1:], loc='lower left', bbox_to_anchor=(1, 0.1))

plt.tight_layout()
########################### COMMENTS ###############################################
# We can notice here that January, February,November are a little
# different from others
########################################################################

# %% 
# Not very usefull
ax = indexed_df['2020':].resample('M').mean().plot(figsize=(7,7)) 
plt.ylim(0,20)
plt.title("Evolution by months")
plt.ylabel("Variation")
plt.xlabel("Months")

# %%

ax = indexed_df['2020':].resample('M').mean().plot(figsize=(7,7)) 
plt.ylim(0,1300)
plt.title("Monthly Variation ")
plt.ylabel("Variation")
plt.xlabel("Months")

# %%









# %%

#------------------------ARIMA MODEL------------------------------
##-------------------applied on indexed_df-------------------------
###----after studing this model we can notive that the------------
###-----prediction  is not 'good'(in our context) because-------------- 
# ------------------------there are too----------------------------
###----much information in the indexed_df dataframe------------------
# %%
from statsmodels.tsa.stattools import adfuller
def ad_test(dataset):
    dftest = adfuller(dataset,autolag= 'AIC')
    print("1. ADF :", dftest[0])
    print("2. p-value :", dftest[1])
    print("3. Num Of Lags :", dftest[2])
    print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
    print("5. Critical values :")
    for key, val in dftest[4].items():
        print("\t", key,": ", val)

# %%
ad_test(indexed_df['Today\'s_total'])
# the p-value is cool: less than 0.05
# %%
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

# %%
stepwise_fit = auto_arima(indexed_df['Today\'s_total'],trace= True, suppress_warnings = True)
stepwise_fit.summary()
### the (p,d,q)=(1,1,2) and there is no seasonality
# %%
from statsmodels.tsa.arima_model import ARIMA

# %%
## DIVIDE THE DATAFRAME INTO 2 DATAFRAMES
print(indexed_df.shape)
train= indexed_df.iloc[:-30]
test=indexed_df.iloc[-30:]
print(train.shape, test.shape)

# %%
# USE THE DATAFRAME train TO MAKE PREDICTION ON THE DATAFRAME test
model = ARIMA(train['Today\'s_total'],order = (1,1,2))
model = model.fit()
model.summary()

# %%
start = len(train)
end =len(train) + len(test) -1
pred = model.predict(start=start, end=end, typ='levels')
pred.index = indexed_df.index[start : end +1]
print(pred)
# the prediction doesn't seem to be good because in a day 
# we can have different values but we will adapt it to our context

# %%
pred.plot(legend= True)
test['Today\'s_total'].plot(legend =True)
# the prediction doesn't seem to be good because in a day 
# we can have different values but we will adapt it to our context

# %%
test['Today\'s_total'].mean()
# Conclusion : the error is too big so the prediction is not good

# %%
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse =sqrt(mean_squared_error(pred,test['Today\'s_total']))
print(rmse)
# Conclusion : the error is too big so the prediction is not good
# but the interpretation can change according to the goal and
# the context
# %%
# prediction for the future by the big dataframe
model2=ARIMA(indexed_df['Today\'s_total'],order=(1,1,2))
model2=model2.fit()
indexed_df.tail()

# %%
index_future_dates = pd.date_range(start='2021-04-01',end='2021-05-01')
pred = model2.predict(start=len(indexed_df),end=len(indexed_df)+30,typ='levels')#.remame('ARIMA Predictions')
pred.index=index_future_dates
print(pred[0])
########################## COMMENTs #########################################
# As we said, the model of prediction is good but we have to
#  change the dataframe with less information
###################################################################

# %%












# %%
###----so let's try the ARIMA model on a smaller dataframe------------
#---------------------ARIMA MODEL ON df_week------------------------
#------------days between monday and friday and hour between--------
#-------------------------07h and 09h59--------------------------

# %%
from statsmodels.tsa.stattools import adfuller
def ad_test(dataset):
    dftest = adfuller(dataset,autolag= 'AIC')
    print("1. ADF :", dftest[0])
    print("2. p-value :", dftest[1])
    print("3. Num Of Lags :", dftest[2])
    print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
    print("5. Critical values :")
    for key, val in dftest[4].items():
        print("\t", key,": ", val)

# %%
ad_test(indexed_df_week['Today\'s_total'])
## the p-value is good , it's smaller than 0.05 so the model is 
#stationnary and this prediction is effective
# %%
from pmdarima import auto_arima
import warnings # Ignores harmless warnings
warnings.filterwarnings('ignore')

# %%
stepwise_fit = auto_arima(indexed_df_week['Today\'s_total'],trace= True, suppress_warnings = True)
stepwise_fit.summary()
#### (p,d,q)=(1,0,1) # the integretad coef is none

# %%
from statsmodels.tsa.arima_model import ARIMA

# %%
## DIVIDE THE DATAFRAME INTO 2 DATAFRAMES
print(indexed_df_week.shape)
train= indexed_df_week.iloc[:-15]
test=df_week.iloc[-15:]
print(train.shape, test.shape)

# %%
# USE THE DATAFRAME train TO MAKE PREDICTION ON THE DATAFRAME test
model = ARIMA(train['Today\'s_total'],order = (1,0,1))
model = model.fit()
model.summary()

# %%
start = len(train)
end =len(train) + len(test) -1
pred = model.predict(start=start, end=end, typ='levels')
pred.index = indexed_df_week.index[start : end +1]
print(pred)
# NB:the model of prediction is effective but this result can be 
# explained by the different values in the same day


# %%
test['Today\'s_total'].mean()


# %%
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse =sqrt(mean_squared_error(pred,test['Today\'s_total']))
print(rmse)
# NB:the model of prediction is effective but this result can be 
# explained by the different values in the same day

# %%
# prediction for the future by the big dataframe
model2=ARIMA(df_week['Today\'s_total'],order=(1,0,1))
model2=model2.fit()
indexed_df.tail(10)

# %%
index_future_dates = pd.date_range(start='2021-04-01',end='2021-05-01')
pred = model2.predict(start=len(df_week),end=len(df_week)+30,typ='levels')#.remame('ARIMA Predictions')
pred.index=index_future_dates
print("The prediction of the number of bikes at Friday April 2, 2021 is: ",pred[0]) # print(pred)
## HERE IS THE PREDICTION FOR FRIDAY April 2 ,2021 :)

# %%




























########################################################################
#---------------------------UNUSED PART-----------------------------
########################################################################
#---------ARIMA MODEL----ROLLING STATISTIC---ON indexed_df_week-------
#--------If our dataset was not stationnary, we could use--------------- 
#---------the Rolling-Statistics test to get something---------------------
#-------------------------more stationary--------------------------
#----------------But the AD-FULLER test was enought--------------------------
# %%
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
rolling_mean = indexed_df_week['Today\'s_total'].rolling(window =12).mean()
rolling_std = indexed_df_week['Today\'s_total'].rolling(window = 12).std()

# %%
plt.plot(indexed_df_week['Today\'s_total'], color='blue', label='Original')
plt.plot(rolling_mean, color ='red', label='Rolling_Mean')
plt.plot(rolling_std, color= 'black' , label='Rolling_Std')
plt.legend(loc ='best')
plt.title('Rolling Mean ansd Rolling Standard Deviation')
plt.show()
# we can take the same conclusion basing on the AD Fuller test
# %%
indexed_df_week_log= np.log(indexed_df_week['Today\'s_total'])
plt.plot(indexed_df_week_log)

# %%
def get_stationnary(timeseries):
    #rolling statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()

    #rolling statistics plot
    original = plt.plot(timeseries,color='blue', label='Original')
    mean = plt.plot(rolling_mean, color ='red', label='Rolling_Mean')
    std= plt.plot(rolling_std, color= 'black' , label='Rolling_Std')
    plt.legend(loc ='best')
    plt.title('Rolling Mean ansd Rolling Standard Deviation')
    plt.show(block= False)



# %%
## TRING TO TRANSFORM OUR TIME SERIE TO SOMETHING MORE STATIONNARY
rolling_mean = indexed_df_week_log.rolling(window=12).mean()
df_log_minus_mean = indexed_df_week_log - rolling_mean # the shift method
#df_log_minus_mean = indexed_df_week_log - indexed_df_week_log.shift()
df_log_minus_mean.dropna(inplace=True)
# the log method

# %%
get_stationnary(df_log_minus_mean)

# %%
#df_log_minus_mean.dropna()
ad_test(df_log_minus_mean)


# %%
df_log_minus_mean.tail(30)




