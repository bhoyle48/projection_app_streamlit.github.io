## IMPORT LIBRARIES
import streamlit as st
from sidebar import mysidebar                                                     


## -------------------------------------------------------------------------
##  PAGE CONFIGURATION & SETUP
## -------------------------------------------------------------------------

## SET PAGE CONFIGURATION(S)
st.set_page_config(layout="wide", page_title='Forecasting Time Series Data')

## GET STYLE GUIDES                                                                 
with open ("style.css" ) as css:
         st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)
         st.write(f'<style>{css.read()}</style>', unsafe_allow_html=True)

## SET SIDEBAR
mysidebar()                                                                       


## -------------------------------------------------------------------------
##  ARIMA BLOG CONTENT
## -------------------------------------------------------------------------
     
# Cusomtize Main Page
st.title('Autoregressive Integrated Moving Average')

st.write('An autoregressive integrated moving average, is a statistical model \
        which uses time series data to help predict future values. For example, \
        an ARIMA model might seek to predict a stock’s future price based on it \
        past performance. ARIMA models are a form of regression analysis that gauges \
        the strength of one dependent variable relative to other changing variables. \
        The model’s goal is to project future values by examining the difference \
        between values in the series instead of other independent features.')
        
st.write("---")

## -------------------------------------------------------------------------
##  IMPORTS
## -------------------------------------------------------------------------
import pandas as pd

import plotly.graph_objects as go

from statsmodels.tsa.stattools import adfuller

## -------------------------------------------------------------------------
##  FORM & FILE
## -------------------------------------------------------------------------

form = st.form(key='arima-params', clear_on_submit=False)

with form:    
    
    ## REQUIRED INFORMATION
    rcol1, rcol2, rcol3 = st.columns([6, 2.9, 1.1])
    
    with rcol1:
        # Add File Component
        uploaded_file = st.file_uploader(label='Upload your Timeseries Data', type='csv', accept_multiple_files=False)
    
        # If there is not a file uploaded
        if uploaded_file is not None:
            if 'load_csv' is st.session_state:
                df = pd.read_csv('sample-data/BTC-USD.csv')             
                df = st.session_state.load_csv
                st.write(uploaded_file.name + " is loaded")
            else:
                df = pd.read_csv(uploaded_file)
                st.session_state.load_csv = df
                st.write(uploaded_file.name + " is loaded")
        else:
            df = pd.read_csv('sample-data/BTC-USD.csv') 
            st.session_state.load_csv = df 
            st.write("BTC-USD.csv is loaded")
        
        ## Get Length of Dataset
        nrows = len(df)
    
    with rcol2:
        st.write('') # Added for alignment
        st.write('') # Added for alignment
        n_periods_to_forecast = st.slider(label = 'Set Number of Periods to Forecast', 
                        min_value=int(nrows/10), 
                        max_value=int(nrows/2),
                        value=int(nrows/4),
                        step=1,
                        key='periods_to_forecast', 
                        help='Enter how many periods to forecast in the same date type (i.e. days, months, etc.). \
                             Minimum = 10% of provided data size. Maximum = 50%. Default = 25%')
                             
    with rcol3:
        st.write('') # Added for alignment
        st.write('') # Added for alignment
        date_type = st.selectbox('Set Period Type',('Day', 'Week', 'Month', 'Quarter', 'Year'),index = 0)
    
    
    ## MODEL CRITIERA
    
    with st.expander('Model Parameters'):
        
        mcol1, mcol2, mcol3 = st.columns([5,2.5,2.5])
        
        with mcol1:
            
            # Get all values for ARIMA (p, d, q) - if value is -1, tune it automatically
            p = st.number_input(label = 'Set Lag Order (p)', 
                            min_value=-1, 
                            max_value=15,
                            value= -1,
                            step=1,
                            key='setp', 
                            help='Set Lag Order, or -1 to optimize this variable')
        
            d = st.number_input(label = 'Set Degree of Differening (d)', 
                            min_value=-1, 
                            max_value=15,
                            value= -1,
                            step=1,
                            key='setd', 
                            help='Set Degree of Differencing, or -1 to optimize this variable')
        
            q = st.number_input(label = 'Set Order of Moving Average (q)', 
                            min_value=-1, 
                            max_value=15,
                            value= -1,
                        step=1,
                        key='setq', 
                        help='Set Order of Moving Average, or -1 to optimize this variable')
    
        with mcol2:
            
            # Set test size, and simulations for modeling (set defaults)
            test_split_size = st.select_slider('Set Test Size',
                        options=['10%','15%','20%','25%','30%','35%','40%','45%','50%'],
                        value = '30%',
                        help = 'How much of the data would you like to use to train your model? \
                                The more data provided in training allows the model to better fit, \
                                however at a potential of overfitting and inability to properly test \
                                the created model. Default = 30%',
                        key = 'test_size')
        
            simulations = st.slider(label = 'Set Simulations Count',
                        min_value = 100,
                        max_value = 10000,
                        value = 1000,
                        step = 100,
                        key = 'simulations',
                        help = 'How many simulations of the model would you like to run? \
                                The higher the value more accurate the confidence range \
                                but at a greater time cost. Default = 1000')

       
        with mcol3:
            
            # Set upper and lower confidence range. Different values allow for reporting risk adverseness
            lower_conf = st.select_slider('Set Lower Confidence Interval',
                        options=['50%','60%', '65%','70%','75%', '80%','85%','90%','95%','99.5%'],
                        value = '90%',
                        key = 'lower_conf')
            upper_conf = st.select_slider('Set Upper Confidence Interval',
                        options=['50%','60%', '65%','70%','75%', '80%','85%','90%','95%','99.5%'],
                        value = '90%',
                        key = 'upper_conf')


    
    # End of Form -- Submit buttom for running model      
    st.form_submit_button(label='Run ARIMA Model')
    
## -------------------------------------------------------------------------
##  CLEAN DATAFRAME
## -------------------------------------------------------------------------

# Keep only first two columns
df = df.iloc[:, :2]

# Get name of first column
metric_name = df.columns[1]

# Rename columns
df = df.rename(columns={df.columns[0]: 'Date', df.columns[1]: 'Metric'})

# Convert the Date column to Datetime
df['Date'] =  df['Date'].astype('datetime64[ns]')


## -------------------------------------------------------------------------
##  TEST FOR TREND TYPE (Multiplicative or Additive)
## -------------------------------------------------------------------------
import statsmodels.api as sm

## Get Period Type as Integer (as a function of days)
time_period_dict    = {"Day" : 1, "Week" : 7, "Month": 365.25/12, "Quarter" : 365.25/4, "Year": 365.25}
time_period_window  = time_period_dict.get(date_type)


# Set Date as index
df = df.set_index('Date')

# Trend
df['Trend']=[0]*(time_period_window-1)+list(pd.Series(df['Metric']).rolling(window=time_period_window).mean().iloc[time_period_window-1:].values)

# De-trend data
df['detrended_a']=df['Metric']-df['Trend']
df['detrended_m']=df['Metric']/df['Trend']

# Seasonals
df['seasonal_a']=[0]*(time_period_window-1)+list(pd.Series(df['detrended_a']).rolling(window=time_period_window).mean().iloc[time_period_window-1:].values)
df['seasonal_m']=[0]*(time_period_window-1)+list(pd.Series(df['detrended_m']).rolling(window=time_period_window).mean().iloc[time_period_window-1:].values)

# Residuals
df['residual_a']=df['detrended_a'] - df['seasonal_a']
df['residual_m']=df['detrended_m'] / df['seasonal_m']

# Auto-Correlation Factor
acf_a = sum(pd.Series(sm.tsa.acf(df['residual_a'])).fillna(0))
acf_m= sum(pd.Series(sm.tsa.acf(df['residual_m'])).fillna(0))

if acf_a > acf_m:
    trend_model_type = 'additive'
else:
    trend_model_type = 'multiplicative'
    
## -------------------------------------------------------------------------
##  SEASONAL DECOMPOSE
## -------------------------------------------------------------------------
from statsmodels.tsa.seasonal import STL

plot_layout = go.Layout(
    margin=go.layout.Margin(
        l=0, #left margin
        r=0, #right margin
        b=0, #bottom margin
        t=0, #top margin
    ))


st.markdown('---')

# Period is required in terms of years, such that a year = 1, month = 12, etc.
decomposition = STL(df['Metric'], robust = True, period=int(365/time_period_window)
                    , seasonal=int(365/time_period_window)).fit()


sd1, sd2 = st.columns([7,3])

with sd1:
    st.subheader('Time Series Decomposition')
    st.markdown(
        """
        Displayed below are the actual values present within the dataset provided. 
        To the right, three plot decompose the time series into the trend, seasonal, and residual components.
        """
        )
    
    obs = go.Figure(data=go.Scatter(x = df.index, y=decomposition.observed, 
                                    mode='lines', name='Trend', marker_color='#0079c2'), layout=plot_layout)
    obs.update_layout(width=750, height=370, title='Actual Values ['+metric_name+']', margin={'t':30}, title_x=0.5, showlegend=False
                      ,xaxis = {'showgrid': False} ,yaxis = {'showgrid': False})
    
    st.plotly_chart(obs)

with sd2:
    ## TREND
    tre = go.Figure(data=go.Scatter(x = df.index, y=decomposition.trend, 
                                    mode='lines', name='Trend', marker_color='#0079c2'), layout=plot_layout)
    tre.update_layout(width=300, height=150, title='Trend', margin={'t':30}, title_x=0.5, showlegend=False
                      ,xaxis = {'showgrid': False} ,yaxis = {'showgrid': False})
    
    st.plotly_chart(tre)
    
    ## SEASONALITY
    sea = go.Figure(data=go.Scatter(x = df.index, y=decomposition.seasonal, 
                                    mode='lines', name='Seasonal', marker_color='#0079c2'), layout=plot_layout)
    sea.update_layout(width=300, height=150, title='Seasonal', margin={'t':30}, title_x=0.5, showlegend=False
                      ,xaxis = {'showgrid': False} ,yaxis = {'showgrid': False})
    
    st.plotly_chart(sea)
    
    ## RESIDUAL
    res = go.Figure(data=go.Scatter(x = df.index, y=decomposition.resid, 
                                    mode='lines', name='Residual', marker_color='#0079c2'), layout=plot_layout)
    res.update_layout(width=300, height=150, title='Residual', margin={'t':30}, title_x=0.5, showlegend=False
                      ,xaxis = {'showgrid': False} ,yaxis = {'showgrid': False})
    
    st.plotly_chart(res)
    

## -------------------------------------------------------------------------
##  STATIONARITY
## -------------------------------------------------------------------------

st.markdown('---')
st.subheader('Checking for Stationarity')


st1, st2 = st.columns([4,6])

with st1:

    st.write('<p style="font-size:24px", font>Augmented Dickey-Fuller (ADF)</p>', unsafe_allow_html=True)

        
with st2:
    st.write('<p style="font-size:24px", font>Auto-Correlation Plots (ACF/PACF)</p>', unsafe_allow_html=True)
        

## -------------------------------------------------------------------------
    
df1, df2, cf1, cf2 = st.columns([1.9,1.9, 3.1,3.1])

## -------------------------------------------------------------------------
##  DICKEY-FULLER TESTS
## -------------------------------------------------------------------------
from itertools import islice   


def ADF_Stationarity_Test(timeseries):
        #Dickey-Fuller test:
        adfTest = adfuller(timeseries, autolag='AIC')
        
        pValue = adfTest[1]
        
        isStationary = True if pValue<0.05 else False
        
        first, second, third = islice(adfTest[4].values(), 3)    

        st.markdown("""
                   - ADF Test Statistic: &nbsp;&nbsp;**{}**
                   - Critical Values:
                     - 1%: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**{}**
                     - 5%: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**{}**
                     - 10%: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**{}**
                   - P-Value: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**{}**
                    """.format(
                    round(adfTest[0],2),
                    round(first,2),
                    round(second,2),
                    round(third,2),
                    round(adfTest[1],2)
                    )
        )
                
        st.write("Is the time series stationary? *_**{}**_*".format(isStationary) )  
        
        return isStationary
    


with df1:
     
    st.write('<p style="font-size:18px", font>ADF Results (Original):</p>', unsafe_allow_html=True)
    ADF_Stationarity_Test(df['Metric'])
        
with df2:
    from scipy import stats
    
    st.write('<p style="font-size:18px", font>ADF Results (Boxcox-ed):</p>', unsafe_allow_html=True)
    
    xt, _ = stats.boxcox(df['Metric'])
    
    ADF_Stationarity_Test(xt)
    
   
## -------------------------------------------------------------------------
##  ACF/PACF GRAPHS 
## -------------------------------------------------------------------------
from statsmodels.tsa.stattools import pacf, acf
import numpy as np

plot_layout = go.Layout(
    margin=go.layout.Margin(
         l=0  #left margin
        ,r=0  #right margin
        ,b=0  #bottom margin
        ,t=0  #top margin
    )
    ,xaxis = {'showgrid': False}
    ,yaxis = {'showgrid': False}
    #,paper_bgcolor='#b6bec2'
    #,plot_bgcolor='#b6bec2'
    )



with cf1:
    ## ACF PLOT
    corr_array = acf(df['Metric'], alpha=0.05)
    lower_y = corr_array[1][:,0] - corr_array[0]
    upper_y = corr_array[1][:,1] - corr_array[0]

    fig = go.Figure(layout=plot_layout)
    
    for x in range(len(corr_array[0])):
        fig.add_scatter(x=(x,x), y=(0,corr_array[0][x]), mode='lines',line_color='#090909')
    
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=corr_array[0], mode='markers', marker_color='#0079c2', marker_size=6)
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=upper_y, mode='lines', line_color='rgba(255,255,255,0)')
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=lower_y, mode='lines', fillcolor='rgba(0,121,194,0.3)', fill='tonexty', line_color='rgba(255,255,255,0)')
    
    fig.update_traces(showlegend=False)
    fig.update_xaxes(range=[-1,len(corr_array[0])], zerolinecolor='rgba(0,0,0,0)')
    fig.update_yaxes(zerolinecolor='#090909')
    
    fig.update_layout(title='Autocorrelation (ACF)', width=325, height=275, margin={'t':30}, title_x=0.5, showlegend=False
                      ,xaxis = {'showgrid': False} ,yaxis = {'showgrid': False})
    
    st.plotly_chart(fig)

with cf2: 
     
    ## PACF PLOT
    corr_array = pacf(df['Metric'], alpha=0.05)
    lower_y = corr_array[1][:,0] - corr_array[0]
    upper_y = corr_array[1][:,1] - corr_array[0]

    fig = go.Figure(layout=plot_layout)
    
    for x in range(len(corr_array[0])):
        fig.add_scatter(x=(x,x), y=(0,corr_array[0][x]), mode='lines',line_color='#090909')
    
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=corr_array[0], mode='markers', marker_color='#0079c2', marker_size=6)
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=upper_y, mode='lines', line_color='rgba(255,255,255,0)')
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=lower_y, mode='lines', fillcolor='rgba(0,121,194,0.3)', fill='tonexty', line_color='rgba(255,255,255,0)')
    
    fig.update_traces(showlegend=False)
    fig.update_xaxes(range=[-1,len(corr_array[0])], zerolinecolor='rgba(0,0,0,0)')
    fig.update_yaxes(zerolinecolor='#090909')
    
    fig.update_layout(title='Partial Autocorrelation (PACF)', width=325, height=275, margin={'t':30}, title_x=0.5, showlegend=False
                      ,xaxis = {'showgrid': False} ,yaxis = {'showgrid': False})
    
    st.plotly_chart(fig)
    
    

    
## -------------------------------------------------------------------------
##  BRUESH-PAGAN TEST 
## -------------------------------------------------------------------------
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.linear_model import LinearRegression

# Convert metric into a 2D array - required input for Bruesh-Pagan Test
def test_model(col):
    s = []
    for i in col:
        a = [1,i]
        s.append(a)
    return (np.array(s))


## LINEAR REGRESSION MODEL

import datetime as dt

# Get Date back into the Dataframe
df['Date'] = df.index

# Intialize DataFrame for simple OLS Model
ols_df = pd.DataFrame()

## Extract extra values from Date and Convert Date to Ordinal values
ols_df['Date'] = pd.to_datetime(df['Date'])

ols_df['MonthNum'] = ols_df['Date'].dt.month
ols_df['Year'] = ols_df['Date'].dt.year

ols_df['Weekday'] = ols_df['Date'].dt.weekday
ols_df['DayOfMonth'] = ols_df['Date'].dt.day
ols_df['DayOfYear'] = ols_df['Date'].dt.dayofyear

ols_df['Date'] = ols_df['Date'].map(dt.datetime.toordinal)

## Add column for Metric to OLS Dataframe
ols_df['Metric'] = df['Metric']

## Create X and Y in the correct shape
x = ols_df.drop(columns='Metric')
y = ols_df['Metric']

# Initialize, Fit and Predict using the OLS Model
ols_model = LinearRegression()
ols_model.fit(x,y)
y_pred = ols_model.predict(x)

# Get resiauals
residuals = y_pred - y



## Breush-Pagan Test
    # Output: 'Lagrange Multiplier Statistic', 'P-Value', 'F-Value', 'F P-Value'
df_array = test_model(df['Metric'])
bp_test = het_breuschpagan(residuals, df_array)

lm      = bp_test[0]
pval    = bp_test[1]
fval    = bp_test[2]
fpval   = bp_test[3]



st.markdown('---')
st.subheader('Checking for Heteroscedasticity')

## -------------------------------------------------------------------------
    
he1, he2 = st.columns([3.8, 6.2])

with he1:
    st.write('<p style="font-size:18px", font>Breush-Pagan Test</p>', unsafe_allow_html=True)

    st.markdown(
        """
        - LM Statistic: **{}**                  
        - P-Value:   **{}**                   
        - F  Statisitic: **{}**                   
        - F P-Value: **{}**                    
    
        Rejects the null hypothesis of homoscedasticity using a 95% confidence level? 
        \r*_**{}**_*
        """.format(
                    round(lm,2),
                    round(pval,5),
                    round(fval,2),
                    round(fpval,5),
                    pval > 0.05
        ))
    
with he2:
 
    res.update_layout(width=665, height=350, title='Residuals Plot', margin={'t':45}, title_x=0.501, showlegend=False,font=dict(size=14)
                      ,xaxis = {'showgrid': False} ,yaxis = {'showgrid': False})
    
    st.plotly_chart(res)


## -------------------------------------------------------------------------
##  TRAIN TEST SPLIT
## ------------------------------------------------------------------------- 
X = df['Date']
y = df['Metric']

## Convert String input to decimal
test_split_size = int(test_split_size.replace('%', ''))/100

train_split_size = 1-test_split_size

## Get train size in relation to list size
train_size = int(len(df) * train_split_size)
test_size  = int(len(df) - train_size)


## Split Data
X_train, X_test = X[0:train_size], X[train_size:]
y_train, y_test = y[0:train_size], y[train_size:]



## -------------------------------------------------------------------------
##  ARIMA MODEL INITIALIZATION AND FITTING
## ------------------------------------------------------------------------- 
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

import warnings
from math import sqrt
warnings.filterwarnings("ignore")

st.markdown('---')
st.subheader('Model Selection and Fitting')

best_score, best_cfg = float("inf"), None

my_bar = st.progress(0)


## P
if p == -1:
    ps = (0,1,2,4,6)
else: 
    ps = range(p,p+1)

## D
if d == -1:
    ds = (0,1,2)
else: 
    ds = range(d,d+1)

## Q
if q == -1:
    qs = (0,1,2)
else:
    qs = range(q,q+1)


iterrations = (100 / (len(ps)*len(ds)*len(qs)))/100
progress_point = 0

for p in ps:
    for d in ds:
        for q in qs:
            param_order = (p, d, q)
                        
            predictions = list()
                  
            model = ARIMA(endog = y_train.values, order = (p,d,q), enforce_stationarity=True).fit()
				
            yhat = model.forecast(steps = test_size)
                       
            rmse = sqrt(mean_squared_error(y_test, yhat))
                
            if rmse < best_score:
                best_score, best_cfg, best_predictions = rmse, param_order, yhat
				
            
            progress_point = progress_point + iterrations if progress_point + iterrations <= 1 else 1

            my_bar.progress(progress_point)
			

# Get the best model
best_arima_model = ARIMA(endog = y_train, order = best_cfg).fit()

p = best_cfg[0]
d = best_cfg[1]
q = best_cfg[2]



## -------------------------------------------------------------------------
##  SIMULATE ARIMA MODEL
## ------------------------------------------------------------------------- 
import scipy.stats as stat
import numpy as np

bam = ARIMA(endog = np.array(df['Metric']), dates=np.array(df.index), order = best_cfg)

bm = bam.fit()

sim = bm.simulate(nsimulations = n_periods_to_forecast, 
                   anchor='end', 
                   repetitions = simulations,
                   )

sim = pd.DataFrame(np.concatenate(sim))


# Upper Confidence Level
upper_confidence = int(upper_conf.replace('%', ''))/100

# Lower Confidence Level 
lower_confidence = int(lower_conf.replace('%', ''))/100

# Get Projections into DataFrame
sim2 = pd.DataFrame()
sim2['conf_upper'] = round(sim.mean(axis=1) + (stat.norm.ppf(upper_confidence)*sim.std(axis=1)))
sim2['conf_mean']  = round(sim.mean(axis=1))
sim2['conf_lower'] = round(sim.mean(axis=1) - (stat.norm.ppf(lower_confidence)*sim.std(axis=1)))

sim2.loc['Column_Total'] = sim2.sum(numeric_only=True, axis=0)

### Results Table Initialization
results = pd.DataFrame(index=["SSE", "SSR", "SST", "p", "d", "q"])

## Set parameters to gather
SSE = round(bm.sse,0)
SSR = round(((bm.fittedvalues - df.Metric.mean())**2).sum(),0)
SST = round(SSE + SSR,0)
p = round(p,0)
d = round(d,0)
q = round(q,0)

results['Best Model'] = [SSE] + [SSR] + [SST] + [p] + [d] + [q]
results['Best Model'] = results['Best Model'].astype(float).round(2)



## -------------------------------------------------------------------------
##  CREATE FINAL DATAFRAME OF PROJECTIONS
## ------------------------------------------------------------------------- 
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

tqdm.pandas()

def add_periods(start_date, delta_period, unit):
    if unit == 'year':
        end_date = start_date + relativedelta(years=delta_period)
    elif unit == 'quarter':
        end_date = start_date + relativedelta(months=delta_period*3)
    elif unit == 'month':
        end_date = start_date + relativedelta(months=delta_period)
    elif unit == 'week':
        end_date = start_date + relativedelta(weeks=delta_period)
    elif unit == 'day':
        end_date = start_date + relativedelta(days=delta_period)
    return end_date

## Get max date from dataframe
max_date = df.Date.max()

# Reset Index (periods to add), and create column with the max date in the original dataframe
sim2dates = pd.DataFrame(sim2.index)
sim2dates['max_date'] = df.Date.max()

# Rename columns, and drop last row (column totals)
sim2dates.columns = ['num_periods', 'max_date']
sim2dates = sim2dates[:-1]

# Add one to the index (to get periods to add) -- index starts at 0
sim2dates['num_periods'] = sim2dates['num_periods']+1


# Get Projection Dates
if date_type.lower() == 'year':
    sim2dates["projected_date"] = sim2dates.progress_apply(lambda row: add_periods(row['max_date'], row['num_periods'], 'year'), axis = 1)
         
elif date_type.lower() == 'qyarter':
    sim2dates["projected_date"] = sim2dates.progress_apply(lambda row: add_periods(row['max_date'], row['num_periods'], 'quarter'), axis = 1)
         
elif date_type.lower() == 'month':
    sim2dates["projected_date"] = sim2dates.progress_apply(lambda row: add_periods(row['max_date'], row['num_periods'], 'month'), axis = 1)
         
elif date_type.lower() == 'week':
    sim2dates["projected_date"] = sim2dates.progress_apply(lambda row: add_periods(row['max_date'], row['num_periods'], 'week'), axis = 1)
         
elif date_type.lower() == 'day':
    sim2dates["projected_date"] = sim2dates.progress_apply(lambda row: add_periods(row['max_date'], row['num_periods'], 'day'), axis = 1)
         
else:
    print('error')

# Create Final Projection DataFrame(s)    
fpd = pd.concat([sim2dates["projected_date"].reset_index(drop=True), sim2[['conf_mean', 'conf_upper', 'conf_lower']][:-1].reset_index(drop=True)], axis=1, ignore_index=True)
fpd.columns = ['Date', 'mean', 'upper', 'lower']

fpd[['mean', 'upper', 'lower']] = fpd[['mean', 'upper', 'lower']].astype(int)


df = df[['Date', 'Metric']].reset_index(drop=True)

## -------------------------------------------------------------------------
##  FINAL OUTPUT AND GRAPH
## ------------------------------------------------------------------------- 

st.markdown('---')
st.subheader('Output')

with st.expander(label='Model Summary'):
    st.write(bm.summary())


fig = go.Figure(layout=plot_layout)

title='Actual Values ['+metric_name+']',   
fig.add_scatter(x = df['Date'],  y=df['Metric'], mode='lines', name='Actual', marker_color='#0079c2',
                        line = dict(width=4))
fig.add_scatter(x = fpd['Date'], y=fpd['mean'], mode='lines',name='Projected', marker_color='#0079c2',
                        line = dict(width=4, dash='dash'))

fig.add_scatter(x = fpd['Date'], y=fpd['upper'], mode='lines',name='Upper {} CI'.format(upper_conf),
                        line = dict(color='rgba(211, 211, 211, 0.15)', width = 2, dash='dot'))
fig.add_scatter(x = fpd['Date'], y=fpd['lower'], mode='lines',name='Lower {} CI'.format(upper_conf), 
                        line = dict(color='rgba(211, 211, 211, 0.15)', width = 2, dash='dot'))
 
for df1 in fpd:
    fig.add_traces(go.Scatter(x=fpd['Date'], y = fpd['lower'], showlegend=False,
                                  line = dict(color='rgba(211, 211, 211, 0.15)',
                                              width = 2, dash='dot')))
        
    fig.add_traces(go.Scatter(x=fpd['Date'], y = fpd['upper'], showlegend=False,
                                  line = dict(color='rgba(211, 211, 211, 0.15)',
                                              width = 2, dash='dot'),
                                  fill='tonexty', 
                                  fillcolor = 'rgba(211, 211, 211, 0.15)',
                                  ))

fig.update_layout(width=1100, height=400, title=metric_name, margin={'t':30}, title_x=0.5, showlegend=True)
st.plotly_chart(fig)


# Column Names for Final Output
upper = str('Upper {}'.format(upper_conf))
lower = str('Lower {}'.format(lower_conf))

columns = list(['Date', 'Actuals', 'Projected', upper, lower])

with st.expander(label='Final Dataframe'):
    
    out_df = df.append(fpd, ignore_index=True)
    
    out_df.columns = columns
    
    out_df['Date'] = out_df['Date'].dt.strftime('%m/%d/%Y')
    
    st.dataframe(data = out_df)
    

## -------------------------------------------------------------------------
##  DOWNLOAD DATAFRAME
## ------------------------------------------------------------------------- 
    
def convert_df(df):
   return df.to_csv().encode('utf-8')


csv = convert_df(out_df)

st.download_button(
   "Download Final Output",
   csv,
   "ARIMA_projected_{}.csv".format(metric_name),
   "text/csv",
   key='download-csv'
)
