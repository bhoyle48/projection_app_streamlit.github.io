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
##  THIS IS THE MAIN PAGE
## -------------------------------------------------------------------------
     
# Cusomtize Main Page
st.title('Forecasting Time Series Data') 

st.write('Time series forecasting is a difficult problem with no easy answer. \
          There are countless forecasting models and methods which claim to \
          outperform each other, yet it is never clear which model is truly best.')
            
st.write('This multi-page app provides a comprehensive, beginner friendly guide to help you \
          understand, deploy and tune your very own forecasts so that you \
          can walk away having the best possible forecast in hand.')
          
st.markdown('---')
          
st.header('Forecasting Models')
st.write('It would be impossible to provide exploration into all possible \
         forecasting methods. For that reason, this will focus on the most popular models.')

markdown = """
1. Autoregressive Integrated Moving Average (__ARIMA__)
2. Seasonal Autoregressive Integrated Moving Average (__SARIMA__) --> *In development*
3. Holt-Winters (HWES) --> *In development*
"""

st.markdown(markdown)


st.write('')
st.write('')
st.write('This is an open-source project created with [Streamlit](https://streamlit.io) \
         and you are very welcome to contribute to the \
         [GitHub repository](https://github.com/bhoyle48/streamlit-projection-app).')

st.markdown('---')

## -------------------------------------------------------------------------
##  Imports
## ------------------------------------------------------------------------- 
import pandas as pd

import plotly.graph_objects as go


## -------------------------------------------------------------------------
##  Functions
## ------------------------------------------------------------------------- 

st.markdown(""" <style> .font {
font-size:px; font-family: 'Montserrat'; color: #FF9633;} 
</style> """, unsafe_allow_html=True)

## -------------------------------------------------------------------------
##  Form
## -------------------------------------------------------------------------      

form = st.form(key='home-params', clear_on_submit=False)


with form:    
    
    ## REQUIRED INFORMATION
    rcol1, rcol2, rcol3 = st.columns([6, 2.9, 1.1])
    
    with rcol1:
        # Add File Component
        uploaded_file = st.file_uploader(label='Upload your Timeseries Data', type='csv', accept_multiple_files=False)
    
        # If there is not a file uploaded
        if uploaded_file is not None:
            if 'load_csv' is st.session_state:
                df = pd.read_csv('sample-data/VTI.csv')             
                df = st.session_state.load_csv
                st.write(uploaded_file.name + " is loaded")
            else:
                df = pd.read_csv(uploaded_file)
                st.session_state.load_csv = df
                st.write(uploaded_file.name + " is loaded")
        else:
            df = pd.read_csv('sample-data/VTI.csv') 
            st.session_state.load_csv = df 
            st.write("VTI.csv is loaded")
        
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
        date_type = st.selectbox('Set Period Type',('Day', 'Week', 'Month', 'Quarter', 'Year'),index = 0, key='date_type')     
 
    
    # End of Form -- Submit buttom for running model      
    st.form_submit_button(label='Analyze Time Series Data')
  
    
    
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
from statsmodels.tsa.stattools import adfuller


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