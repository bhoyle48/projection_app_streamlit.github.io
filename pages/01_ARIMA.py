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
st.title('Autoregressive Integrated Moving Average')

st.write('An autoregressive integrated moving average, is a statistical model \
         which uses time series data to help predict future values. For example, \
        an ARIMA model might seek to predict a stock’s future price based \
        on it past performance. ARIMA models are a form of regression \
        analysis that gauges the strength of one dependent variable \
        relative to other changing variables. The model’s goal is to project \
        future values by examining the difference between values in the series \
        instead of other independent features.')
        
                
with st.expander('Read More'):        
    st.subheader('What are the Components of an ARIMA Model?')
    
    markdown = """
    - **Autoregression** - refers to a model that shows a variables regresses \
        on it own lagged values
    - **Integrated** - represents the difference of raw obsersvations to allow \
        for the time series to become stationary (i.e. values are replaced by \
            the difference between the current values and the prior values)
    - **Moving Average** - incorporates the dependency between observations and \
        residual errors from a moving average model applied to lagged observations
    """
    
    st.markdown(markdown)
         
    
    st.subheader('ARIMA Parameters')
    
    markdown = """
    - p: the number of lag observations in the model (i.e. lag order)
    - d: the number of times the raw observations are differenced (i.e. degree of differencing)
    - q: the size of the averaged window (i.e. order of moving average)
    """
    
    st.markdown(markdown)
    
    st.write('')
    st.write('Okay, so what does this actually look like? Suppose we are trying to \
             predict the price of Microsoft Stock (MSFT), defined as P, the simple \
                 ARIMA equation would look as follows:')
    
    formula = r'''
    $$
        \Delta P_t = c+ϕ_1 \Delta P_{t-1}+ θ_1ϵ_{t-1}+ϵ_1
    $$
    '''
              
    st.write(formula)
    
    st.write('Now what does this mean? For starters, $$P_t$$ and $$P_{t-1}$$ represent the \
             values in the current period and the value one period ago, respectively, \
            while the $$E_t$$ and $$E_{t-1}$$ represent the error terms for these periods, \
            and of course $$c$$ is our constant. The two parameters express \
            what parts of the value $$P_{t-1}$$ and error $$ϵ_{t-1}$$ last period \
            are relevant in estimating the current one. Lastly, the \
            $$\Delta P_{t-1}$$ represents the difference between prices in period \
            t and the preceding period - meaning the deltaP is an entire time \
            series, which represents the disparity between prices \
            in consecutive periods.')
            
    st.subheader('ARIMA and Stationarity')
    
    st.write('In an ARIMA model, the observations are difference in order to make \
             it stationary, or that shows constancy to the data over time. Most \
            economic, market, and business models show trends, so the purpose of \
            differencing is to remove any trends or seasonal structures. Seasonal \
            structures however, can cause issues with ARIMA models as the \
            computations for differencing to determine constancy cannot be made \
            effectively, resulting in a less accurate forecast. SARIMA models adjust \
            for seasonality and should be used for time series with regular and \
            predictable patterns (i.e. tax filings, Christmas tree sales, snowfall')
                                          
st.markdown('---')
