## IMPORT LIBRARIES
import streamlit as st
# from sidebar import mysidebar


## -------------------------------------------------------------------------
##  PAGE CONFIGURATION & SETUP
## -------------------------------------------------------------------------

## SET PAGE CONFIGURATION(S)
st.set_page_config(layout="wide", page_title='Forecasting Time Series Data')

## GET STYLE GUIDES
# with open ("style.css" ) as css:
#         st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)
#         st.write(f'<style>{css.read()}</style>', unsafe_allow_html=True)

## SET SIDEBAR
# mysidebar()


## -------------------------------------------------------------------------
##  THIS IS THE MAIN PAGE
## -------------------------------------------------------------------------
     
# Cusomtize Main Page
st.title('Seasonal Autoregressive Integrated Moving Average with Exogenous Variables') 

st.write('I hope by now, you’re familiar with the basic concepts of ARIMA models, \
         whether they are seasonal and/or have exogenous variables. If not, it’s \
         worth re-reading, the other models to get a grasp on the fundamental \
        components of ARIMA modeling. With that said, a Seasonal Autoregressive \
        Integrated Moving Average with Exogenous Variables (SARIMAX) model is an \
        improved version of the SARIMA model, just like a ARIMAX is the jelly \
        to the ARIMA peanut butter, that allows for forecasting seasonal data. \
        Just like a ARIMAX model, a SARIMAX can deal with external (or exogenous \
        variables).')
        

with st.expander('Read More'):        
    st.subheader('What are the components of SARIMAX?')
    
    markdown = """
    The components are very similar to that of SARIMA, where there is an autoregressive \
    (AR), Integrated (I), Moving-Average (MA), and seasonal (S) component. \
    In addition to these SARIMAX models include the addition of explanatory \
    variables, also called exogenous variables (X).
    
    - **Autoregression** - refers to a model that shows a variables regresses \
        on it own lagged values
    - **Integrated** - represents the difference of raw obsersvations to allow \
        for the time series to become stationary (i.e. values are replaced by \
            the difference between the current values and the prior values)
    - **Moving Average** - incorporates the dependency between observations and \
        residual errors from a moving average model applied to lagged observations
    - **Seaonal Factor** - refers to the cyclical or recurring patterns within the \
        endogenous variable 
    - **Exogenous Variables** - these are the additional explanatory variables \
        which help define the relationship between exogenous and endogenous \
        allowing it to be discovered much quicker
        
    So what does this look like?
    """
    
    st.markdown(markdown)
       
        
    st.write('')
    st.subheader('SARIMAX Parameters')    
    
    st.write('Just like ARIMA(X) models, there are three main components: p, d, and q. \
             Now that we are adjusting for seasonality though, we have to begin \
            configuring the seasonality which requires four new parameters: ')
    
    markdown = """
    - **p**: the number of lag observations in the model
    - **d**: the number of times the raw observations are differenced 
    - **q**: the size of the averaged window
    
    - **P**: the number of lag observations of the seasonal component of the model
    - **D**: the number of times the seasonal component is differenced 
    - **Q**: the size of the averaged window for the seasonal component (
    - **m**: the number of time steps for a single seasonal period 

    """

    st.markdown(markdown)
    
                                          
st.markdown('---')
