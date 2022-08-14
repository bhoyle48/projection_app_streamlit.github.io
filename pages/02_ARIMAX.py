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
##  ARIMAX BLOG CONTENT
## -------------------------------------------------------------------------
     
## -------------------------------------------------------------------------
##  ARIMAX BLOG CONTENT
## -------------------------------------------------------------------------
     
# Cusomtize Main Page
st.title('Autoregressive Integrated Moving Average with Exogenous Variables') 

st.write('An Autoregressive Integrated Moving Average with Explanatory Variable \
        (ARIMAX) model can be described as a multiple regression models one or \
        more autoregressive terms, and/or one or more moving averages. Like ARIMA \
        models, this model is suitable for forecasting when data is stationary or \
        non-stationary, but unlike ARIMA, ARIMAX handles scenarios of multi-variability, \
        or where there are additional explanatory variables in categorical or \
        numeric formats.')
        
                
with st.expander('Read More'):        
    st.subheader('What are the Components of an ARIMAX Model?')
    
    markdown = """
    The components are very similar to that of ARIMA, where there is an autoregressive \
    (AR) component, as well an Integrated (I), Moving-Average (MA). See ARIMA model, \
    for more detail on these components. However, in addition to these ARIMAX model \
    includes the additional of these explanatory variables, also called exogenous variables.
    
    - **Autoregression** - refers to a model that shows a variables regresses \
        on it own lagged values
    - **Integrated** - represents the difference of raw obsersvations to allow \
        for the time series to become stationary (i.e. values are replaced by \
            the difference between the current values and the prior values)
    - **Moving Average** - incorporates the dependency between observations and \
        residual errors from a moving average model applied to lagged observations
    - **Exogenous Varibles** - these are the additional explanatory variables which \
        help define the relationship between exogenous and endogenous allowing it to \
        be  discovered much quicker
        
    So what does this look like?
    """
    
    st.markdown(markdown)
       
        
    st.write('')
    st.subheader('ARIMAX Parameters')    
    
    markdown = """
    - p: the number of lag observations in the model (i.e. lag order)
    - d: the number of times the raw observations are differenced (i.e. degree of differencing)
    - q: the size of the averaged window (i.e. order of moving average)
    """
           
    st.markdown(markdown)

    st.write('')
    st.write('&ensp;&emsp; So what does this look like?')

    formula = r'''
    $$
        \Delta P_t = c+βX+ ϕ_1 \Delta P_{t-1}+ θ_1ϵ_{t-1}+ϵ_1
    $$
    '''
       
    st.write(formula)
        
    st.write('Just like ARIMA models, we have the error terms (ϵ), values (P), the \
              constant (C), and the delta. However, you’ll see the inclusion of the \
              X and its coefficient, beta (β). Beta is a coefficient which will be estimated\
              based on the model selection and data and X is the exogenous variable(s) \
              that we are interested in. The exogenous variable can really be anything \
              from a time-varying measurement, to a categorical variable of day of week, \
              or even a boolean accounting for whether we are within tax season or not.')
       
    st.write('The idea is the the exogenous variable can really be anything as long as \
              we have the data available, they are not impacted by the dependent variable \
              (the thing we are predicting).')
                                          
st.markdown('---')

## -------------------------------------------------------------------------
##  ARIMAX MODELING
## -------------------------------------------------------------------------
