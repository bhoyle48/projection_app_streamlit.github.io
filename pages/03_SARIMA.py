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
##  SARIMA BLOG CONTENT
## -------------------------------------------------------------------------
     
# Cusomtize Main Page
st.title('Seasonal Autoregressive Integrated Moving Average') 

st.write('So by now, I bet you’re thinking - what wrong with an ARIMA or \
         ARIMAX model? Well, the biggest drawback on these two models is their \
        inability to accurately and efficiently forecast data that is seasonal, \
        or has some repeating cycle. ARIMA(X) models expect that the data is either \
        not seasonal or has the seasonality removed (i.e. mathematically \
        removed), and that is the exact place where SARIMA models come to the rescue.')
        
st.write('Like the two before, a Seasonal Autoregressive Integrated Moving Average \
         is an updated version of the ARIMA model, which has new parameters adjusting \
         for the seasonal effects within the autoregressive and moving average components \
         of the model. Knowing that tax season lasts from January until mid April, or \
        that Christmas Tree sales spike the weeks prior to December 25th, we should think \
        about accounting for that otherwise, we would mistakenly assume that tax filings, \
        or Christmas trees are no longer the hot new thing after their time when in reality, \
        the service is just not longer needed.')
        
                
with st.expander('Read More'):        
    st.subheader('What are the components of SARIMA?')
    
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
    - **Seaonal Factor** - refers to the cyclical or recurring patterns within the \
        endogenous variable 
        
    So what does this look like?
    """
    
    st.markdown(markdown)
       
        
    st.write('')
    st.subheader('ARIMAX Parameters')    
    
    st.write('Just like ARIMA(X) models, there are three main components: p, d, and q. \
             Now that we are adjusting for seasonality though, we have to begin \
            configuring the seasonality which requires four new parameters: ')
    
    markdown = """
    - **p**: the number of lag observations in the model
    - **d**: the number of times the raw observations are differenced 
    - **q**: the size of the averaged window
    """
    
    markdown2 = """
    - **P**: the number of lag observations of the seasonal component of the model
    - **D**: the number of times the seasonal component is differenced 
    - **Q**: the size of the averaged window for the seasonal component (
    - **m**: the number of time steps for a single seasonal period 

    """

    st.markdown(markdown)
    st.write('')
    st.write('Our four new parameters:')
    st.markdown(markdown2)
    
    st.write('')
    st.write('To save you the headache, I won’t show the formula here but feel \
             free to explore some more on *_Towards Data Science_*: \
            [here](https://towardsdatascience.com/understanding-sarima-955fe217bc77) \
            or \
            [here](https://towardsdatascience.com/time-series-forecasting-with-a-sarima-model-db051b7ae459)')

                                          
st.markdown('---')

## -------------------------------------------------------------------------
##  SARIMA MODELING
## -------------------------------------------------------------------------
