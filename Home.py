## IMPORT LIBRARIES
import streamlit as st
import os

os.chdir('/Users/Benjamin/Documents/GitHub/forecasting') 
# from sidebar import mysidebar


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
# mysidebar()


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
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots



## -------------------------------------------------------------------------
##  Functions
## ------------------------------------------------------------------------- 

st.markdown(""" <style> .font {
font-size:px; font-family: 'Montserrat'; color: #FF9633;} 
</style> """, unsafe_allow_html=True)

figheader = st.markdown('<p class="font">file.columns[1]</p>', unsafe_allow_html=True)

## -------------------------------------------------------------------------
##  File Upload
## -------------------------------------------------------------------------      

# Add File Component
uploaded_file = st.file_uploader("", type='csv')


# If there is not a file uploaded
if uploaded_file is not None:
    if 'load_csv' is st.session_state:
        df = pd.read_csv('/Users/Benjamin/Desktop/BTC-USD.csv')
        df = st.session_state.load_csv
        st.write(uploaded_file.name + " is loaded")
    else:
        df = pd.read_csv(uploaded_file)
        st.session_state.load_csv = df
        st.write(uploaded_file.name + " is loaded")
else:
    df = pd.read_csv('/Users/Benjamin/Desktop/BTC-USD.csv')
    st.session_state.load_csv = df 
    st.write("BTC-USD.csv is loaded")
    
fig = go.Figure(
        go.Scatter(x=df.iloc[:, 0], y=df.iloc[:, 1], mode="lines", 
                           line=dict(color='#0079c2', width=1)))

fig.update_layout(width=900, height=500, title=df.columns[1], title_x=0.5)
    
st.plotly_chart(fig, use_container_width = True)
    

