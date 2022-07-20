import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
import time


st.title('This is my first application')

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')

# save load_iris() sklearn dataset to iris
# if you'd like to check dataset type use: type(load_iris())
# if you'd like to view list of attributes use: dir(load_iris())
iris = datasets.load_iris()

# np.c_ is the numpy concatenate function
# which is used to concat iris['data'] and iris['target'] arrays 
# for pandas column argument: concat iris['feature_names'] list
# and string list (in this case one string); you can make this anything you'd like..  
# the original dataset would probably call this ['Species']
data = pd.DataFrame(d1= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])


data = pd.DataFrame(iris)
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')


st.subheader('Raw Iris Data')
st.write(data)
