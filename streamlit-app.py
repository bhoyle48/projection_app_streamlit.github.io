import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
import time


st.title('This is my first application')

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')

iris = pd.DataFrame(data=data.data, columns=data.feature_names)

data = iris
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')


st.subheader('Raw Iris Data')
st.write(data)
