import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets


st.title('This is my first application')

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')

# Load 10,000 rows of data into the dataframe.
data = datasets.load_iris()

# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')


st.subheader('Raw Iris Data')
st.write(data)
