import streamlit as st
import pandas as pd
import numpy as np


st.title('This is my first application')

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')

d = []
for p in range(0,51):
    d.append(
        {
            'N': p,
            'N^2': p**2,
            'N^3':  p**3
        }
    )


data = pd.DataFrame(d)
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')


st.subheader('Raw Iris Data')
st.write(data)
