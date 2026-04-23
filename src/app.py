#app.py

import streamlit as st
import plotly.express as px
from data_loader import init_dataframe, returns_df

#Load csv
init_dataframe()
print(returns_df)

#Streamlit
st.write("""
         # My first app
         Hello *world!*
         """)
