import streamlit.components.v1 as components
import streamlit as st 

st.title("Hello Custom React Table")

with open('./Vis/pages/aaa.html') as f:
    data = f.read()
    components.html(data, height=800)