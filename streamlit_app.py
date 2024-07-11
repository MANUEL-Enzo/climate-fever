import streamlit as st
from fonctions import *

# usage : python -m streamlit run streamlit_app.py  

st.set_page_config(page_title='Climate Fever', page_icon='🤖')
st.title('Climate Fever')

st.markdown('**What can this app do?**')
st.markdown("This app allow you to check if a sentence about climate is true, false of if we don't have enough information about it.")


user_input = st.text_input("Write your sentence here :")
prediction = predict(user_input)

st.markdown(f"According to our model the label of your sentence is {prediction}")
