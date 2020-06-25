from response_generation import *
import streamlit as st

st.title("Response Generation")

i =0
while True:
    string = st.text_input("What d'you say?", key = i)
    
    i+=1
    if string is not "":
        # st.write("`This looks like code`")
        st.write(f"`{get_response_beam(string)}`")
        
    if string is not "":
        continue
    else:
        break