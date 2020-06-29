import streamlit as st
import response_generation as stos
import TA_response_generation as tastos
print("Libs loaded")
st.title("Response Generation")

# Getting the response mechanism
option = st.selectbox('Response Mechanism',('Seq2Seq', 'TA-Seq2Seq'))

if option is 'Seq2Seq':
    respfunc = stos.get_response_beam
elif option in 'TA-Seq2Seq':
    respfunc = tastos.get_response_beam

i = 0
while True:
    string = st.text_input("Stimulus", key = i)
    
    i+=1
    if string is not "":
        st.write(f" **Response : ** `{respfunc(string)}`")
        
    if string is not "":
        continue
    else:
        break