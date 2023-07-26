import streamlit as st

from main_v2 import *

col1, col2, col3 = st.columns(2)
with col1:
    with st.form("my_form"):
        st.write("Inside the form")
        pretrained = st.selectbox(
            'Pretrained Model?',
            ('Resnet50', 'Resnet152'))
        path_to_data = st.selectbox(
            'Data_set?',
            ('Dog breed'))
        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
with col2:
    if submitted:
        main()

st.write("Outside the form")