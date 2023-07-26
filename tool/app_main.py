import streamlit as st
import time
from main_v2 import *


def sec1():
    # with st.empty():
    for i in range(4):
        st.write(i)
        time.sleep(1)
def sec2():
    # with st.empty():
    st.write("ABC")
    time.sleep(1)
col1, col2 = st.columns(2)
with col1:
    with st.form("my_form"):
        st.write("Inside the form")
        pretrained = st.selectbox(
            'Pretrained Model?',
            ('resnet50', 'resnet152'))
        path_to_data = st.selectbox(
            'Data_set?',
            ('Dog breed',))
        num_class = st.text_input(
            'Number of classes')
        epoch = st.text_input(
            'Number of epoch')
        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
with col2:
    if submitted:
        with st.empty():
            # print(type(pretrained), type(num_class))
            # main(pretrained,int(num_class))
            for i in range(int(num_class)):
                sec1()
                sec2()




