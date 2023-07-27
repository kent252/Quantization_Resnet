import streamlit as st
import time
from model_main import *


def sec1():
    # with st.empty():
    for i in range(4):
        st.write(i)
        time.sleep(1)
def sec2():
    # with st.empty():
    st.write("ABC")
    time.sleep(1)
def total():
    sec1()
    sec2()
col1, col2 = st.columns(2)
with col1:
    with st.form("my_form"):
        st.write("Inside the form")
        pretrained = st.selectbox(
            'Pretrained Model?',
            ('resnet50', 'resnet152'))
        path_to_data = st.text_input(
            'Data_set?','./data/dog_breed')
        num_class = st.text_input(
            'Number of classes','120')
        epoch = st.text_input(
            'Number of epoch','10')
        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
with col2:
    if submitted:
            train_model(pretrained,int(num_class),int(epoch))
            inference_quant(pretrained,int(num_class))



