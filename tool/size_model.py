import torch
import os
import streamlit as st

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    st.write('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')