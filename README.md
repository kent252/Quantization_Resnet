# Resnet Quantizated Model
## Overview
    This project include the Resnet pretrained model and the process to Quantizated it 
    The quantization method uses in this project is QAT (Quantization-Aware-Training)
## Installation
    Install the requirement package in the requirement.txt
    '''
        pip install -r requirements.txt
    '''
## Datasets
    Before using this code, please to download the data from:
    [Dog breeds kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)
    And then run the following command to prepare the dataset
    '''
        python ./tool/split.py --input <Your data path> --output <Your new data path>
    '''
## Training and Quantization
    To run the streamlit, please run this command:
    '''
        streamlit run ./tool/app_main.py
    '''