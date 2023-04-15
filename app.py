

## Importing Libraries
import pandas as pd

import numpy as np

import pickle

import base64

import matplotlib.pyplot as plt

import streamlit as st


st.write("""

# Targetted Marketting App


The Original dataset was created by The Criteo AI Lab .The dataset consists of 13M rows, each one representing a user with 12 features, a treatment indicator and 2 binary labels (visits and conversions). Positive labels mean the user visited/converted on the advertiser website during the test period (2 weeks). The global treatment ratio is 84.6%. It is usual that advertisers keep only a small control population as it costs them in potential revenue.

Following is a detailed description of the features:

f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11: feature values (dense, float)
exposure: treatment effect, whether the user has been effectively exposed (binary)


""")




## Taking inputs

exposure = st.sidebar.selectbox('exposure-(Yes-1,No-0)',(1,0))


f0 = st.sidebar.number_input('Feature 0  :  Enter value from 0.0 to 40.0')

f1 = st.sidebar.number_input('Feature 1  :  Enter value from 0.0 to 30.0')

f2 = st.sidebar.number_input('Feature 2  :  Enter value from 0.0 to 20.0')

f3 = st.sidebar.number_input('Feature 3  :  Enter value from -15.0 to 15.0')

f4 = st.sidebar.number_input('Feature 4  :  Enter value from 0.0 to 35.0')

f5 = st.sidebar.number_input('Feature 5  :  Enter value from -20.0 to 20.0')

f6 = st.sidebar.number_input('Feature 6  :  Enter value from -50.0 to 10.0')

f7 = st.sidebar.number_input('Feature 7  :  Enter value from 0.0 to 25.0')

f8 = st.sidebar.number_input('Feature 8  :  Enter value from 0.0 to 10.0')

f9 = st.sidebar.number_input('Feature 9  :  Enter value from 5.0 to 90.0')

f10 = st.sidebar.number_input('Feature 10  :  Enter value from 0.0 to 80.0')

f11 = st.sidebar.number_input('Feature 11  :  Enter value from -10.0 to 10.0')


## Making a datframe from input to form test dataset on which prediction is to be done

data = {

        'f0':[f0],
        'f1':[f1],
        'f2':[f2],
        'f3':[f3],
        'f4':[f4],
        'f5':[f5],
        'f6':[f6],
        'f7':[f7],
        'f8':[f8],
        'f9':[f9],
        'f10':[f10],
        'f11':[f11],
        'exposure':[exposure]
        }


st.write(" The input features")
input_df = pd.DataFrame(data)

st.write(input_df)


a='pred_final.sav'


load_clf = pickle.load(open( a, 'rb'))

prediction = load_clf.predict(input_df)

st.subheader('Class Prediction')

st.write(prediction)







