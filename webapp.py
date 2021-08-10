#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, classification_report, roc_auc_score


df=pd.read_csv('loan_data_cleaned.csv')

st.title("Refinance Classification Model")

st.write("""
This is a Machine Learning model that determines whether someone is eligible for refinancing a loan.
Adjust the slider to the left to observer the model's behavior with additional depth. 
""")

#Create interactive slider
get_n_estimators = st.sidebar.slider("Select depth", 1,25)


#Creating variables

target = 'refi_possible'
y=df[target]
X=df.drop(columns=target)

#TTS

X_train, X_val, y_train, y_val = train_test_split(X,y,
        test_size=0.2, random_state = 1)


st.write(df.head())
#This function trains the model, returns the metrics of said model, and directly connects to the slider on the web app to make it interactive.

def adjust_depth(slider):
    params = dict()


    n_estimators = get_n_estimators
    params['n_estimators'] = n_estimators


    model_xg = XGBClassifier(use_label_encoder=False, n_estimators=params['n_estimators'])

    model_xg.fit(X_train, y_train)

    st.write("Training accuracy: ", model_xg.score(X_train, y_train))
    st.write("Validation accuracy: ", model_xg.score(X_val, y_val))
    st.write(classification_report(y_val,model_xg.predict(X_val),target_names = ['Approved','Declined']))

    return params


adjust_depth(get_n_estimators)
