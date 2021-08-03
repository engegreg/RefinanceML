import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_auc_score
from sklearn.metrics import plot_confusion_matrix
from xgboost import plot_tree

st.title("Refinance Classification Model")

st.write("""
This is a Machine Learning model that determines whether someone is eligible for refinancing a loan.
Adjust the slider to the left to observer the model's behavior with additional depth. 
""")

get_n_estimators = st.sidebar.slider("Select depth", 1,25)


dfimb = pd.read_csv('loan_data.csv')
dfimb.drop(dfimb.index[dfimb['purpose'] == 'credit_card'], inplace=True)
    #create target variable#

dfimb['refi_possible'] = 1
dfimb.loc[dfimb['fico'] <= 650, 'refi_possible'] = 0
dfimb.loc[dfimb['inq.last.6mths'] > 1, 'refi_possible'] = 0
dfimb.loc[dfimb['delinq.2yrs'] > 0, 'refi_possible'] = 0
    #drop vars used to create target variable#


count_class0, count_class1 = dfimb['credit.policy'].value_counts()

    # Creating 2 separate dataframes that contain the different values for credit policy. We'll use this to undersample the data.#

df_class0 = dfimb[dfimb['credit.policy'] == 0]
df_class1 = dfimb[dfimb['credit.policy'] == 1]
print(df_class0.shape)
df_class1.shape

    #Using sample function to create undersampling base. This randomizes the dataset and avoids bias.

df_under_base = df_class1.sample(count_class1)

    #concatenate results.

df = pd.concat([df_under_base, df_class0], axis=0)

mask = {'debt_consolidation': 0, 'all_other': 1, 'home_improvement': 2, 'small_business': 3, 'major_purchase': 4,
            'educational': 5}

df["Loan_type"] = df["purpose"].replace(mask)
df.drop(columns='purpose', inplace=True)
df.drop(columns=['fico', 'inq.last.6mths'], inplace=True)

df.reset_index(drop=True, inplace=True)



#Creating variables

target = 'refi_possible'
y=df[target]
X=df.drop(columns=target)

#TTS

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2, random_state = 1)


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
