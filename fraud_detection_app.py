import json
import requests
import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd
from typing import Dict
def load_model():
    with open('fraud_detection_model-0.0.1.pkl', 'rb') as file:
        data = pickle.load(file)
    return data
# App title
st.title("Credit Card Transaction Fraud Detection App")

#some image
st.image("img/credit_card_fraud.jpg")

# Description
st.write(
    """
    ## About
    
    With the growth of e-commerce websites, people and financial companies rely on online services
    to carry out their transactions that have led to an exponential increase in the credit card frauds.
    Fraudulent credit card transactions lead to a loss of huge amount of money. The design of an
    effective fraud detection system is necessary in order to reduce the losses incurred by the
    customers and financial companies. 

    **This Streamlit App  utilizes a Machine Learning model(XGBoost) API to detect potential fraud in credit card transactions.**


    """
)


###################### Funtions to transform categorical variable #############################################
def type_transaction(content):
    if content == "PAYMENT":
        content = 0
    elif content == "TRANSFER":
        content = 1
    elif content == "CASH_OUT":
        content = 2
    elif content == "DEBIT":
        content = 3
    elif content == "CASH_IN":
        content = 4
    return content

######################################### Input elements #############################################################
st.sidebar.header("Input user and transaction information")

# User data
sender_name = st.sidebar.text_input(" Sender Name ID")
receiver_name = st.sidebar.text_input(" Receiver Name ID")

## Transaction information
type_lebels = ("PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN")
type = st.sidebar.selectbox(" Type of transaction", type_lebels)

step = st.sidebar.slider("Number of Hours it took the Transaction to complete:", min_value = 0, max_value = 744)

amount = st.sidebar.number_input("Amount in $",min_value=0, max_value=110000)
oldbalanceorg = st.sidebar.number_input("""Sender Balance Before Transaction was made""",min_value=0, max_value=110000)
newbalanceorg = st.sidebar.number_input("""Sender Balance After Transaction was made""",min_value=0, max_value=110000)
oldbalancedest = st.sidebar.number_input("""Recipient Balance Before Transaction was made""",min_value=0, max_value=110000)
newbalancedest = st.sidebar.number_input("""Recipient Balance After Transaction was made""",min_value=0, max_value=110000)
## flag 
isflaggedfraud = "Non fraudulent"
if amount >= 200000:
  isflaggedfraud = "Fraudulent transaction"
else:
  isflaggedfraud = "Non fraudulent"


result_button = st.button("Detect Result")

if result_button:

    ## Features
    #data= np.array([step, type_transaction(type), amount, oldbalanceorg, newbalanceorg, oldbalancedest, newbalancedest]).reshape(1,-1)
    data = {
        "step": step,
        "type": type_transaction(type),
        "amount": amount,
        "oldbalanceOrg": oldbalanceorg,
        "newbalanceOrig": newbalanceorg,  # corrected from newbalanceOrg
        "oldbalanceDest": oldbalancedest,
        "newbalanceDest": newbalancedest  # corrected from newbalancedDest
    }

    features = pd.DataFrame([data])

    ## Transaction detail
    st.write(
        f""" 
        ## **Transaction Details**

        #### **User informantion**:

        Sender Name(ID): {sender_name}\n
        Receiver Name(ID): {receiver_name}

        #### **Transaction information**:

        Number of Hours it took to complete: {step}\n
        Type of Transaction: {type}\n
        Amount Sent: {amount}$\n
        Sender Balance Before Transaction: {oldbalanceorg}$\n
        Sender Balance After Transaction: {newbalanceorg}$\n
        Recepient Balance Before Transaction: {oldbalancedest}$\n
        Recepient Balance After Transaction: {newbalancedest}$\n
        System Flag Fraud Status(Transaction amount greater than $200000): {isflaggedfraud}

        """
    )

    st.write("""## **Prediction**""")

  # Load the model
    with open("fraud_detection_model-0.0.1.pkl", "rb") as file:
        model = joblib.load(file)
    prediction = model.predict(features)
    if prediction[0] == 0:
        st.write(f"""### The **'{type}'** transaction that took place between {sender_name} and {receiver_name} is not fraud.""")
    else:
        st.write(f"""### The **'{type}'** transaction that took place between {sender_name} and {receiver_name} is  fraud.""")

    



