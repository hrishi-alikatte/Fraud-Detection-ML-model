#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[2]:


balanced_data = pd.read_csv("Balanced_data.csv")


# In[3]:


balanced_data


# In[ ]:





# In[4]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE

# 1. Separate features (X) and target (y)
X = balanced_data.drop(columns=['is_fraud_Yes'])  # Features
y = balanced_data['is_fraud_Yes']  # Target

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=330)

# 3. Apply SMOTE to handle class imbalance in the training set
smote = SMOTE(sampling_strategy='auto', random_state=330)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 4. Train Random Forest Classifier with balanced class weights
rf_model = RandomForestClassifier(n_estimators=50, random_state=330, n_jobs=-1, class_weight='balanced')
rf_model.fit(X_resampled, y_resampled)

# 5. Evaluate model performance with AUC
fpr, tpr, thresholds = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc}")

# 6. Print classification report for detailed performance metrics
print(classification_report(y_test, rf_model.predict(X_test)))


# In[5]:


import pickle

# Save the model to a file
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)


# In[6]:


import pickle

# Load the saved model
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)


# In[7]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report
from imblearn.over_sampling import SMOTE

# Load your dataset (replace with actual data loading code)
@st.cache_data
def load_data():
    # Replace with your dataset loading logic
    data = pd.read_csv('balanced_data.csv')  # Placeholder path
    return data

data = load_data()

# Preprocess the data and train the model
X = data.drop(columns=['is_fraud_Yes'])
y = data['is_fraud_Yes']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=330)

# Apply SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=330)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=50, random_state=330, n_jobs=-1, class_weight='balanced')
rf_model.fit(X_resampled, y_resampled)

# Streamlit app
st.title("Fraud Detection Model Demo")
st.write("This demo predicts whether a transaction is fraudulent based on input features.")

# Sidebar for user input
st.sidebar.header("Input Features")
user_input = {}
for feature in X.columns:
    user_input[feature] = st.sidebar.number_input(f"{feature}", min_value=0.0, max_value=1000000.0, step=0.01)

# Convert user input into DataFrame
input_df = pd.DataFrame([user_input])

# Predict fraud
if st.sidebar.button("Predict"):
    prediction_proba = rf_model.predict_proba(input_df)[:, 1][0]
    prediction = "Fraudulent" if prediction_proba > 0.5 else "Not Fraudulent"
    st.write(f"### Prediction: {prediction}")
    st.write(f"### Fraud Probability: {prediction_proba:.2f}")

# Evaluation metrics
if st.checkbox("Show Model Performance Metrics"):
    fpr, tpr, thresholds = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    st.write(f"*AUC:* {roc_auc:.2f}")
    st.write("*Classification Report:*")
    st.text(classification_report(y_test, rf_model.predict(X_test)))


# In[ ]:





# In[ ]:





# In[ ]:




