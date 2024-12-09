# Fraud Detection Model with Streamlit

This repository contains a fraud detection model developed using Random Forest and implemented with Streamlit for an interactive user interface. The model is trained on a dataset to classify fraudulent transactions and allows users to test predictions via a web-based app.

## Features

Machine Learning Model: A Random Forest Classifier trained on balanced data using SMOTE for oversampling.
Streamlit App: An intuitive web interface to upload datasets, view predictions, and interact with the model.
Data Handling: Limits large datasets to avoid memory issues and ensures compatibility with the Streamlit framework.
Real-time Predictions: Input data manually or via a dataset for fraud detection predictions.
Technologies Used

Python: The core programming language.
Scikit-learn: For building and training the Random Forest model.
Pandas & NumPy: For data manipulation and processing.
Streamlit: For building the user-friendly web interface.
LightGBM (optional): Experimentation with an alternate ML model.
