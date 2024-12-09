# Fraud Detection Model with Streamlit

This repository contains a fraud detection model developed using Random Forest and implemented with Streamlit for an interactive user interface. The model is trained on a dataset to classify fraudulent transactions and allows users to test predictions via a web-based app.

## Features

1. Machine Learning Model: A Random Forest Classifier trained on balanced data using SMOTE for oversampling.
2. Streamlit App: An intuitive web interface to upload datasets, view predictions, and interact with the model.
3. Data Handling: Limits large datasets to avoid memory issues and ensures compatibility with the Streamlit framework.
4. Real-time Predictions: Input data manually or via a dataset for fraud detection predictions.

## Technologies Used

* Python: The core programming language.
* Scikit-learn: For building and training the Random Forest model.
* Pandas & NumPy: For data manipulation and processing.
* Streamlit: For building the user-friendly web interface.
* LightGBM (optional): Experimentation with an alternate ML model.

## NOTEBOOKS: 
I created a custom dataset named Balanced_data.csv to improve the performance of my machine learning model. The project is structured across two separate notebooks:

### First Notebook:[Final project.ipynb]
- Processes the original datasets (3GB total) by performing tasks such as data cleaning, preprocessing, exploration, and combining multiple formats (CSV, JSON).
- Includes an initial machine learning model with suboptimal results.

### Second Notebook:[Fraud detection.ipynb]
- Utilizes the Balanced_data.csv (a 20MB file) for building a proper machine learning model.
- Covers evaluation methods, optimized model development, and implementation of Streamlit code for deployment.

