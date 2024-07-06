import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from logger import logging
from exception import CustomException



def load_data(data_path):
    """
    Loads the housing data from a CSV file.

    Args:
        data_path (str): The path to the CSV file containing the housing data.

    Returns:
        pandas.DataFrame: The loaded housing data as a pandas DataFrame.
    """
    logging.info('Load Data Function Starts')
    try:
        df = pd.read_csv(data_path)
        return df.dropna()  # Handle missing values
        logging.info("Dataset read as pandas Dataframe")
    except Exception as e:
        logging.info("Exception occured in load Data function")
        raise CustomException(e,sys)


def split_data(df, test_size=0.3, random_state=42):
    """
    Splits the data into training and testing sets.

    Args:
        df (pandas.DataFrame): The DataFrame containing the housing data.
        test_size (float, optional): The proportion of data to be used for the test set. Defaults to 0.3.
        random_state (int, optional): The random state seed for reproducibility. Defaults to 42.

    Returns:
        tuple: A tuple containing the training and testing sets for features (X) and target variable (y).
    """

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    logging.info("X y is divided based on Independent and Dependent features")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def preprocess_data(X_train):
    """
    Preprocesses the data by standardizing features.

    Args:
        X_train (pandas.DataFrame): The training set features.

    Returns:
        tuple: A tuple containing the standardized X_train and a fitted StandardScaler object.
    """

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    logging.info("X_train is Scaled")
    logging.info("sclaer object is created")
    return X_train_scaled, scaler


def train_model(X_train, y_train):
    """
    Trains a linear regression model on the provided data.

    Args:
        X_train (pandas.DataFrame): The training set features.
        y_train (pandas.Series): The training set target variable.

    Returns:
        sklearn.linear_model._base.LinearRegression: The trained linear regression model.
    """

    regression = LinearRegression()
    regression.fit(X_train, y_train)
    logging.info("Regression model is trained")
    return regression


def make_predictions(model, X_test, scaler):
    """
    Makes predictions on the test data using the trained model.

    Args:
        model (sklearn.linear_model._base.LinearRegression): The trained linear regression model.
        X_test (pandas.DataFrame): The test set features.
        scaler (sklearn.preprocessing._data.StandardScaler): The fitted StandardScaler object.

    Returns:
        pandas.Series: The predicted target values for the test set.
    """

    X_test_scaled = scaler.transform(X_test)
    reg_pred = model.predict(X_test_scaled)
    logging.info("Predictions are made")
    return reg_pred


def save_model(model, scaler, scaler_path, model_path):
    """
    Saves the trained model and scaler using pickle.

    Args:
        model (sklearn.linear_model._base.LinearRegression): The trained linear regression model.
        scaler (sklearn.preprocessing._data.StandardScaler): The fitted StandardScaler object.
        scaler_path (str): The path to save the StandardScaler object.
        model_path (str): The path to save the trained model.
    """

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    logging.info("Model and scaler are saved")


def main():
    """
    The main function that executes the housing price prediction pipeline.
    """

    data_path = "data/HousingData.csv"  # Replace with your data path
    scaler_path = "model/scaling.pkl"
    model_path = "model/regmodel.pkl"

    try:
        df = load_data(data_path)
        X_train, X_test, y_train, y_test = split_data(df)
        X_train_scaled, scaler = preprocess_data(X_train)
        model = train_model(X_train_scaled, y_train)
        reg_pred = make_predictions(model, X_test, scaler)

        # Additional analysis or evaluation could be done here, e.g., calculating metrics

        save_model(model, scaler, scaler_path, model_path)
        print("Model and scaler saved successfully!")
        

    except Exception as e:
        raise CustomException(e,sys)

if __name__ == "__main__":
    main()

        
