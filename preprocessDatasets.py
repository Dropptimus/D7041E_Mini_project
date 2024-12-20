from ucimlrepo import fetch_ucirepo, list_available_datasets
import pandas as pd
from sklearn.model_selection import train_test_split


# followed this tutorial https://medium.com/@cuneytharp/make-a-classification-model-with-sklearn-step-by-step-2199a12e4dfe
def preprocess_adult():
    # load data
    adult = fetch_ucirepo(id=2)
    # load data into dataframe for easier preprocessing
    df = pd.concat([adult["data"]["features"],adult["data"]["targets"] ],axis=1)
    # replace wrong labels by correct one
    df["income"].replace("<=50K.","<=50K",inplace=True)
    df["income"].replace(">50K.",">50K",inplace=True)
    # drop education column as education and education-num contain the same data
    df.drop("education",axis=1,inplace=True)
    # drop all missing values as they are categorical
    df.dropna(inplace=True)
    
    return df


def preprocess_iris():
    
    iris = fetch_ucirepo(id=53)
    # load data into dataframe for easier preprocessing
    df = pd.concat([iris["data"]["features"],iris["data"]["targets"] ],axis=1)
    # remove nan values
    df.dropna(inplace=True)
    # no preprocessing needed
    return df

def preprocess_heart_disease():
    data = fetch_ucirepo(id=45)
    # load data into dataframe for easier preprocessing
    df = pd.concat([data["data"]["features"],data["data"]["targets"] ],axis=1)
    # remove nan values
    df.dropna(inplace=True)
    return df

def preprocess_accute_inflammations():
    data = fetch_ucirepo(id=184)
    # load data into dataframe for easier preprocessing
    df = pd.concat([data["data"]["features"],data["data"]["targets"] ],axis=1)
    # remove nan values
    df.dropna(inplace=True)
    # no further preprocessing needed
    return df

# numerical
def preprocess_credit_approval():
    
    data = fetch_ucirepo(id=143)
    # load data into dataframe for easier preprocessing
    df = pd.concat([data["data"]["features"],data["data"]["targets"] ],axis=1)
    # remove nan values
    df.dropna(inplace=True)
    # no further preprocessing needed
    return df

def preprocess_wholesale_customers():
    data = fetch_ucirepo(id=292)
    # load data into dataframe for easier preprocessing
    df = pd.concat([data["data"]["features"],data["data"]["targets"] ],axis=1)
    # remove nan values
    df.dropna(inplace=True)
    # no further preprocessing
    return df

def preprocess_glass_identification():
    data = fetch_ucirepo(id=42)
    # load data into dataframe for easier preprocessing
    df = pd.concat([data["data"]["features"],data["data"]["targets"] ],axis=1)
    # remove nan values
    df.dropna(inplace=True)
    # no further preprocessing
    return df

def preprocess_ilpd():
    data = fetch_ucirepo(id=225)
    # load data into dataframe for easier preprocessing
    df = pd.concat([data["data"]["features"],data["data"]["targets"] ],axis=1)
    # remove nan values
    df.dropna(inplace=True)
    return df

def preprocess_hcv():
    data = fetch_ucirepo(id=503)
    # load data into dataframe for easier preprocessing
    df = pd.concat([data["data"]["features"],data["data"]["targets"] ],axis=1)
    # remove nan values
    df.dropna(inplace=True)
    # no further preprocessing
    return df

def preprocess_land_mines():
    data = fetch_ucirepo(id=763)
    # load data into dataframe for easier preprocessing
    df = pd.concat([data["data"]["features"],data["data"]["targets"] ],axis=1)
    # remove nan values
    df.dropna(inplace=True)
    # no further preprocessing
    return df

# check balance of classes, one has less samples
def preprocess_balance_scale():
    data = fetch_ucirepo(id=12)
    # load data into dataframe for easier preprocessing
    df = pd.concat([data["data"]["features"],data["data"]["targets"] ],axis=1)
    # remove nan values
    df.dropna(inplace=True)
    # no further preprocessing
    return df
    



    
    
    