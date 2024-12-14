from ucimlrepo import fetch_ucirepo, list_available_datasets
import pandas as pd
from sklearn.model_selection import train_test_split


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
    
    
    