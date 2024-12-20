# TODO imports!!
from ucimlrepo import fetch_ucirepo, list_available_datasets
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA 
import scipy.cluster.hierarchy as shc 
from sklearn.model_selection import KFold

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score
from itertools import product

# TODO
RANDOM_SEED = 42
K_FOLDS = 5

# function for converting categorical features into numerical
def encode_categorical_features(X, encoder):
    X = encoder.fit_transform(X)
    return X

def import_dataset(uci_id, encoder):
    # get the dataset
    dataset = fetch_ucirepo(id=uci_id) 
    # load data into dataframe for easier preprocessing
    df = pd.concat([dataset["data"]["features"],dataset["data"]["targets"] ],axis=1)
    # remove nan values
    df.dropna(inplace=True)
    
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1:]
    # encode categorical data only for features not for target itself
    # https://stackoverflow.com/questions/29803093/check-which-columns-in-dataframe-are-categorical
    cols = X.columns
    num_cols = X._get_numeric_data().columns
    #print(num_cols)
    categorical_cols = list(set(cols) - set(num_cols))
    print(categorical_cols)
    #X_temp = X.copy()
    #X.loc[:, categorical_cols] = encode_categorical_features(X[categorical_cols], encoder)
    X.loc[:, categorical_cols] = encode_categorical_features(X.loc[:, categorical_cols], encoder)

    
    # check if encoding has worked
    # https://stackoverflow.com/questions/26924904/check-if-dataframe-column-is-categorical
    for c in X.columns:
        if X[c].dtype.name == "category":
            print(f"WARNING: Column {c} still has categorical values!")
            
    # last column is target
    return X, y

# supervised classifiers
def test_classifier(clf, clf_name, params, X_train, y_train, X_test, y_test):
    # https://towardsdatascience.com/gridsearchcv-for-beginners-db48a90114ee
    pipe = Pipeline([('scaler', StandardScaler()), (clf_name, clf)])
    scoring = 'accuracy'
    gs_knn = GridSearchCV(pipe,
                        param_grid=params,
                        scoring=scoring,
                        cv=K_FOLDS)

    # Ravel to convert from (len, 1) shape to (len,), warning from sk-learn
    y_train = np.ravel(y_train)

    gs_knn.fit(X_train, y_train)
    print(gs_knn.best_params_)
    # find best model score
    y_pred_train = gs_knn.best_estimator_.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)
    # TODO
    train_f1 = f1_score(y_train, y_pred_train, average="weighted")
    #train_score = gs_knn.score(X_train, y_train)

    # test on test set
    y_pred_test = gs_knn.best_estimator_.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    # TODO
    test_f1 = f1_score(y_test, y_pred_test, average="weighted")
    #test_score = gs_knn.score(X_test, y_test)

    print(f"{clf_name}, Train accuracy = {train_acc}, Test accuracy = {test_acc}")
    print(f"{clf_name}, Train f1-score = {train_f1}, Test f1-score = {test_f1}")

    # Confusion Matrix
    print("Confusion Matrix for train set")
    cm_train = confusion_matrix(y_train, y_pred_train, labels=gs_knn.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=gs_knn.classes_)
    disp.plot()
    plt.show()

    print("Confusion Matrix for test set")
    cm_test = confusion_matrix(y_test, y_pred_test, labels=gs_knn.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=gs_knn.classes_)
    disp.plot()
    plt.show()
    
    return cm_train

# unsupervised classifiers

# function to align cluster labels with ground truth and calculate accuracy
# based on these sources:
# https://docs.neurodata.io/notebooks/pedigo/graspologic/clustering/2020/10/12/remapping-labels.html
# https://stackoverflow.com/questions/4075669/hungarian-algorithm-in-python
# https://se.mathworks.com/matlabcentral/answers/1470216-matching-the-labels-of-a-clustering-with-ground-truth-labels-for-performance-analysis
def cluster_accuracy(y_true, y_pred):
    """
    Align clustering labels with true labels and calculate accuracy.
    """
    # create contingency table, similar to confusion matrix   
    contingency_table = confusion_matrix(y_true, y_pred)

    # Hungarian algorithm to find the optimal one-to-one mapping between the predicted labels 
    # and the true labels based on the contingency table
    # based on https://stackoverflow.com/questions/4075669/hungarian-algorithm-in-python we can use linear_sum_assignment for this
    row_ind, col_ind = linear_sum_assignment(contingency_table, maximize=True)

    # remapping of labels
    label_map = dict(zip(col_ind, row_ind))
    remapped_y_pred = np.vectorize(label_map.get)(y_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, remapped_y_pred)
    return accuracy

def agg_clustering(X_train, y_train, X_test, y_test):
        
    # split train set into train and validation set to perform a grid search
    # TODO
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=RANDOM_SEED)
    
    # Scaling based on train set
    scaler = StandardScaler() 
    X_train_scaled = scaler.fit_transform(X_train)
    
    # scale val and test set with params of train set
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # create grid or params that we want to test
    metrics = ["euclidean", "l1", "l2", "manhattan"]
    linkages = ["complete", "average", "single"]
    pca_options = [True, False]
    
    # Ensure ward is only paired with euclidean
    param_grid = [
        {'metric': 'euclidean', 'linkage': 'ward', 'pca': pca}
        for pca in pca_options
    ] + [
        {'metric': metric, 'linkage': linkage, 'pca': pca}
        for metric, linkage, pca in product(metrics, linkages, pca_options)
        if linkage != 'ward' or metric == 'euclidean'
    ]
    
    best_score = -1
    best_params = None
    
    for params in param_grid:
        
        # apply PCA if specified
        if params["pca"]:
            pca = PCA(n_components = 2, random_state=RANDOM_SEED) 
            X_transformed = pca.fit_transform(X_train_scaled) 
            X_val_transformed = pca.transform(X_val_scaled)
        else:
            X_transformed = X_train_scaled
            X_val_transformed = X_val_scaled
            
        agg_clustering = AgglomerativeClustering(n_clusters=len(np.unique(y_train)), metric=params["metric"], linkage=params["linkage"])
        train_pred = agg_clustering.fit_predict(X_transformed)
        val_pred = agg_clustering.fit_predict(X_val_transformed)
        
        train_acc = cluster_accuracy(y_train, train_pred)
        val_acc = cluster_accuracy(y_val, val_pred)
        
        # use a score to select hyperparams both based on best train accuracy but also on best validation accuracy
        #gamma = 0.5
        #score = val_acc - gamma * abs(train_acc - val_acc)
        score = 0.8 * val_acc + 0.2 * train_acc
        if score > best_score:
            best_params = params
            best_score = score
            
    print(f"Best score {best_score} with params {best_params}")
    if best_params["pca"]:
        pca = PCA(n_components = 2, random_state=RANDOM_SEED) 
        pca.fit(X_train_scaled) 
        X_test_transformed = pca.transform(X_test_scaled)
    else:
        X_test_transformed = X_test_scaled
        
    
    agg_clustering = AgglomerativeClustering(n_clusters=len(np.unique(y_train)), metric=best_params["metric"], linkage=best_params["linkage"])    
    test_pred = agg_clustering.fit_predict(X_test_transformed)
    test_acc = cluster_accuracy(y_test, test_pred)
    print(f"Test accuracy = {test_acc}")
    
    