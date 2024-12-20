from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import mode
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.preprocessing import StandardScaler
from itertools import product
from collections import ChainMap
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import pandas as pd
import seaborn as sns
from ucimlrepo import fetch_ucirepo, list_available_datasets

def cluster_accuracy(y_true, y_pred):
    """
    Align clustering labels with true labels and calculate accuracy.
    """
    for i in y_true:
        print(i, end=" ")
    print(y_pred)
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

def clustering_classification(ClusteringClass, cls_name, params, X_train, y_train, X_test, y_test, random_seed, k_folds):
    """
    Test clustering classification on dataset
    """
    RANDOM_SEED = random_seed

    params = classification_cv(ClusteringClass, cls_name, params, X_train, y_train, k_folds, random_seed)

    # create k-means classifier
    kmeans = ClusteringClass(random_state=RANDOM_SEED, **params)

    # normalize train data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    kmeans.fit(X_train)

    cluster_labels = kmeans.predict(X_train)
    # Map using mode because it is the most common class for that cluster
    # https://stats.stackexchange.com/questions/51418/assigning-class-labels-to-k-means-clusters
    labels_map = {}
    # loops through all cluster labels
    for cluster in np.unique(cluster_labels):
        class_label = mode(y_train[cluster_labels == cluster])[0] # selects the most common label for that cluster
        labels_map[cluster] = class_label # map that label to the cluster

    y_pred_train = np.array([labels_map[cluster] for cluster in cluster_labels]) # maps cluster to labels

    # normalize test data using train data mean and std
    X_test = scaler.transform(X_test)

    # predict and map for test set
    test_clusters = kmeans.predict(X_test)
    y_pred_test = np.array([labels_map[cluster] for cluster in test_clusters])

    train_acc, train_f1, test_acc, test_f1, cm_train, cm_test = metrics_and_plot_cm(cls_name, y_train, y_pred_train, y_test, y_pred_test)

    return train_acc, train_f1, test_acc, test_f1, cm_train, cm_test


def classification_cv(ClusteringClass, cls_name, params, X_train, y_train, k_folds, random_seed):
    K_FOLDS = k_folds
    RANDOM_SEED = random_seed

    # Create all parameter combinations
    # https://stackoverflow.com/questions/64645075/how-to-iterate-through-all-dictionary-combinations
    # https://stackoverflow.com/questions/3494906/how-do-i-merge-a-list-of-dicts-into-a-single-dict
    params_dict = dict(ChainMap(*params))
    keys, values = zip(*params_dict.items())
    param_combinations = [dict(zip(keys, p)) for p in product(*values)]

    highest_val_acc = 0

    # Loop through the combinations
    for param_comb in param_combinations:
        avg_train_acc = 0
        avg_valid_acc = 0
        # create clustering class
        clustering_algorithm = ClusteringClass(random_state=RANDOM_SEED, **param_comb)
        kf = KFold(n_splits=K_FOLDS, random_state=42, shuffle=True)
        kf.get_n_splits(X_train)

        # Loop through the folds
        for i, (train_index, validation_index) in enumerate(kf.split(X_train)):
            # Split into train and validation folds
            X_train_cv, X_valid = X_train.iloc[train_index-1], X_train.iloc[validation_index]
            y_train_cv, y_valid = y_train[train_index], y_train[validation_index]

            # Normalize
            scaler_cv = StandardScaler()
            X_train_cv = scaler_cv.fit_transform(X_train_cv)

            # Fit and predict training
            clustering_algorithm.fit(X_train_cv)
            cluster_labels_cv = clustering_algorithm.predict(X_train_cv)

            # fit and predict validation
            X_valid = scaler_cv.transform(X_valid)
            valid_clusters = clustering_algorithm.predict(X_valid)

            #TODO make work for all clustering methods
            # hungarian algorithm 
            if cls_name == "agglomerative_clustering":
                train_acc = cluster_accuracy(y_train_cv, cluster_labels_cv)
                valid_acc = cluster_accuracy(y_valid, valid_clusters)

            # Using most common label in cluster to give label to whole cluster
            else:
                labels_map_cv = {}
                for cluster_cv in np.unique(cluster_labels_cv):
                    # selects the most common label for that cluster
                    class_label_cv = mode(y_train_cv[cluster_labels_cv == cluster_cv])[0]
                    labels_map_cv[cluster_cv] = class_label_cv # map that label to the cluster
                # maps cluster to labels
                y_pred_train_cv = np.array([labels_map_cv[cluster_cv] for cluster_cv in cluster_labels_cv])
                y_pred_valid = np.array([labels_map_cv[cluster_cv] for cluster_cv in valid_clusters])
                valid_acc = accuracy_score(y_valid, y_pred_valid)
                train_acc = accuracy_score(y_train_cv, y_pred_train_cv)

            avg_train_acc += train_acc
            avg_valid_acc += valid_acc

        # Saves best params
        if highest_val_acc < avg_valid_acc/K_FOLDS:
            best_params = param_comb
            highest_val_acc = avg_valid_acc/K_FOLDS

        # Prints for cv results
            
        # print("-"*100)
        # print("Params: ", param_comb)
        # print("Cross validation average train accuracy:", avg_train_acc / K_FOLDS)
        # print("Cross validation average validation accuracy:", avg_valid_acc / K_FOLDS)
    
    print("Cross validation best parameters: ", best_params)

    return best_params

def metrics_and_plot_cm(clf_name, y_train, y_pred_train, y_test, y_pred_test, display = False):
    """
    Calculate train and test accuracy and f1-score
    """
    labels = np.unique(y_train)
    train_acc = accuracy_score(y_train, y_pred_train)
    # TODO
    train_f1 = f1_score(y_train, y_pred_train, average="weighted")
    #train_score = gs_knn.score(X_train, y_train)

    # test on test set
    test_acc = accuracy_score(y_test, y_pred_test)
    # TODO
    test_f1 = f1_score(y_test, y_pred_test, average="weighted")
    #test_score = gs_knn.score(X_test, y_test)

    print(f"{clf_name}, Train accuracy = {train_acc}, Test accuracy = {test_acc}")
    print(f"{clf_name}, Train f1-score = {train_f1}, Test f1-score = {test_f1}")

    # Confusion Matrix
    cm_train = confusion_matrix(y_train, y_pred_train, labels=labels)
    if display:
        disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=labels)
        disp_train.plot()
        print("Confusion Matrix for train set")
        plt.show()

    
    cm_test = confusion_matrix(y_test, y_pred_test, labels=labels)
    if display:
        disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=labels)
        disp_test.plot()
        print("Confusion Matrix for test set")
        plt.show()

    return train_acc, train_f1, test_acc, test_f1, cm_train, cm_test

def test_classifier(clf, clf_name, params, X_train, y_train, X_test, y_test, display = False):
    labels = np.unique(y_train)
    # https://towardsdatascience.com/gridsearchcv-for-beginners-db48a90114ee
    pipe = Pipeline([('scaler', StandardScaler()), (clf_name, clf)])
    scoring = 'accuracy'
    gs_knn = GridSearchCV(pipe,
                        param_grid=params,
                        scoring=scoring,
                        cv=5)

    # Ravel to convert from (len, 1) shape to (len,), warning from sk-learn
    y_train = np.ravel(y_train)
    gs_knn.fit(X_train, y_train)
    print("Cross validation best parameters:", gs_knn.best_params_)
    # find best model score
    y_pred_train = gs_knn.best_estimator_.predict(X_train)
    y_pred_test = gs_knn.best_estimator_.predict(X_test)

    train_acc, train_f1, test_acc, test_f1, cm_train, cm_test = metrics_and_plot_cm(clf_name, y_train, y_pred_train, y_test, y_pred_test)

    return train_acc, train_f1, test_acc, test_f1, cm_train, cm_test

def write(writer, cls_name, cm, dataset_split, step):
        """
        Writes confusion matrix to tensorboard
        """
        # For adding cm to tensoboard
        # https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-with-tensorboard-and-pytorch-3344ad5e7209
        # https://www.geeksforgeeks.org/how-to-dump-confusion-matrix-using-tensorboard-logger-in-pytorch-lightning/
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title(cls_name)
        writer.add_figure("Confusion matrix " + dataset_split + " " + cls_name, fig, step)
        plt.close(fig)

def save_metrics_to_dict(clf_name, 
                         train_acc, 
                         train_f1, 
                         test_acc, 
                         test_f1, 
                         train_acc_dict, 
                         train_f1_dict, 
                         test_acc_dict, 
                         test_f1_dict):
        
        train_acc_dict[clf_name] = train_acc
        train_f1_dict[clf_name] = train_f1
        test_acc_dict[clf_name] = test_acc
        test_f1_dict[clf_name] = test_f1

        return train_acc_dict, train_f1_dict, test_acc_dict, test_f1_dict

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
    #print(categorical_cols)
    X.loc[:, categorical_cols] = encode_categorical_features(X[categorical_cols], encoder)
    
    # check if encoding has worked
    # https://stackoverflow.com/questions/26924904/check-if-dataframe-column-is-categorical
    for c in X.columns:
        if X[c].dtype.name == "category":
            print(f"WARNING: Column {c} still has categorical values!")
            
    # last column is target
    return X, y