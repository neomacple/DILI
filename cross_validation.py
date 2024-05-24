import numpy as np
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


def split_data(k, seed):
    skf = StratifiedKFold(n_splits=k, random_state=seed, shuffle=True)
    return skf


def adaboost_model(estimator=None, n_estimators=50, learning_rate=1.0, random_state=123):
    adc = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    return adc

def cv_adaboost_model(k, dataset, n_estimators, learning_rate, random_state, base_model):
    skf = split_data(k, random_state)
    X_data, y_data = dataset
    scores = {}
    for idx, (train_ids, val_ids) in enumerate(skf.split(X_data, y_data)):
        model = adaboost_model(estimator=base_model, n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
        X_train = X_data[train_ids]
        y_train = y_data[train_ids]
        X_val = X_data[val_ids]
        y_val = y_data[val_ids]
        model.fit(X_train, y_train)
        tr_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        scores[idx] = [tr_score, val_score]
        del model
    return scores



def rf_model(min_split=2, min_leaf=1, random_state=123, max_depth=None):
    rfc = RandomForestClassifier(criterion='entropy',
                                 max_depth=max_depth,
                                 min_samples_split = min_split,
                                 min_samples_leaf = min_leaf,
                                 random_state=random_state)
    return rfc

def cv_random_forest(k, dataset, min_split, min_leaf, random_state, max_depth=None):
    skf = split_data(k, random_state)
    X_data, y_data = dataset
    scores = {}
    for idx, (train_ids, val_ids) in enumerate(skf.split(X_data, y_data)):
        model = rf_model(min_split=min_split, min_leaf=min_leaf, random_state=random_state, max_depth=max_depth)
        X_train = X_data[train_ids]
        y_train = y_data[train_ids]
        X_val = X_data[val_ids]
        y_val = y_data[val_ids]
        model.fit(X_train, y_train)
        tr_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        scores[idx] = [tr_score, val_score]
        del model
    return scores
        


def knn_model(n_neigh=5, weights='distance', leaf_size=30, metric='jaccard'):
    knn = KNeighborsClassifier(
        n_neighbors=n_neigh,
        weights = weights,
        leaf_size = leaf_size,
        metric = metric
    )
    return knn

def cv_knn(k, dataset, neighbor, leaf, seed):
    skf = split_data(k, seed)
    X_data, y_data = dataset
    scores = {}
    for idx ,(train_ids, val_ids) in enumerate(skf.split(X_data, y_data)):
        model = knn_model(n_neigh=neighbor, leaf_size=leaf)
        X_train = X_data[train_ids]
        y_train = y_data[train_ids]
        X_val = X_data[val_ids]
        y_val = y_data[val_ids]
        model.fit(X_train, y_train)
        tr_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        scores[idx] = [tr_score, val_score]
        del model
    return scores



def dt_model(depth=None, min_split=2, min_leaf=1, seed=123):
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=depth, 
                                 min_samples_leaf=min_leaf, 
                                 random_state=seed, 
                                 class_weight='balanced')
    return clf

def cv_decision_tree(k, dataset, min_split, min_leaf, seed, depth=None):
    skf = split_data(k, seed)
    X_data, y_data = dataset
    scores = {}
    for idx ,(train_ids, val_ids) in enumerate(skf.split(X_data, y_data)):
        model = dt_model(depth=depth, min_split=min_split, min_leaf=min_leaf, seed=seed)
        X_train = X_data[train_ids]
        y_train = y_data[train_ids]
        X_val = X_data[val_ids]
        y_val = y_data[val_ids]
        model.fit(X_train, y_train)
        tr_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        scores[idx] = [tr_score, val_score]
        del model
    return scores

