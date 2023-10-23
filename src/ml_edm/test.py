# Package imports
from classification import EarlyClassifier, ChronologicalClassifiers
from trigger_models import EconomyGamma

# Other imports
import copy
import numpy as np
from aeon.datasets import load_classification
from sktime import datasets
from sktime.classification.dictionary_based import WEASEL
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import cohen_kappa_score
import xgboost as xgb
from metrics import average_cost


def test():

    #X, Y, meta = load_classification("Italian") # Get dataset

    #y = np.array([label for i, label in enumerate(Y) if X[i].shape[1]==24]).astype(int) - 1
    #X = np.array([x for x in X if x.shape[1]==24]).squeeze()

    X, y = datasets.load_UCR_UEA_dataset("ItalyPowerDemand")
    X  = np.vstack(X["dim_0"].apply(lambda x: x.values)) # Convert to numpy array
    # separate train test
    test_index = int(.7*len(X))
    
    y = y.astype(int) - 1
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=44)
    #train_x[np.isnan(train_x)] = 0
    #test_x[np.isnan(test_x)] = 0
    """
    train_x, train_y = X[:test_index], y[:test_index].astype(int)
    test_x, test_y = X[test_index:], y[test_index:].astype(int)
    train_y -= 1
    test_y -= 1
    """

    misclassification_cost = [[0, 1], [1, 0]]
    #misclassification_cost = np.ones((10, 10)) - np.eye(10)

    def delay_cost(series_length):
        return series_length / train_x.shape[1]

    clf = WEASEL(support_probabilities=True)

    ec = EarlyClassifier(misclassification_cost, delay_cost, 
                     nb_intervals=9, 
                     nb_classifiers=24,
                     base_classifier=xgb.XGBClassifier(),
                     min_length=1,
                     trigger_model=None)
    ec.fit(train_x, train_y, val_proportion=0.7)
    
    t_post, f_post = ec.get_post(test_x, test_y, use_probas=True)
    dict_res = ec.score(test_x, test_y, return_metrics=True)
    score(ec, test_x, test_y)

    test_x_with_one_ts = [ts[:1] for ts in test_x]
    classes, probas, triggers, costs = ec.predict(test_x_with_one_ts)

    print(costs)

def score(ec, X_test, y_test):

    past_trigger = np.zeros((X_test.shape[0], )).astype(bool)
    trigger_mask = np.zeros((X_test.shape[0], )).astype(bool)
    all_preds = np.zeros((X_test.shape[0], )) - 1
    all_costs = copy.deepcopy(all_preds)
    all_t_star = copy.deepcopy(all_preds)
    for l in ec.chronological_classifiers.models_input_lengths:

        classes, probas, triggers, _ = ec.predict(X_test[:, :l])
        
        trigger_mask = triggers
        # already made predictions
        if past_trigger.sum() != 0: 
            trigger_mask[np.where(past_trigger==True)] = False

        all_preds[trigger_mask] = classes[trigger_mask]
        #all_costs[trigger_mask] = np.array(costs)[trigger_mask][:, 0]
        all_t_star[trigger_mask] = l

        past_trigger = past_trigger | triggers

        # if all TS have been triggered 
        if past_trigger.sum() == X_test.shape[0]:
            break

        if l == ec.chronological_classifiers.models_input_lengths[-1]:
            all_t_star[np.where(all_t_star < 0)] = l
            all_preds[np.where(all_preds < 0)] = classes[np.where(all_preds < 0)]
    
    acc = (all_preds==y_test).mean()
    earl = np.mean(all_t_star) / X_test.shape[1]
    #avg_cost = np.mean(all_costs)
    med_t_star = np.median(all_t_star)

    avg_cost = average_cost(acc, earl, 1/10)

    return acc, earl, avg_cost

test()