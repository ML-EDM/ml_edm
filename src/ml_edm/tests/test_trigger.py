import numpy as np 

from aeon.datasets import load_gunpoint, load_acsf1
from aeon.classification.convolution_based._rocket_classifier import MiniRocket

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import RidgeClassifierCV

from ml_edm.cost_matrice import CostMatrices
from ml_edm.classification.chrono_classifier import ClassifiersCollection
from ml_edm.trigger_models import CALIMERA


def load_ts(n_classes):

    if n_classes == 2:
        X, y = load_gunpoint()
    else:
        X, y = load_acsf1()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, 
                                                        random_state=44, stratify=y)
    lb = LabelEncoder()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)

    data_dict  = {
        "X_train": X_train.squeeze(),
        "y_train": y_train,
        "X_test": X_test.squeeze(),
        "y_test": y_test
    }

    return data_dict

def test_trigger(n_classes, trigger):

    data = load_ts(n_classes)

    X_clf, X_trigger, y_clf, y_trigger = train_test_split(
        data['X_train'], data['y_train'], test_size=0.4, 
        random_state=44, stratify=data['y_train']
    )
    X_clf, X_trigger, y_clf, y_trigger = (
        data["X_train"], data["X_train"], 
        data["y_train"], data["y_train"]
    )

    chrono_clf = ClassifiersCollection(
        base_classifier=RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), 
                                          scoring='accuracy'),
        sampling_ratio=0.05,
        min_length=1,
        feature_extraction={"method": "minirocket", 
                            "params": {"random_state":42}}
    )
    chrono_clf.fit(X_clf, y_clf)

    def delay_cost(t):
        inflexion_point = 0
        return np.exp(((t/data["X_train"].shape[1])-inflexion_point) * np.log(100))
    
    small_values = 1
    misclf_cost = small_values - np.eye(n_classes) * small_values

    classes, counts = np.unique(data["y_train"], return_counts=True)
    idx_min_class = classes[counts.argmin()]
    misclf_cost[:, idx_min_class] *= 100

    cost_matrices = CostMatrices(chrono_clf.timestamps, n_classes, alpha=0.5, 
                                 delay_cost=None, missclf_cost=None)
    
    trigger.models_input_lengths = chrono_clf.timestamps

    X_probas = np.stack(chrono_clf.predict_past_proba(X_trigger))
    trigger.fit(X_trigger, X_probas, y_trigger, cost_matrices)




trigger = CALIMERA(models_input_lengths=np.arange(100))
test_trigger(2, trigger)
