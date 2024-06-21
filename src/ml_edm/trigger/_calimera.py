import numpy as np
from sklearn.kernel_ridge import KernelRidge

from ._base import BaseTriggerModel

class CALIMERA(BaseTriggerModel):
    """
    CALIMERA: A new early time series classification method
    Inspired by : https://github.com/JakubBilski/CALIMERA 
    """
    def __init__(self,
                 timestamps,
                 n_jobs=1):
        
        super().__init__(timestamps)
        self.n_jobs = n_jobs
    
    def _generate_features(self, probas, time_idx, y):

        max_probas = np.max(probas, axis=-1)
        second_max_probas = np.partition(probas, -2)[:,-2]
        diff = max_probas - second_max_probas

        features = np.concatenate(
            (probas, diff[:,None], max_probas[:,None]), axis=-1
        )
        #delay_cost = self.alpha * self.models_input_lengths[time_idx] / self.max_length
        #costs = 1 - max_probas + delay_cost
        
        ## Hypothesis : max P(y|X) == proba to be in diag and 1 - max P(y|X) == not in diag 
        #
        #delay = np.mean(self.cost_matrices.delay_cost[time_idx])
        #proba_correct = np.mean(np.diagonal(self.cost_matrices.missclf_cost[time_idx])) * max_probas # weight Cd by prob ?
        #non_diag = self.cost_matrices[time_idx] - \
        #    (np.eye(self.n_classes) * np.diagonal(self.cost_matrices[time_idx]))
        #proba_incorrect = np.sum(non_diag) / (self.n_classes**2 - self.n_classes) * (1-max_probas) # weight Cm one by one ?
        #costs = proba_correct + proba_incorrect + delay 

        delay = np.mean(self.cost_matrices.delay_cost[time_idx])
        misclf_cost = [(prob * [self.cost_matrices.missclf_cost[time_idx][prob.argmax()][yy]
                               for yy in self.classes_]).sum() for prob in probas]
        costs = misclf_cost + delay

        return features, costs
    
    def _fit(self, X_probas, y):

        self.max_timestamp_idx = len(self.timestamps)

        results  = [self._generate_features(X_probas[:,t,:], t, y) 
                    for t in range(X_probas.shape[1])]
        
        features, costs = zip(*results)
        features, costs = (np.array(features), np.array(costs))

        self.halters = [None for _ in range(self.max_timestamp_idx-1)]

        for t in range(self.max_timestamp_idx-2, -1, -1):

            X_trigger = features[t]
            y_trigger = costs[t+1] - costs[t] 

            model = KernelRidge(kernel='rbf').fit(X_trigger, y_trigger)
            self.halters[t] = model
            predicted_cost_difference = model.predict(X_trigger)
            for j in range(len(X_trigger)):
                if predicted_cost_difference[j] < 0:
                    costs[t, j] = costs[t+1, j]

        return self 
    
    def _predict(self, X_probas, X_timestamps):

        triggers, self.cost_forecast = [], []
        for i, probas in enumerate(X_probas):
            trigger = False
            # if last timestamp is reached
            if X_timestamps[i] == self.timestamps[-1]:
                triggers.append(True)
                self.cost_forecast.append(np.nan)
                continue

            time_idx = np.where(X_timestamps[i] == self.timestamps)[0][0]
            X_trigger, _ = self._generate_features(probas[None,:], time_idx, 
                                                   np.zeros(X_probas.shape[0], dtype=int))
            
            predicted_cost_diff = self.halters[time_idx].predict(X_trigger)

            if predicted_cost_diff > 0:
                trigger = True
            
            triggers.append(trigger)
            self.cost_forecast.append(predicted_cost_diff[0])

        return np.array(triggers)