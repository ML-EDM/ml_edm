import numpy as np

from numbers import Number
from warnings import warn 

class CostMatrices:

    def __init__(self,
                 timestamps,
                 n_classes,
                 all_matrices=None, 
                 missclf_cost=None,
                 delay_cost=None,
                 alpha=1.):
        
        self.alpha = alpha
        if all_matrices:
            self.values = np.array(all_matrices)
            self.delay_cost = [self.values[i] - self.values[0] 
                               for i in range(len(all_matrices))]
            self.missclf_cost = [self.values[i] - self.delay_cost[i] 
                                 for i in range(len(all_matrices))]

            if (missclf_cost or delay_cost) is not None:
                warn("When giving all costs matrices, every other parameters is ignored")
        else:
            if isinstance(missclf_cost, np.ndarray):
                missclf_matrix = missclf_cost
            elif isinstance(missclf_cost, Number):
                missclf_matrix = (np.zeros((n_classes, n_classes)) + missclf_cost) * (1-np.eye(n_classes))
            elif missclf_cost is None:
                warn("No missclassification cost defined, using default binary set up," 
                    "bad classification cost =  1, 0 otherwise")
                missclf_matrix = 1 - np.eye(n_classes)
            else:
                raise ValueError("Missclassification cost should be defined"
                                "as an numpy array or as a number")
            self.missclf_cost = [alpha * missclf_matrix for _ in range(len(timestamps))]


            if isinstance(delay_cost, np.ndarray) or isinstance(delay_cost, list):
                self.delay_cost = [np.ones((n_classes, n_classes)) * delay_cost[i] 
                                   for i in range(len(timestamps))]
            elif callable(delay_cost):
                self.delay_cost = [np.ones((n_classes, n_classes)) * delay_cost(t) 
                                   for t in timestamps]
            elif delay_cost is None:
                warn("No delay cost defined, using default linear delay function")
                self.delay_cost = [np.ones((n_classes, n_classes)) * t/timestamps[-1] 
                                   for t in timestamps]
            else:
                raise ValueError("Delay cost should be defined as a callable function of time or as a"
                                "list/array of numbers whose length is the number of timestamps")
            self.delay_cost = [(1 - alpha) * costs for costs in self.delay_cost]
            self.values = np.array([self.missclf_cost[i] + self.delay_cost[i]
                                    for i in range(len(timestamps))])
    
    def __getitem__(self, index):
        return self.values[index]

    def __len__(self):
        return len(self.values)

"""
import random
def my_func(t):
    return t/24
rnd = [np.random.rand(2,2) for _ in range(24)]
a = CostMatrices(n_classes=2, timestamps=np.arange(1,25))
print(a[0])
"""