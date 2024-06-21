import numpy as np
from numbers import Number
from warnings import warn 
import matplotlib.pyplot as plt 

from .utils import check_timestamps

class CostMatrices:
    """
    Object defining the cost values for each type of error 
    for each desired timestamps

    Parameters
    ----------
    timestamps : array-like of shape (n_timestamps,)
        Input timestamps, one matrix will be instanciated 
        for each of the imput timesstamps.
    
    n_classes : int
        Number of classes ; each of the matrix within the
        object will have the shape (n_classes, n_classes).
    
    all_matrices : array-like of shape (n_timestamps, n_classes, n_classes), \
            default=None
        Option to manually defined all cost matrices. 
    
    missclf_cost : array-like of shape (n_classes, n_classes), int or float, \
            default=1
        Cost to be paid when making a wrong prediction.
        If int or float the cost is the same for all type
        of errors. For asymetric costs define an array-like.
         
        Note that misclassification cost cannot be time-dependant for now.

    delay_cost : array-like of shape (n_timestamps,) or callable function of timestamps, \
            default=None
        Cost to be paid for waiting new measurements.
        If callable function should be only take one argument, i.e. timestamp t.

    """

    def __init__(self,
                 timestamps,
                 n_classes,
                 all_matrices=None, 
                 missclf_cost=None,
                 delay_cost=None,
                 alpha=0.5):
        
        self.alpha = alpha
        self.timestamps = check_timestamps(timestamps)

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
            self.missclf_cost = [alpha * missclf_matrix for _ in range(len(self.timestamps))]


            if isinstance(delay_cost, np.ndarray) or isinstance(delay_cost, list):
                self.delay_cost = [np.ones((n_classes, n_classes)) * delay_cost[i] 
                                   for i in range(len(self.timestamps))]
            elif callable(delay_cost):
                self.delay_cost = [np.ones((n_classes, n_classes)) * delay_cost(t) 
                                   for t in self.timestamps]
            elif delay_cost is None:
                warn("No delay cost defined, using default linear delay function")
                self.delay_cost = [np.ones((n_classes, n_classes)) * t/self.timestamps[-1] 
                                   for t in self.timestamps]
            else:
                raise ValueError("Delay cost should be defined as a callable function of time or as a"
                                "list/array of numbers whose length is the number of timestamps")
            self.delay_cost = [(1 - alpha) * costs for costs in self.delay_cost]
            self.values = np.array([self.missclf_cost[i] + self.delay_cost[i]
                                    for i in range(len(self.timestamps))])
    
    def __getitem__(self, index):
        return self.values[index]

    def __len__(self):
        return len(self.values)
    
    """
    def plot(self, alphas=[0, 0.2, 0.5, 0.8]):

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,6))

        # plot delay cost
        t = np.linspace(0, 1, 1000)

        axes[0].ylim(0, np.max(self.delay_cost))
        axes[0].xlim(0, 1)

        for alpha in alphas:
            axes[0].plot(t, self.delay_cost)
    """