import numpy as np

def average_cost(accuracy, earliness, alpha):
    return (1-accuracy) + alpha * earliness
