import sys
import os
import torch
from jes.bo import bayesian_optimization
from botorch.test_functions import Branin

test_objective = Branin(negate=True)
def obj(X):  
    return test_objective(torch.Tensor([X, 2.275]).reshape(1, -1))

best_X, best_y = bayesian_optimization(obj, 50, 1, torch.tensor([0, 15]).unsqueeze(1))
