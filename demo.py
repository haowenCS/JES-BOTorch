import sys
import os
import torch
from jes.bo import bayesian_optimization
from botorch.test_functions import Branin

test_objective = Branin(negate=True)

best_X, best_y = bayesian_optimization(test_objective, 50, test_objective.dim, test_objective.bounds)
