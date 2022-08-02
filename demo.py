import sys
import os
import torch
from jes.bo import bayesian_optimization
from botorch.test_functions import Branin

test_objective = Branin(negate=True)

best_X, best_y, best_guess = bayesian_optimization(test_objective, 25, test_objective.dim, test_objective.bounds)
