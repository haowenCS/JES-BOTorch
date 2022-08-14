import sys
import os
import torch
from torch import Tensor
from jes.bo import bayesian_optimization
from botorch.test_functions import *

test_objective = Branin(negate=True)

    
obj = lambda x: test_objective(torch.Tensor([x, 2.275]).unsqueeze(0))

best_X, best_y, best_guess = bayesian_optimization(obj, 25, 1, [-5, 10])
