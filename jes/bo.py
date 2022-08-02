import sys
import os
from time import time

import pandas
import torch
import matplotlib.pyplot as plt
from torch.quasirandom import SobolEngine
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.analytic import PosteriorMean
from botorch.optim.fit import fit_gpytorch_torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, standardize, unnormalize
from jes.sampler import RFFSampler, ExactSampler
from jes.jes import JointEntropySearch, GreedyJointEntropySearch
from jes.utils import NUM_RESTARTS, RAW_SAMPLES, report_iteration, plot_points

def bayesian_optimization(objective, iterations, dim, bounds, n_optima=100):
    doe = SobolEngine(dim)
    init_samples = dim + 2
    train_X = unnormalize(doe.draw(init_samples), bounds).double()
    train_y = torch.zeros(init_samples).double()

    for i in range(init_samples):
        train_y[i] = objective(train_X[i, :])
    train_y = train_y.unsqueeze(-1)
    
    # No need to get the first guess - we want as many guesses as queries anyway
    guesses = train_X[1:]

    for i in range(iterations):
        norm_X = normalize(train_X, bounds)
        norm_y = standardize(train_y)

        gp = SingleTaskGP(norm_X, norm_y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll, optimizer=fit_gpytorch_torch, options={'disp': False})

        # checking the best guess by maximizing the posterior mean
        pm = PosteriorMean(gp)
        guess, _ = optimize_acqf(
            acq_function=pm,
            bounds=torch.Tensor([[0, 1] for d in range(dim)]).T,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={'nonnegative': True, 'sample_around_best': True},
        )
        guesses = torch.cat([guesses, unnormalize(guess, bounds)])
        
        start = time()
        if i == iterations - 1: 
            # the last iteration, we always want the best guess that the model has to offer
            acq_function = GreedyJointEntropySearch(model=gp, sampler_type='exact', greedy_fraction=1)
        else:
            acq_function = GreedyJointEntropySearch(model=gp, sampler_type='exact', num_opt_samples=100)
        
        setup_time = time() - start
        
        # TODO - fix so that some of the raw samples come from X_opt - these points are obviously the promising ones
        norm_point, _ = optimize_acqf(
            acq_function=acq_function,
            bounds=torch.Tensor([[0, 1] for d in range(dim)]).T,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={'nonnegative': True, 'sample_around_best': True},
        )
        opt_time = time() - start  - setup_time

        print('Setup time:', setup_time)
        print('Optimization time:', opt_time)
        new_point = unnormalize(norm_point, bounds)
        new_eval = torch.Tensor([objective(new_point)]).reshape(1, 1)
        train_X = torch.cat([train_X, new_point])
        train_y = torch.cat([train_y, new_eval])
        
        
                # TODO - fix so that some of the raw samples come from X_opt - these points are obviously the promising ones
        

        report_iteration(i, train_X, train_y)

    # The last point is picked greedily anyway
    guesses = torch.cat([guesses, unnormalize(new_point, bounds)])
    best_arg = train_y.argmax()
    best_guess = train_y[-1, :]
    return train_X[best_arg, :], train_y[best_arg, :], guesses
