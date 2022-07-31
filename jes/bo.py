import sys
import os

import pandas
import torch
from torch.quasirandom import SobolEngine
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedImprovement
from botorch.optim.fit import fit_gpytorch_torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, standardize, unnormalize
from jes.sampler import RFFSampler, SimpleRFFSampler
from jes.utils import NUM_RESTARTS, RAW_SAMPLES, report_iteration, plot_points

def bayesian_optimization(objective, iterations, dim, bounds, n_optima=100):
    doe = SobolEngine(dim)
    init_samples = dim + 2
    train_X = unnormalize(doe.draw(init_samples), bounds).double()
    train_y = torch.zeros(init_samples).double()

    for i in range(init_samples):
        train_y[i] = objective(train_X[i, :])
    train_y = train_y.unsqueeze(-1)

    for i in range(iterations):
        norm_X = normalize(train_X, bounds)
        norm_y = standardize(train_y)

        gp = SingleTaskGP(norm_X, norm_y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll, optimizer=fit_gpytorch_torch, options={'disp': False})
        best_f = norm_y.max()

        sampler = RFFSampler(gp)
        candidate_set = torch.linspace(0, 1, 169).unsqueeze(-1).to(train_X)
        optimal_X, optimal_y, samples = sampler.sample(2, candidate_set=candidate_set)
        
        plot_points(optimal_X, optimal_y, gp, samples)
        acq_function = ExpectedImprovement(model=gp, best_f=best_f)
        norm_point, _ = optimize_acqf(
            acq_function=acq_function,
            bounds=torch.Tensor([[0, 1] for d in range(dim)]).T,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={},
        )
        print('norm_point', norm_point)
        new_point = unnormalize(norm_point, bounds)
        new_eval = torch.Tensor([objective(new_point)]).reshape(1, 1)
        train_X = torch.cat([train_X, new_point])
        train_y = torch.cat([train_y, new_eval])

        report_iteration(i, train_X, train_y)
    return train_y.max()
