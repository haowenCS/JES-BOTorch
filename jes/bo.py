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
from jes.sampler import RFFSampler, ExactSampler, CanonicalSampler
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

    num_points = 355
    true_obj = torch.Tensor([objective(val)
                            for val in torch.linspace(*bounds, num_points)])
    # No need to get the first guess - we want as many guesses as queries anyway
    guesses = train_X
        
    for i in range(iterations):
        norm_X = normalize(train_X, bounds)
        norm_y = standardize(train_y)

        gp = SingleTaskGP(norm_X, norm_y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp) 
        fit_gpytorch_model(mll, optimizer=fit_gpytorch_torch,
                           options={'disp': False})

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
            acq_function = GreedyJointEntropySearch(
                model=gp, sampler_type='exact', greedy_fraction=1)
        else:
            acq_function = GreedyJointEntropySearch(
                model=gp, sampler_type='exact', num_opt_samples=1000, greedy_fraction=0)
        
        rff_sampler = RFFSampler(gp, num_features=4096)
        exact_sampler = ExactSampler(gp)
        candidate_set = torch.linspace(0, 1, num_points).unsqueeze(-1)
        
        opt_X_rff, opt_y_rff, samples_rff = rff_sampler.sample(
            12, candidate_set=candidate_set, num_append_points=0, return_samples=True)
        opt_X_exact, opt_y_exact, samples_exact = exact_sampler.sample(
            12, candidate_set=candidate_set, num_append_points=0, return_samples=True)
        opt_X, opt_y = acq_function.get_sampled_optima()
        
        posterior = gp.posterior(candidate_set, observation_noise=False)
        mean = posterior.mean
        variance = posterior.variance

        i = 1

        fig, ax = plt.subplots(4-i, 1, figsize=(16, 24), sharex=True)

        def d(X):
            return X.detach().numpy().flatten()

        ax[0].plot(d(candidate_set), d(mean), color='k')
        ax[0].scatter(d(norm_X), d(norm_y), s=50, color='red')
        ax[0].fill_between(d(candidate_set), d(mean) - 2 * d(torch.sqrt(variance)), d(mean) + 2 * d(torch.sqrt(variance)), alpha=0.2, color='k')
        ax[0].plot(d(candidate_set), samples_rff.T.detach().numpy())
        
        ax[1].plot(d(candidate_set), d(mean), color='k')
        ax[1].scatter(d(norm_X), d(norm_y), s=50, color='red')
        ax[1].fill_between(d(candidate_set), d(mean) - 2 * d(torch.sqrt(variance)), d(mean) + 2 * d(torch.sqrt(variance)), alpha=0.2, color='k')
        ax[1].plot(d(candidate_set), samples_exact.T.detach().numpy())
        plt.show()
        # TODO - fix so that some of the raw samples come from X_opt - these points are obviously the promising ones
        norm_point, _ = optimize_acqf(
            acq_function=acq_function,
            bounds=torch.Tensor([[0, 1] for d in range(dim)]).T,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={'nonnegative': True, 'sample_around_best': True},
        )

        new_point = unnormalize(norm_point, bounds)
        new_eval = torch.Tensor([objective(new_point)]).reshape(1, 1)
        train_X = torch.cat([train_X, new_point])
        train_y = torch.cat([train_y, new_eval])
        # TODO - fix so that some of the raw samples come from X_opt - these points are obviously the promising ones

        report_iteration(i, train_X, train_y)

    # The last point is picked greedily anyway
    best_arg = train_y.argmax()
    return train_X[best_arg, :], train_y[best_arg, :], 0
