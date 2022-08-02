
import math
from typing import Any, Callable, Optional
from abc import ABC, abstractclassmethod


import numpy as np
import torch
from torch import Tensor
import torch.distributions as dist

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import PosteriorMean
from botorch.models.model import Model
from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.models.utils import check_no_nans
from botorch.sampling.samplers import SobolQMCNormalSampler
from jes.sampler import OptSampler, RFFSampler, ExactSampler
from jes.utils import batchify_state_dict, compute_truncated_variance

CLAMP_LB = 1.0e-8
MIN_VAR = 1e-6


class JointEntropySearch(AcquisitionFunction):

    def __init__(self,
                 model: Model,
                 num_opt_samples: int = 100,
                 sampler_type: str = 'exact',
                 sampler_kwargs: dict = {}
                 ) -> None:

        super(JointEntropySearch, self).__init__(model=model)
        self.num_opt_samples = num_opt_samples

        if sampler_type == 'exact':
            self.sampler = ExactSampler(self.model)
        elif sampler_type == 'rff':
            self.sampler = RFFSampler(
                self.model, num_features=1024, **sampler_kwargs)
        else:
            raise NotImplementedError(
                f'The OptSampler type {sampler_type} is not available.')

        self.X_opt, self.f_opt = self.sampler.sample(num_opt_samples)
        self.noise_var = self.model.likelihood.noise[0]
        train_X = model.train_inputs[0]
        train_Y = model.train_targets
        batch_train_X = train_X.repeat(num_opt_samples, 1, 1)
        batch_train_Y = train_Y.repeat(num_opt_samples, 1).unsqueeze(-1)
        if isinstance(model, SingleTaskGP):
            batch_model = SingleTaskGP(batch_train_X, batch_train_Y)

        elif isinstance(model, FixedNoiseGP):
            batch_yvar = fix_gp.likelihood.noise.repeat(num_opt_samples, 1, 1)
            batch_model = FixedNoiseGP(
                batch_train_X, batch_train_Y, batch_yvar)
        else:
            raise NotImplementedError(
                f'JES is not implemented for model type {type(self.model)}')
        batch_model.load_state_dict(batchify_state_dict(
            self.model, self.num_opt_samples))

        # Predict something because we must before we can condition
        batch_model.posterior(self.model.train_inputs[0])
        self.conditioned_batch_model = batch_model.condition_on_observations(
            self.X_opt.unsqueeze(1), self.f_opt.unsqueeze(1), noise=torch.ones_like(self.f_opt.unsqueeze(-1)) * 1e-6)

    def forward(self, X) -> Tensor:
        """Computes the Joint Entropy Search acquisition function
        Args:
            X (Tensor): The array of points to query on the acquisition function
        Returns:
            Tensor: The JES acquisition function values at the points X.
        """
        # everything needs squeezing because we always add extra dimensions inside due to the batching 
        base_entropy = self.compute_base_entropy(X.squeeze(1))
        conditional_entropy = self.compute_conditional_entropy(X.squeeze(1))
        res = (base_entropy - conditional_entropy.mean(axis=0)).squeeze(1)

        return res

    def compute_base_entropy(self, X: Tensor) -> Tensor:
        """Computes the entropy of the normal distribution at point(s) X, noise included
        Args:
            X (Tensor): The array of points to compute the entropy of a normal distribution on
        Returns:
            Tensor: The entropy at the points X
        """
        posterior = self.model.posterior(X, observation_noise=True)
        variance = posterior.variance.clamp_min(MIN_VAR)
        entropy = 0.5 * torch.log(2 * math.pi * variance) + 0.5
        return entropy

    def compute_conditional_entropy(self, X: Tensor) -> Tensor:
        """Computes the entropy of the normal distribution at point(s) X, noise included
        Args:
            X (Tensor): The array of points to compute the moment matched truncated Gaussian entropy
        Returns:
            Tensor: The entropy at the points X
        """
        # This argument is default, but just to clarify that noise is usually
        # not included in the posterior...
        posterior = self.conditioned_batch_model.posterior(
            X, observation_noise=False)
        conditional_batch_mean = posterior.mean
        conditional_batch_variance = posterior.variance

        reshaped_f_opt = self.f_opt.repeat(1, X.shape[0]).unsqueeze(-1)
        truncated_variances = self.compute_truncated_variance(
            conditional_batch_mean, conditional_batch_variance, reshaped_f_opt)
        reduced_entropy = 0.5 * \
            torch.log(2 * math.pi * (self.noise_var +
                      truncated_variances)) + 0.5

        return reduced_entropy

    # TODO consider moving somewhere else, there are no attributes used
    def compute_truncated_variance(self, mean, variance, upper_truncation) -> Tensor:
        """Computes the variance after the truncation of the conditioned distribution.
        This method tends to have some slight numerical issues, which is the leading
        reason why to run with double precision.
        Args:
            mean (Tensor): A vector of posterior means from the normal distribution to truncate 
            variance (Tensor): A vector of posterior variances from the normal distribution to truncate
            upper_truncation (Tensor): A vector of upper truncation bounds
        Returns:
            Tensor: The variance after truncation.
        """
        norm_dist = dist.Normal(0, 1)
        beta = (upper_truncation - mean) / torch.sqrt(variance)

        # Truncate the maximum and minimum amount of truncation (determined by the Z-value of the inputs)
        beta = beta.clamp_min(math.log10(CLAMP_LB)).clamp_max(
            -math.log10(CLAMP_LB))
        density_beta = torch.exp(norm_dist.log_prob(beta))

        # This risks being zero, as some points simply cannot be the optimum (reasonably)
        # under a certain realization. As such, the truncation is practically non-existant
        Z = norm_dist.cdf(beta)
        relative_variance_reduction = beta * \
            density_beta / Z + torch.pow(density_beta / Z, 2)
        trunc_variance = variance * (1 - relative_variance_reduction)
        return trunc_variance

    def get_sampled_optima(self):
        return self.X_opt, self.f_opt


# TODO make eps-greedy JES by subclassing it and change forward
class GreedyJointEntropySearch(JointEntropySearch):

    def __init__(self,
                 model: Model,
                 greedy_fraction: float = 0.1,
                 num_opt_samples: int = 100,
                 sampler_type: str = 'exact',
                 sampler_kwargs: dict = {}
                 ) -> None:
        super(GreedyJointEntropySearch, self).__init__(
            model,
            num_opt_samples,
            sampler_type,
            sampler_kwargs
            )
        if np.random.uniform() < greedy_fraction:
            self.greedy = True
            self.pm = PosteriorMean(model)
            
        else:
            self.greedy = False
            
    def forward(self, X):
        if self.greedy:
            return self.pm.forward(X)

        else:
            return super(GreedyJointEntropySearch, self).forward(X)