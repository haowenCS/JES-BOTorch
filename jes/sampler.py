from abc import ABC, abstractmethod
import math
from typing import Any, Callable, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.distributions import Normal, Gamma, MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel
from gpytorch.utils.cholesky import psd_safe_cholesky

from torch.quasirandom import SobolEngine


# TODO generalize to arbitrary bounds - now presumed to be 0, 1
# TODO integer and categorical support

class OptSampler(ABC):
    def __init__(self, model) -> None:
        pass

    @abstractmethod
    def sample(self, num_samples) -> Tensor:
        pass


class RFFSampler(OptSampler):

    def __init__(self, model, num_features=1024):
        """A sampler of optimal locations and values based on a Random Fourier feature
        approximation of the posterior distribution.

        Args:
            model (botorch.models.Model): A BOTorch GP.
            num_features (int, optional): The number of features to use for approximating the posterior. Defaults to 1048.

        Raises:
            NotImplementedError: [description]
        """
        self.train_X = model.train_inputs[0]
        self.train_Y = model.train_targets
        self.dim = self.train_X.shape[1]
        self.num_features = num_features
        ls = model.covar_module.base_kernel.lengthscale
        sigma = model.covar_module.outputscale
        gp_noise = model.likelihood.raw_noise
        self.scaling = torch.sqrt(2 * sigma / num_features)
        if isinstance(model.covar_module.base_kernel, RBFKernel):
            self.weights = Normal(0, 1).sample(torch.Size(
                [num_features, self.dim])).to(self.train_X) / ls

        # Adjust weights if Matern
        elif isinstance(model.covar_module.base_kernel, MaternKernel):
            self.weights = Normal(0, 1).sample(torch.Size(
                [num_features, self.dim])).to(self.train_X) / ls
            gamma_weights = Gamma(2.5, 2.5).sample(torch.Size(
                [self.num_features, 1])).to(self.train_X)
            self.weights = self.weights * torch.rsqrt(gamma_weights)

        else:
            raise NotImplementedError(
                'RFFSampler does not support other kernels than RBF and Matérn.')

        # Precompute all the necessary quantities for being able to draw samples
        self.b = math.pi * 2 * torch.rand(self.num_features, 1)
        self.Z = self.scaling * \
            torch.cos(torch.matmul(self.weights, self.train_X.T) + self.b)
        Sigma = torch.matmul(self.Z.T, self.Z) + gp_noise * \
            torch.eye(self.train_X.shape[0])
        self.mu = torch.matmul(torch.matmul(
            self.Z, torch.inverse(Sigma)), self.train_Y).unsqueeze(-1)
        D, self.U = torch.linalg.eig(Sigma)
        # TODO may want a warning here, casting from "imaginary"
        self.D = D.unsqueeze(-1).to(self.train_X)
        self.U = self.U.to(self.train_X)
        self.R = torch.pow(torch.sqrt(self.D) *
                           (torch.sqrt(self.D) + gp_noise), -1)

    def sample(self,
               num_samples: int,
               candidate_set: Union[Tensor, None] = None,
               num_candidate_points: int = 4096,
               num_append_points: int = 20,
               top_fraction: float = 0.25,
               grad_opt: bool = False,
               return_samples: bool = False) -> Tuple[Tensor, Tensor]:
        """Draws a number of optimal locations an their corresponding optimal values from a candidate set of points.

        Args:
            num_samples (int): The number of optimal samples (x, f) to draw from the sampler.
            candidate_set (Tensor, optional): A candidate set of points to query to find the optima. Defaults to None.
            num_candidate_points (int, optional): [description]. The number of Sobol-generated points'
                'to to query to find the optima if there is no candidate_set. Defaults to 4096.
            num_append_points (Tensor, optional): [description].  Number of points near recent queries to append to the candidate_set. Defaults to None.
            grad_opt: (bool, optional): Whether to optimize each sample with a gradient-based optimizer. Defaults to False.

        Returns:
            [Tuple[Tensor, Tensor]]: Tuple of Tensors containing (optimal locations, optimal values).
        """
        if candidate_set is None:
            sobol = SobolEngine(dimension=self.dim)
            candidate_set = sobol.draw(num_candidate_points).to(self.train_X)
        else:
            # assert the input candidate set is of the same type (float/double)
            candidate_set = candidate_set.to(self.train_X)

        if num_append_points > 0:
            # TODO create the append set
            append_set = None
            candidate_set = torch.cat(
                (candidate_set, append_set)).to(self.train_X)

        rho = Normal(0, 1).sample(torch.Size(
            [self.num_features, num_samples])).to(self.train_X)
        theta = rho - torch.matmul(self.Z, (torch.matmul(self.U, (self.R *
                                   torch.matmul(self.U.T, (torch.matmul(self.Z.T, rho))))))) + self.mu
        #theta = self.mu + torch.matmul(torch.linalg.cholesky(self.Sigma), rho)
        
        samples = torch.matmul(
            theta.T * self.scaling, torch.cos(torch.matmul(self.weights, candidate_set.T) + self.b))
        f_max, argmax = samples.max(dim=1)

        if grad_opt:
            raise NotImplementedError(
                'Gradient-based optimization of the RFF samples is not yet implemented.')
        else:
            X_max = candidate_set[argmax, :]
        f_max = f_max.unsqueeze(-1)

        print('X_max.shape', X_max.shape, 'f_max.shape', f_max.shape)
        if return_samples:
            return X_max, f_max, samples
        return X_max, f_max


class ExactSampler(OptSampler):

    def __init__(self, model):
        self.model = model
        self.train_X = model.train_inputs[0]
        self.train_Y = model.train_targets
        self.dim = self.train_X.shape[1]

    def sample(self,
               num_samples: int,
               candidate_set: Union[Tensor, None] = None,
               num_candidate_points: int = 4096,
               num_append_points: int = 20,
               top_fraction: float = 0.25,
               sample_std=0.01,
               return_samples: bool = False) -> Tuple[Tensor, Tensor]:
        """Draws a number of optimal locations an their corresponding optimal values from a candidate set of points.

        Args:
            num_samples (int): The number of optimal samples (x, f) to draw from the sampler.
            candidate_set (Tensor, optional): A candidate set of points to query to find the optima. Defaults to None.
            num_candidate_points (int, optional): The number of Sobol-generated points'
                'to to query to find the optima if there is no candidate_set. Defaults to 4096.
            num_append_points (Tensor, optional): Number of points near recent queries to append to the candidate_set. Defaults to None.
            top_fraction (float, optional): The fraction of previous configurations that should be sampled around and added to the candidate set. Defaults to 0.25.
            sample_std (float, optional): The standard deviation for the points generated by top_fraction. Defaults to 0.02.
        Returns:
            [Tuple[Tensor, Tensor]]: Tuple of Tensors containing (optimal locations, optimal values).
        """
        if candidate_set is None:
            sobol = SobolEngine(dimension=self.dim)
            candidate_set = sobol.draw(num_candidate_points).to(self.train_X)
        else:
            # assert the input candidate set is of the same type (float/double)
            candidate_set = candidate_set.to(self.train_X)
        
        if num_append_points > 0:
            num_top_points = math.ceil(len(self.train_X) * top_fraction)
            # create the append set for better guesses around the best observations
            X_top = self.train_X[self.train_Y.argsort(
                descending=True)[:num_top_points], :]

            # sample the candidates around the best observed and clip them to the presumed input range (0, 1 for now)
            top_candidates = torch.clamp(X_top.repeat(num_append_points, 1) + torch.normal(
                torch.zeros(torch.Size([num_top_points * num_append_points, self.dim])), sample_std), 0, 1)
            candidate_set = torch.cat(
                (candidate_set, top_candidates)).to(self.train_X)

        posterior = self.model.posterior(candidate_set)
        samples = posterior.rsample(
            sample_shape=torch.Size([num_samples])).squeeze(-1)
        f_max, argmax = samples.max(dim=1)
        X_max = candidate_set[argmax, :]
        f_max = f_max.unsqueeze(-1)

        if return_samples:
            return X_max, f_max, samples.squeeze(0)
        return X_max, f_max


class CanonicalSampler(OptSampler):
    def __init__(self, model, num_features=1024):
        """A sampler of optimal locations and values based on a Random Fourier feature
        approximation of the prior distribution, and canonical bases for the posterior update.

        Args:
            model (botorch.models.Model): A BOTorch GP.
            num_features (int, optional): The number of features to use for approximating the posterior. Defaults to 1048.

        Raises:
            NotImplementedError: [description]
        """
        self.train_X = model.train_inputs[0]
        self.train_Y = model.train_targets
        self.dim = self.train_X.shape[1]
        self.num_features = num_features
        ls = model.covar_module.base_kernel.lengthscale
        sigma = model.covar_module.outputscale
        gp_noise = model.likelihood.raw_noise
        self.scaling = torch.sqrt(2 * sigma / num_features)
        if isinstance(model.covar_module.base_kernel, RBFKernel):
            self.weights = Normal(0, 1).sample(torch.Size(
                [num_features, self.dim])).to(self.train_X) / ls

        # Adjust weights if Matern
        elif isinstance(model.covar_module.base_kernel, MaternKernel):
            self.weights = Normal(0, 1).sample(torch.Size(
                [num_features, self.dim])).to(self.train_X) / ls
            gamma_weights = Gamma(2.5, 2.5).sample(torch.Size(
                [self.num_features, 1])).to(self.train_X)
            self.weights = self.weights * torch.rsqrt(gamma_weights)

        else:
            raise NotImplementedError(
                'CanonicalSampler does not support other kernels than RBF and Matérn.')

        # Precompute all the necessary quantities for being able to draw samples
        self.b = math.pi * 2 * torch.rand(self.num_features, 1)
        self.Z = self.scaling * \
            torch.cos(torch.matmul(self.weights, self.train_X.T) + self.b)
        Sigma = torch.matmul(self.Z.T, self.Z) + gp_noise * \
            torch.eye(self.train_X.shape[0])
        self.mu = torch.matmul(torch.matmul(
            self.Z, torch.inverse(Sigma)), self.train_Y).unsqueeze(-1)
        D, self.U = torch.linalg.eig(Sigma)
        # TODO may want a warning here, casting from "imaginary"
        self.D = D.unsqueeze(-1).to(self.train_X)
        self.U = self.U.to(self.train_X)
        self.R = torch.pow(torch.sqrt(self.D) *
                           (torch.sqrt(self.D) + gp_noise), -1)