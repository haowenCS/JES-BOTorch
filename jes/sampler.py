from abc import ABC, abstractmethod
import math
import torch 
from torch import Tensor
from torch.distributions import Normal, Gamma, MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel
from gpytorch.utils.cholesky import psd_safe_cholesky

from torch.quasirandom import SobolEngine


class OptSampler(ABC):
    def __init__(self, model) -> None:
        pass

    @abstractmethod
    def sample(self, num_samples) -> Tensor:
        pass


class RFFSampler(Sampler):

    def __init__(self, model, num_features=1001):
    
        self.X_train = model.train_inputs[0]
        self.y_train = model.train_targets
        self.num_features = num_features
        ls = model.covar_module.base_kernel.lengthscale
        sigma = model.covar_module.outputscale
        gp_noise = model.likelihood.raw_noise
        self.scaling = torch.sqrt(2 * sigma / num_features)
        if (model.covar_module.base_kernel, RBFKernel):
            self.weights = Normal(0, 1).sample(torch.Size([num_features, self.X_train.shape[1]])).to(self.X_train) / ls
        
        # Adjust weights if Matern
        elif isinstance(model.covar_module.base_kernel, MaternKernel):
            weights = Normal(0, 1).sample(torch.Size([self.num_features, X_train.shape[1]])).to(self.X_train)
            gamma_weights = Gamma(2.5, 2.5).sample(torch.Size([self.num_features, self.X_train.shape[1]])).to(self.X_train)
            self.weights = weights * gamma_weights
        
        else:
            raise NotImplementedError('RFFSampler does not support other kernels than RBF and Matérn.')
        
        self.b = math.pi * 2 * torch.rand(self.num_features, 1)
        self.Z = self.scaling * torch.cos(torch.matmul(self.weights, self.X_train.T) + self.b)
        print(ls)
        print('self.Z.mean()', self.Z.mean(dim=0))
        print('self.Z.std()', self.Z.std(dim=0))
        Sigma = torch.matmul(self.Z.T, self.Z) + gp_noise * torch.eye(self.X_train.shape[0])
        self.mu = torch.matmul(torch.matmul(self.Z, torch.inverse(Sigma)), self.y_train).unsqueeze(-1)
        D, self.U = torch.linalg.eig(Sigma)
        self.D = D.unsqueeze(-1).to(self.X_train)
        self.U = self.U.to(self.X_train)
        self.R = torch.pow(torch.sqrt(self.D) * (torch.sqrt(self.D) + gp_noise), -1)
        #self.mu = (torch.matmul(torch.matmul(self.Z, Sigma), self.y_train) / gp_noise).unsqueeze(-1);
        
        
        #self.Sigma = torch.inverse(torch.matmul(self.Z, self.Z.T) / gp_noise + torch.eye(self.num_features))
        #self.mu = (torch.matmul(torch.matmul(self.Sigma, self.Z), self.y_train) / gp_noise).unsqueeze(-1)

    def sample(self, num_samples, candidate_set=None, num_candidate_points=131, append_set=None):
        if candidate_set is None:
            sobol = SobolEngine(self.X_train.shape[1])
            candidate_set = sobol.draw(num_candidate_points).to(self.X_train)
        if append_set is not None:
            candidate_set = torch.cat((candidate_set, append_set)).to(self.X_train)
        
        rho = Normal(0, 1).sample(torch.Size([self.num_features, num_samples])).to(self.X_train)
        theta = rho - torch.matmul(self.Z, (torch.matmul(self.U, (self.R * torch.matmul(self.U.T, (torch.matmul(self.Z.T, rho))))))) + self.mu
        #theta = self.mu + torch.matmul(torch.linalg.cholesky(self.Sigma), rho)
        samples = torch.matmul(theta.T * self.scaling, torch.cos(torch.matmul(self.weights, candidate_set.T) + self.b))
        y_max, argmax = samples.max(dim=1)  
        X_max = candidate_set[argmax, :]
        return X_max, y_max, samples


class PosteriorSampler(OptSampler):
    pass


