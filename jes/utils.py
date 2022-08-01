from copy import copy
import numpy as np
import torch
from torch import Tensor
from botorch.models import SingleTaskGP, FixedNoiseGP


RAW_SAMPLES = 2048
NUM_RESTARTS = 20
NUM_FEATURES = 1024


def report_iteration(iteration, X, y):
    best = y.max()
    current_y = np.round(y[-1, :].detach().numpy(), 5)[0]
    current_X = np.round(X[-1, :].detach().numpy(), 3).tolist()
    report_string = f'Iteration {iteration}: --- X: {current_X} --- y: {current_y}'
    if best == current_y:
        report_string += '    New best!'

    print(report_string)


def plot_points(X, y, gp, samples=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16, 12))
    plt.scatter(X.detach().numpy(), y.detach().numpy(), color='red', s=30)
    train_X = gp.train_inputs[0]
    train_y = gp.train_targets
    plot_X = torch.linspace(0, 1, 251)
    plt.scatter(train_X.detach().numpy(),
                train_y.detach().numpy(), color='k', s=100)
    posterior = gp.posterior(plot_X)
    mean, var = posterior.mean, posterior.variance
    plt.plot(plot_X.detach().numpy(), mean.detach().numpy())
    plt.fill_between(plot_X.detach().numpy(),
                     (mean - 2 * torch.sqrt(var)).detach().flatten().numpy(), (mean + 2 * torch.sqrt(var)).detach().flatten().numpy(), alpha=0.2)

    if samples is not None:
        plt.plot(np.linspace(0, 1, len(samples.T)),
                 samples[0:3, :].T.detach().numpy())
    plt.show()


def batchify_state_dict(single_obj_gp, num_opt_samples):
    if isinstance(single_obj_gp, SingleTaskGP):
        required_batch_updates = ['likelihood.noise_covar.raw_noise',
                                  'mean_module.constant',
                                  'covar_module.raw_outputscale',
                                  'covar_module.base_kernel.raw_lengthscale']
    elif isinstance(single_obj_gp, FixedNoiseGP):
        required_batch_updates = ['mean_module.constant',
                                  'covar_module.raw_outputscale',
                                  'covar_module.base_kernel.raw_lengthscale']
    single_state_dict = single_obj_gp.state_dict()
    multi_state_dict = copy(single_obj_gp.state_dict())
    for update_name in required_batch_updates:
        old_param_value = single_state_dict.get(update_name)
        new_param_value = old_param_value.repeat(torch.Size(
            (num_opt_samples, ) + tuple(1 for i in range(old_param_value.ndim))))
        #print(update_name, new_param_value.shape)
        multi_state_dict[update_name] = new_param_value
    return multi_state_dict

# TODO consider moving somewhere else, there are no attributes used
def compute_truncated_variance(mean, variance, upper_truncation) -> Tensor:
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
    beta = (upper_truncation.unsqueeze(1) - mean) / torch.sqrt(variance)

    # Truncate the maximum and minimum amount of truncation (determined by the Z-value of the inputs)
    beta = beta.clamp_min(torch.log10(CLAMP_LB)).clamp_max(
        -torch.log10(CLAMP_LB))
    density_beta = torch.exp(norm_dist.log_prob(beta))

    # This risks being zero, as some points simply cannot be the optimum (reasonably)
    # under a certain realization. As such, the truncation is practically non-existant
    Z = norm_dist.cdf(beta)
    relative_variance_reduction = beta * \
        density_beta / Z + torch.pow(density_beta / Z, 2)
    trunc_variance = variance * (1 - relative_variance_reduction)
    return trunc_variance