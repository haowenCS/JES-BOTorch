import numpy as np
import torch
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
    plt.scatter(train_X.detach().numpy(), train_y.detach().numpy(), color='k', s=100)
    posterior = gp.posterior(plot_X)
    mean, var = posterior.mean, posterior.variance
    plt.plot(plot_X.detach().numpy(), mean.detach().numpy())
    plt.fill_between(plot_X.detach().numpy(),
                 (mean - 2 * torch.sqrt(var)).detach().flatten().numpy(), (mean + 2 * torch.sqrt(var)).detach().flatten().numpy(), alpha=0.2)
    
    if samples is not None:
        plt.plot(np.linspace(0, 1, len(samples.T)), samples[0:3, :].T.detach().numpy())
    plt.show()