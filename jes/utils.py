import numpy as np

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