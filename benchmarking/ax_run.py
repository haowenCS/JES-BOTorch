import sys
sys.path.append('.')

import torch
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from botorch.test_functions import Branin, Hartmann, Ackley
from botorch.acquisition import qMaxValueEntropy, qKnowledgeGradient, qExpectedImprovement, qNoisyExpectedImprovement
#from botorch.acquisition.input_constructors import ( 
#    construct_inputs_qEI, 
#    construct_inputs_qNEI, 
#    construct_inputs_qNEI, 
#    construct_inputs_qKG
#)

from jes.jes import GreedyJointEntropySearch
from jes.constructor import construct_inputs_JES

TEST_FUNCTIONS = {
    'branin': Branin(noise_std=1, negate=True),
    'ackley': Ackley(dim=4, noise_std=1, negate=True),
    'hartmann3': Hartmann(dim=3, noise_std=1, negate=True),
    'hartmann6':  Hartmann(dim=6, noise_std=1, negate=True)
}

ACQUISITION_FUNCTIONS = {
    'JES': GreedyJointEntropySearch,
    'MES': qMaxValueEntropy,
    'EI': qExpectedImprovement,
    'NEI': qNoisyExpectedImprovement,
    'KG': qKnowledgeGradient
}

experiment, acqfunc_name, seed = sys.argv[1:]
test_function = TEST_FUNCTIONS[experiment]
acq_func = ACQUISITION_FUNCTIONS[acqfunc_name]
gs = GenerationStrategy(
    steps=[
        # Quasi-random initialization step
        GenerationStep(
            model=Models.SOBOL,
            # How many trials should be produced from this generation step
            num_trials=test_function.dim + 1,
            # Any kwargs you want passed into the model
            model_kwargs={"seed": int(seed)},
        ),
        # Bayesian optimization step using the custom acquisition function
        GenerationStep(
            model=Models.BOTORCH_MODULAR,
            num_trials=test_function.dim * 49 - 1,  # 50D iterations in total
            # For `BOTORCH_MODULAR`, we pass in kwargs to specify what surrogate or acquisition function to use.
            # `acquisition_options` specifies the set of additional arguments to pass into the input constructor.
            model_kwargs={
                "botorch_acqf_class": acq_func,
                "acquisition_options": {},
            },
        ),
    ]
)

# Initialize the client - AxClient offers a convenient API to control the experiment
ax_client = AxClient(generation_strategy=gs)
# Setup the experiment
ax_client.create_experiment(
    name=f"{experiment}_{acqfunc_name}_run{seed}",
    parameters=[
        {
            "name": f"x_{i+1}",
            "type": "range",
            # It is crucial to use floats for the bounds, i.e., 0.0 rather than 0.
            # Otherwise, the parameter would 
            "bounds": test_function.bounds[:, i].tolist(),
        }
        for i in range(test_function.dim)
    ],
    objectives={
        experiment: ObjectiveProperties(minimize=True),
    },
)


def evaluate(parameters):
    x = torch.tensor([[parameters.get(f"x_{i+1}") for i in range(test_function.dim)]])
    bc_eval = test_function(x).squeeze().tolist()
    # In our case, standard error is 0, since we are computing a synthetic function.
    return {experiment: bc_eval}

for i in range(50 * test_function.dim):
    parameters, trial_index = ax_client.get_next_trial()
    # Local evaluation here can be replaced with deployment to external system.
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

results_df = ax_client.get_trials_data_frame()
configs = torch.tensor(results_df.loc[:, ['x_' in col for col in results_df.columns]].to_numpy())
results_df['True Eval'] = test_function.evaluate_true(configs)

results_df.to_csv(f"{ax_client.experiment.name}.csv")
