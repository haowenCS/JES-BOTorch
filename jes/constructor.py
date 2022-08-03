
import math
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from abc import ABC, abstractclassmethod


import numpy as np
import torch
from torch import Tensor
import torch.distributions as dist

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.input_constructors import acqf_input_constructor
#from botorch.utils.datasets import SupervisedDataset
from botorch.acquisition.objective import (
    PosteriorTransform,
)
from botorch.models.model import Model
from jes.jes import JointEntropySearch, GreedyJointEntropySearch



T = TypeVar("T")
MaybeDict = Union[T, Dict[Hashable, T]]


# TODO if the sampler is moved outside the acquisition function (which is should be),
# then the input constructor needs to change to accomodate this
@acqf_input_constructor(JointEntropySearch, GreedyJointEntropySearch)
def construct_inputs_JES(
    model: Model,
    #training_data: MaybeDict[SupervisedDataset], # Not sure what this is used for quite yyet
    bounds: List[Tuple[float, float]],
    posterior_transform: Optional[PosteriorTransform] = None, # Not sure what this is used for quite yyet
    num_opt_samples: int = 100,
    sampler_type: str = 'exact',
    sampler_kwargs: dict = {},
    **kwargs: Any
) -> Dict[str, Any]:
    return {
        'model': model,
        'num_opt_samples': num_opt_samples,
        'sampler_type': sampler_type,
        'sampler_kwargs': sampler_kwargs
    }