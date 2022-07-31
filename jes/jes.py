
from math import log
from typing import Any, Callable, Optional
from abc import ABC, abstractclassmethod


import numpy as np
import torch
from torch import Tensor

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.models.utils import check_no_nans
from botorch.sampling.samplers import SobolQMCNormalSampler
from sampler import OptSampler, RFFSampler


CLAMP_LB = 1.0e-8


class JointEntropySearch(AcquisitionFunction):

    def __init__(self,
                 model: Model,
                num_fantasies: int = 16,
                num_mv_samples: int = 10,
                sampler_type: OptSampler = RFFSampler) -> None:
