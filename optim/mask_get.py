import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
from torch import Generator, nn
from torch.utils.data import random_split, Dataset

from functools import reduce
from thop import profile


def get_gradient_mask(k, reduced_activation_delta, grad_mask, eps):
    grad_mask = evaluated_mask(k, reduced_activation_delta, grad_mask, eps)

def evaluated_mask(k, reduced_activation_delta, grad_mask, eps):
    mask = torch.where(torch.abs(reduced_activation_delta) <= eps)[0]
    grad_mask[k] = mask
    
  