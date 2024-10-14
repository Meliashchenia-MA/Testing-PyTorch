import numpy as np
import pandas as pd

from collections import defaultdict
from collections import OrderedDict

import matplotlib.pyplot as plt
from IPython.display import clear_output

import torch
from torch import nn
import torch.nn.functional as F

def func_x(t):
    return 2 * torch.sin(t) + torch.sin(2 * t) * torch.cos(60 * t)

def func_y(t):
    return torch.sin(2 * t) + torch.sin(60 * t)

t = torch.linspace(0, 10, 100)
x = func_x(t)
y = func_y(t)

plt.plot(x.numpy(), y.numpy())
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

