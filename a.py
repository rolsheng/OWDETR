from collections import deque
import torch
import numpy as np 
import os 
a = torch.ones((100,31))
entrop = torch.mul(a,a.log2())
print(entrop.shape)