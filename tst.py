#!/bin/env -S python -i
import torch,gloria
from datasets.general import CXR_dataset

model = gloria.load_gloria(device='cuda')
dataset = CXR_dataset