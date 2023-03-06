from kernels import BhattKernel, BhattKernelNodes
from graph import Graph

import numpy as np
import pandas as pd
import pickle

import os

print('0')
from ogb.graphproppred import Evaluator

d_name = "ogbg-molesol"

print('1')

evaluator = Evaluator(name = d_name)

print('2')
print(evaluator.expected_input_format) 
print(evaluator.expected_output_format)  