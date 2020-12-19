# %% imports
import matplotlib.pyplot as plt
plt.ioff()
import time
import pathlib
import numpy as np
import pandas as pd
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
_code_git_version="acafad223a2da75ccd186cb0e3b3ae4a35c99f57"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="12:51:08 of Saturday, 2020-12-19 (GMT+1)"
start_time=time.time()
fns=list(pathlib.Path("b/").glob("*sample*.pt"))
frameinfo=inspect.getframeinfo(inspect.currentframe())
print("""{:09.5f} {}:{} fns={}""".format(((time.time())-(start_time)), frameinfo.filename, frameinfo.lineno, fns))