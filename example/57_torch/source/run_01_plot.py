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
_code_git_version="ed05a1f99b530363e4f7ae487ac343453a040966"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="12:55:42 of Saturday, 2020-12-19 (GMT+1)"
start_time=time.time()
fns=list(pathlib.Path("b/").glob("*sample*.pt"))
for fn in fns:
    frameinfo=inspect.getframeinfo(inspect.currentframe())
    print("""{:09.5f} {}:{} fn={}""".format(((time.time())-(start_time)), frameinfo.filename, frameinfo.lineno, fn))
    module=torch.jit.load(str(fn))
    images=list(module.parameters())[0]
    dimension=3
    for index in range(((dimension)*(dimension))):
        image=images[index].detach().cpu().reshape(28, 28).mul(255).to(torch.uint8)
        array=image.numpy()
        ax=plt.subplot(dimension, dimension, ((1)+(index)))
        plt.imshow(array, cmap="gray")
    plt.savefig("{}.png".format(fn.stem))