import matplotlib
import matplotlib.pyplot as plt
plt.ion()
font={("size"):("5")}
matplotlib.rc("font", **font)
# %% imports
import sys
import time
import pathlib
import numpy as np
import pandas as pd
s=np.memmap("./o_15283_17078.cf", dtype=np.float32, mode="r", shape=(15283,17078,))