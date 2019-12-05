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
import numpy.fft
import pandas as pd
s=np.memmap(next(pathlib.Path("./").glob("o*.cf")), dtype=np.complex64, mode="r", shape=(22778,15283,))
k=np.fft.fft(s.astype(np.complex128), axis=1)
plt.imshow(np.log((((1.0000000474974513e-3))+(np.abs(k)))))