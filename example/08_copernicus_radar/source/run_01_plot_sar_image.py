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
import scipy.signal
import xml.etree.ElementTree as et
df=pd.read_csv("./o_range.csv")
s=np.memmap(next(pathlib.Path("./").glob("o_cal*.cf")), dtype=np.complex64, mode="r", shape=(800,3000,))
fig=plt.figure()
ax=fig.add_subplot("111")
ax.imshow(np.real(s))
ax.set_aspect("auto")