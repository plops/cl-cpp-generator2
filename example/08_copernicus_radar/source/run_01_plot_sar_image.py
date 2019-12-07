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
s=np.memmap(next(pathlib.Path("./").glob("o*.cf")), dtype=np.complex64, mode="r", shape=(24766,24601,))
a2=scipy.signal.decimate(np.abs(s[8000:,:]), 10)
a3=scipy.signal.decimate(np.abs(a2), 10, axis=0)
del(a2)
fig=plt.figure()
ax=fig.add_subplot("111")
ax.imshow(a3)
ax.set_aspect("auto")