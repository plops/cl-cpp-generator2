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
import xml.etree.ElementTree as et
df=pd.read_csv("./o_range.csv")
s=np.memmap(next(pathlib.Path("./").glob("o*.cf")), dtype=np.complex64, mode="r", shape=(300,30199,))
skip=0
ys0=np.empty_like(s)
fig=plt.figure()
ax=fig.add_subplot("111")
ax.imshow(np.log((((9.999999776482582e-3))+(np.abs(s)))))
ax.set_aspect("auto")