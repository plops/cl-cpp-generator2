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
cal_type_dtype=pd.api.types.CategoricalDtype(ordered=True, categories=["tx_cal", "rx_cal", "epdn_cal", "ta_cal", "apdn_cal", "na_0", "na_1", "txh_cal_iso"])
dfc=pd.read_csv("./o_cal_range.csv")
s=np.memmap(next(pathlib.Path("./").glob("o_cal*.cf")), dtype=np.complex64, mode="r", shape=(800,3000,))
fig=plt.figure()
ax=fig.add_subplot("111")
ax.imshow(np.angle(s))
ax.set_aspect("auto")