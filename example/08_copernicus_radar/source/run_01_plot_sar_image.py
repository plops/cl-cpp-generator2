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
s=np.memmap(next(pathlib.Path("./").glob("o_cal*.cf")), dtype=np.complex64, mode="r", shape=(800,6000,))
plt.plot(np.real(np.mean(s[0:50,:], axis=0)), label="real")
plt.plot(np.imag(np.mean(s[0:50,:], axis=0)), label="imag")
fref=(3.7534721374511715e+1)
row=dfc.iloc[0]
txprr=row.txprr
txprr_=row.txprr_
txpsf=row.txpsf
txpl=row.txpl
txpl_=row.txpl_
tn=np.linspace((((-5.e-1))*(txpl)), (((5.e-1))*(txpl)), ((2)*(row.number_of_quads)))
p1=((txpsf)-(((txprr)*((-5.e-1))*(txpl))))
p2=(((5.e-1))*(txprr))
arg=((((p1)*(tn)))+(((p2)*(tn)*(tn))))
ys=((175)*(np.exp(((-2j)*(np.pi)*(arg)))))
plt.plot(np.abs(np.real(ys)), label="analytic")
plt.legend()
fig=plt.figure()
ax=fig.add_subplot("111")
ax.imshow(np.angle(s))
ax.set_aspect("auto")