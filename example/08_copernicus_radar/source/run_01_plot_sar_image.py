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
df=pd.read_csv("./o_range.csv")
fref=(3.7534721374511715e+1)
row=df.iloc[0]
txprr=row.txprr
txprr_=row.txprr_
txpsf=row.txpsf
txpl=row.txpl
txpl_=row.txpl_
ns=np.arange(txpl_)
xs=((ns)/(fref))
arg=((((txpsf)*(xs)))+((((5.e-1))*(txprr)*(xs)*(xs))))
ys=np.exp(((2j)*(np.pi)*(arg)))
plt.plot(xs, ys)
s=np.memmap(next(pathlib.Path("./").glob("o*.cf")), dtype=np.complex64, mode="r", shape=(22778,15283,))