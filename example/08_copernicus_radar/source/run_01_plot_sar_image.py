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
s=np.memmap(next(pathlib.Path("./").glob("o*.cf")), dtype=np.complex64, mode="r", shape=(16516,24223,))
skip=8000
spart=s[skip::,:]
del(s)
ys0=np.zeros((1,24223,), dtype=np.complex64)
for idx in [0]:
    fref=(3.7534721374511715e+1)
    row=df.iloc[((skip)+(idx))]
    txprr=row.txprr
    txprr_=row.txprr_
    txpsf=row.txpsf
    txpl=row.txpl
    txpl_=row.txpl_
    ns=np.arange(txpl_)
    xs=((ns)/(fref))
    arg=((((txpsf)*(xs)))+((((5.e-1))*(txprr)*(xs)*(xs))))
    ys=np.exp(((-2j)*(np.pi)*(arg)))
    ys0[idx,0:len(ns)]=ys
k0=np.fft.fft(spart, axis=1)
kp=np.fft.fft(ys0, axis=1)
a=((k0)*(kp))
del(kp)
del(k0)
img=np.fft.ifft(a)
del(a)
plt.imshow(np.log((((1.0000000474974513e-3))+(np.abs(img)))))