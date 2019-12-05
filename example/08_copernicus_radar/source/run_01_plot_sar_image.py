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
xmlfn="/home/martin/Downloads/s1a-iw1-slc-vh-20181106t135248-20181106t135313-024468-02aeb9-001.xml"
xm=et.parse(xmlfn)
df=pd.read_csv("./o_range.csv")
s=np.memmap(next(pathlib.Path("./").glob("o*.cf")), dtype=np.complex64, mode="r", shape=(5000,15283,))
ys0=np.zeros((5000,15283,), dtype=np.complex64)
for idx in range(ys0.shape[0]):
    fref=(3.7534721374511715e+1)
    row=df.iloc[idx]
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
k0=np.fft.fft(s, axis=1)
kp=np.fft.fft(ys0, axis=1)
plt.imshow(np.log((((1.0000000474974513e-3))+(np.real(np.fft.ifft(((k0)*(kp))))))))