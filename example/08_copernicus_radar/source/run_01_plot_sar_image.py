import matplotlib
import matplotlib.pyplot as plt
plt.ion()
# %% imports
import sys
import time
import pathlib
import numpy as np
import numpy.fft
import pandas as pd
import scipy.signal
import numpy.polynomial
import xml.etree.ElementTree as et
df=pd.read_csv("./o_range.csv")
cal_type_desc=["tx_cal", "rx_cal", "epdn_cal", "ta_cal", "apdn_cal", "na_0", "na_1", "txh_iso_cal"]
pol_desc=["txh", "txh_rxh", "txh_rxv", "txh_rxvh", "txv", "txv_rxh", "txv_rxv", "txv_rxvh"]
rx_desc=["rxv", "rxh"]
signal_type_desc=["echo", "noise", "na2", "na3", "na4", "na5", "na6", "na7", "tx_cal", "rx_cal", "epdn_cal", "ta_cal", "apdn_cal", "na13", "na14", "txhiso_cal"]
dfc=pd.read_csv("./o_cal_range.csv")
dfc["pcc"]=np.mod(dfc.cal_iter, 2)
dfc["pol_desc"]=list(map(lambda x: pol_desc[x], dfc.pol))
dfc.pol_desc=dfc.pol_desc.astype("category")
dfc["rx_desc"]=list(map(lambda x: rx_desc[x], dfc.rx))
dfc.rx_desc=dfc.rx_desc.astype("category")
dfc["signal_type_desc"]=list(map(lambda x: signal_type_desc[x], dfc.signal_type))
dfc.signal_type_desc=dfc.signal_type_desc.astype("category")
dfc["cal_type_desc"]=list(map(lambda x: cal_type_desc[x], dfc.cal_type))
dfc.cal_type_desc=dfc.cal_type_desc.astype("category")
df["pol_desc"]=list(map(lambda x: pol_desc[x], df.pol))
df.pol_desc=df.pol_desc.astype("category")
df["rx_desc"]=list(map(lambda x: rx_desc[x], df.rx))
df.rx_desc=df.rx_desc.astype("category")
df["signal_type_desc"]=list(map(lambda x: signal_type_desc[x], df.signal_type))
df.signal_type_desc=df.signal_type_desc.astype("category")
s=np.memmap(next(pathlib.Path("./").glob("o_cal*.cf")), dtype=np.complex64, mode="r", shape=(700,6000,))
ss=np.memmap(next(pathlib.Path("./").glob("o_r*.cf")), dtype=np.complex64, mode="r", shape=(10000,29884,))
u=dfc.cal_type_desc.unique()
un=dfc.number_of_quads.unique()
count=0
kernel_size=8
sub=dfc[((((dfc.cal_type_desc)==("tx_cal"))) & (((dfc.number_of_quads)==(un[0]))) & (((dfc.pcc)==(0))))]
tx_cal_0=np.zeros((len(sub),6000,), dtype=np.complex64)
j=0
for i in sub.cal_iter:
    tx_cal_0[j,:]=(((5.e-1))*(((s[i,:])-(s[((i)+(1)),:]))))
    j=((j)+(1))
sub=dfc[((((dfc.cal_type_desc)==("rx_cal"))) & (((dfc.number_of_quads)==(un[0]))) & (((dfc.pcc)==(0))))]
rx_cal_0=np.zeros((len(sub),6000,), dtype=np.complex64)
j=0
for i in sub.cal_iter:
    rx_cal_0[j,:]=(((5.e-1))*(((s[i,:])-(s[((i)+(1)),:]))))
    j=((j)+(1))
sub=dfc[((((dfc.cal_type_desc)==("epdn_cal"))) & (((dfc.number_of_quads)==(un[0]))) & (((dfc.pcc)==(0))))]
epdn_cal_0=np.zeros((len(sub),6000,), dtype=np.complex64)
j=0
for i in sub.cal_iter:
    epdn_cal_0[j,:]=(((5.e-1))*(((s[i,:])-(s[((i)+(1)),:]))))
    j=((j)+(1))
sub=dfc[((((dfc.cal_type_desc)==("ta_cal"))) & (((dfc.number_of_quads)==(un[0]))) & (((dfc.pcc)==(0))))]
ta_cal_0=np.zeros((len(sub),6000,), dtype=np.complex64)
j=0
for i in sub.cal_iter:
    ta_cal_0[j,:]=(((5.e-1))*(((s[i,:])-(s[((i)+(1)),:]))))
    j=((j)+(1))
sub=dfc[((((dfc.cal_type_desc)==("apdn_cal"))) & (((dfc.number_of_quads)==(un[0]))) & (True))]
apdn_cal_0=np.zeros((len(sub),6000,), dtype=np.complex64)
j=0
for i in sub.cal_iter:
    apdn_cal_0[j,:]=s[i,:]
    j=((j)+(1))
sub=dfc[((((dfc.cal_type_desc)==("txh_iso_cal"))) & (((dfc.number_of_quads)==(un[0]))) & (True))]
txh_iso_cal_0=np.zeros((len(sub),6000,), dtype=np.complex64)
j=0
for i in sub.cal_iter:
    txh_iso_cal_0[j,:]=s[i,:]
    j=((j)+(1))
reps=np.zeros(tx_cal_0.shape, dtype=np.complex64)
for count in range(reps.shape[0]):
    top=((np.fft.fft(tx_cal_0[count,:]))*(np.fft.fft(rx_cal_0[count,:]))*(np.fft.fft(ta_cal_0[count,:])))
    bot=((np.fft.fft(np.mean(apdn_cal_0, axis=0)))*(np.fft.fft(epdn_cal_0[count,:])))
    win=np.fft.fftshift(scipy.signal.tukey(tx_cal_0.shape[1], alpha=(1.0000000149011612e-1)))
    reps[count,:]=np.fft.ifft(((win)*(((top)/(bot)))))
# %% fit polynomial to magnitude 
a=np.abs(reps[0])
th_level=(8.99999976158142e-1)
th=((th_level)*(np.max(a)))
mask=((th)<(a))
start=np.argmax(mask)
end=((len(mask))-(np.argmax(mask[::-1])))
cut=a[start:end]
xs=np.arange(len(cut))
cba, cba_diag=np.polynomial.chebyshev.chebfit(xs, cut, 23, full=True)
plt.figure()
pl=(2,1,)
plt.subplot2grid(pl, (0,0,))
plt.plot(a)
plt.plot(((start)+(xs)), np.polynomial.chebyshev.chebval(xs, cba))
plt.axvline(x=start, color="r")
plt.axvline(x=end, color="r")
plt.xlim(((start)-(100)), ((end)+(100)))
plt.subplot2grid(pl, (1,0,))
plt.plot(((start)+(xs)), ((cut)-(np.polynomial.chebyshev.chebval(xs, cba))))
plt.xlim(((start)-(100)), ((end)+(100)))
plt.axvline(x=start, color="r")
plt.axvline(x=end, color="r")
# %% fit polynomial to phase
a=np.abs(reps[0])
arg=np.unwrap(np.angle(reps[0]))
th=((th_level)*(np.max(a)))
mask=((th)<(a))
start=np.argmax(mask)
end=((len(mask))-(np.argmax(mask[::-1])))
cut=arg[start:end]
xs=np.arange(len(cut))
cbarg, cbarg_diag=np.polynomial.chebyshev.chebfit(xs, cut, 22, full=True)
plt.figure()
pl=(2,1,)
plt.subplot2grid(pl, (0,0,))
plt.plot(arg)
plt.plot(((start)+(xs)), np.polynomial.chebyshev.chebval(xs, cbarg))
plt.axvline(x=start, color="r")
plt.axvline(x=end, color="r")
plt.xlim(((start)-(100)), ((end)+(100)))
plt.subplot2grid(pl, (1,0,))
plt.plot(((start)+(xs)), ((cut)-(np.polynomial.chebyshev.chebval(xs, cbarg))))
plt.xlim(((start)-(100)), ((end)+(100)))
plt.axvline(x=start, color="r")
plt.axvline(x=end, color="r")