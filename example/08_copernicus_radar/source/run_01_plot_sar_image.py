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
import xml.etree.ElementTree as et
df=pd.read_csv("./o_range.csv")
cal_type_desc=["tx_cal", "rx_cal", "epdn_cal", "ta_cal", "apdn_cal", "na_0", "na_1", "txh_iso_cal"]
pol_desc=["txh", "txh_rxh", "txh_rxv", "txh_rxvh", "txv", "txv_rxh", "txv_rxv", "txv_rxvh"]
rx_desc=["rxv", "rxh"]
dfc=pd.read_csv("./o_cal_range.csv")
dfc["pcc"]=np.mod(dfc.cal_iter, 2)
dfc["cal_type_desc"]=list(map(lambda x: cal_type_desc[x], dfc.cal_type))
dfc.cal_type_desc=dfc.cal_type_desc.astype("category")
dfc["pol_desc"]=list(map(lambda x: pol_desc[x], dfc.pol))
dfc.pol_desc=dfc.pol_desc.astype("category")
dfc["rx_desc"]=list(map(lambda x: rx_desc[x], dfc.rx))
dfc.rx_desc=dfc.rx_desc.astype("category")
s=np.memmap(next(pathlib.Path("./").glob("o_cal*.cf")), dtype=np.complex64, mode="r", shape=(800,6000,))
ss=np.memmap(next(pathlib.Path("./").glob("o_r*.cf")), dtype=np.complex64, mode="r", shape=(16516,24695,))
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
sub=dfc[((((dfc.cal_type_desc)==("tx_cal"))) & (((dfc.number_of_quads)==(un[1]))) & (((dfc.pcc)==(0))))]
tx_cal_1=np.zeros((len(sub),6000,), dtype=np.complex64)
j=0
for i in sub.cal_iter:
    tx_cal_1[j,:]=(((5.e-1))*(((s[i,:])-(s[((i)+(1)),:]))))
    j=((j)+(1))
sub=dfc[((((dfc.cal_type_desc)==("rx_cal"))) & (((dfc.number_of_quads)==(un[0]))) & (((dfc.pcc)==(0))))]
rx_cal_0=np.zeros((len(sub),6000,), dtype=np.complex64)
j=0
for i in sub.cal_iter:
    rx_cal_0[j,:]=(((5.e-1))*(((s[i,:])-(s[((i)+(1)),:]))))
    j=((j)+(1))
sub=dfc[((((dfc.cal_type_desc)==("rx_cal"))) & (((dfc.number_of_quads)==(un[1]))) & (((dfc.pcc)==(0))))]
rx_cal_1=np.zeros((len(sub),6000,), dtype=np.complex64)
j=0
for i in sub.cal_iter:
    rx_cal_1[j,:]=(((5.e-1))*(((s[i,:])-(s[((i)+(1)),:]))))
    j=((j)+(1))
sub=dfc[((((dfc.cal_type_desc)==("epdn_cal"))) & (((dfc.number_of_quads)==(un[0]))) & (((dfc.pcc)==(0))))]
epdn_cal_0=np.zeros((len(sub),6000,), dtype=np.complex64)
j=0
for i in sub.cal_iter:
    epdn_cal_0[j,:]=(((5.e-1))*(((s[i,:])-(s[((i)+(1)),:]))))
    j=((j)+(1))
sub=dfc[((((dfc.cal_type_desc)==("epdn_cal"))) & (((dfc.number_of_quads)==(un[1]))) & (((dfc.pcc)==(0))))]
epdn_cal_1=np.zeros((len(sub),6000,), dtype=np.complex64)
j=0
for i in sub.cal_iter:
    epdn_cal_1[j,:]=(((5.e-1))*(((s[i,:])-(s[((i)+(1)),:]))))
    j=((j)+(1))
sub=dfc[((((dfc.cal_type_desc)==("ta_cal"))) & (((dfc.number_of_quads)==(un[0]))) & (((dfc.pcc)==(0))))]
ta_cal_0=np.zeros((len(sub),6000,), dtype=np.complex64)
j=0
for i in sub.cal_iter:
    ta_cal_0[j,:]=(((5.e-1))*(((s[i,:])-(s[((i)+(1)),:]))))
    j=((j)+(1))
sub=dfc[((((dfc.cal_type_desc)==("ta_cal"))) & (((dfc.number_of_quads)==(un[1]))) & (((dfc.pcc)==(0))))]
ta_cal_1=np.zeros((len(sub),6000,), dtype=np.complex64)
j=0
for i in sub.cal_iter:
    ta_cal_1[j,:]=(((5.e-1))*(((s[i,:])-(s[((i)+(1)),:]))))
    j=((j)+(1))
sub=dfc[((((dfc.cal_type_desc)==("apdn_cal"))) & (((dfc.number_of_quads)==(un[0]))) & (True))]
apdn_cal_0=np.zeros((len(sub),6000,), dtype=np.complex64)
j=0
for i in sub.cal_iter:
    apdn_cal_0[j,:]=s[i,:]
    j=((j)+(1))
sub=dfc[((((dfc.cal_type_desc)==("apdn_cal"))) & (((dfc.number_of_quads)==(un[1]))) & (True))]
apdn_cal_1=np.zeros((len(sub),6000,), dtype=np.complex64)
j=0
for i in sub.cal_iter:
    apdn_cal_1[j,:]=s[i,:]
    j=((j)+(1))
sub=dfc[((((dfc.cal_type_desc)==("txh_iso_cal"))) & (((dfc.number_of_quads)==(un[0]))) & (True))]
txh_iso_cal_0=np.zeros((len(sub),6000,), dtype=np.complex64)
j=0
for i in sub.cal_iter:
    txh_iso_cal_0[j,:]=s[i,:]
    j=((j)+(1))
sub=dfc[((((dfc.cal_type_desc)==("txh_iso_cal"))) & (((dfc.number_of_quads)==(un[1]))) & (True))]
txh_iso_cal_1=np.zeros((len(sub),6000,), dtype=np.complex64)
j=0
for i in sub.cal_iter:
    txh_iso_cal_1[j,:]=s[i,:]
    j=((j)+(1))
reps=np.zeros(tx_cal_0.shape, dtype=np.complex64)
for count in range(reps.shape[0]):
    top=((np.fft.fft(tx_cal_0[count,:]))*(np.fft.fft(rx_cal_0[count,:]))*(np.fft.fft(ta_cal_0[count,:])))
    bot=((np.fft.fft(np.mean(apdn_cal_0, axis=0)))*(np.fft.fft(epdn_cal_0[count,:])))
    win=np.fft.fftshift(scipy.signal.tukey(tx_cal_0.shape[1], alpha=(1.0000000149011612e-1)))
    reps[count,:]=np.fft.ifft(((win)*(((top)/(bot)))))
# %% fit polynomial to magnitude and phase
a=np.abs(reps[0])
th=(((5.e-1))*(np.max(a)))
mask=((th)<(a))
start=np.argmax(mask)
end=((len(mask))-(np.argmax(mask[::-1])))
plt.plot(a)
plt.axvline(x=start)
plt.axvline(x=end)