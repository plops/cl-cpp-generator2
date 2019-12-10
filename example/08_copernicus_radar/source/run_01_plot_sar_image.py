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
plt.figure()
plt.plot(scipy.signal.savgol_filter(np.unwrap(np.angle(tx_cal_0[count,:])), ((1)+(((2)*(kernel_size)))), 2), label="angle <tx_cal_0>")
plt.plot(scipy.signal.savgol_filter(np.unwrap(np.angle(rx_cal_0[count,:])), ((1)+(((2)*(kernel_size)))), 2), label="angle <rx_cal_0>")
plt.plot(scipy.signal.savgol_filter(np.unwrap(np.angle(epdn_cal_0[count,:])), ((1)+(((2)*(kernel_size)))), 2), label="angle <epdn_cal_0>")
plt.plot(scipy.signal.savgol_filter(np.unwrap(np.angle(ta_cal_0[count,:])), ((1)+(((2)*(kernel_size)))), 2), label="angle <ta_cal_0>")
plt.plot(scipy.signal.savgol_filter(np.unwrap(np.angle(np.mean(apdn_cal_0, axis=0))), ((1)+(((2)*(kernel_size)))), 2), label="angle <apdn_cal_0>")
plt.plot(scipy.signal.savgol_filter(np.unwrap(np.angle(txh_iso_cal_0[count,:])), ((1)+(((2)*(kernel_size)))), 2), label="angle <txh_iso_cal_0>")
plt.grid()
plt.legend()
plt.figure()
plt.plot(scipy.signal.savgol_filter(np.unwrap(np.angle(tx_cal_1[count,:])), ((1)+(((4)*(kernel_size)))), 2), label="angle <tx_cal_1>")
plt.plot(scipy.signal.savgol_filter(np.unwrap(np.angle(rx_cal_1[count,:])), ((1)+(((4)*(kernel_size)))), 2), label="angle <rx_cal_1>")
plt.plot(scipy.signal.savgol_filter(np.unwrap(np.angle(epdn_cal_1[count,:])), ((1)+(((4)*(kernel_size)))), 2), label="angle <epdn_cal_1>")
plt.plot(scipy.signal.savgol_filter(np.unwrap(np.angle(ta_cal_1[count,:])), ((1)+(((4)*(kernel_size)))), 2), label="angle <ta_cal_1>")
plt.plot(scipy.signal.savgol_filter(np.unwrap(np.angle(np.mean(apdn_cal_1, axis=0))), ((1)+(((4)*(kernel_size)))), 2), label="angle <apdn_cal_1>")
plt.plot(scipy.signal.savgol_filter(np.unwrap(np.angle(txh_iso_cal_1[count,:])), ((1)+(((4)*(kernel_size)))), 2), label="angle <txh_iso_cal_1>")
plt.grid()
plt.legend()
plt.figure()
plt.plot(scipy.signal.savgol_filter(np.abs(tx_cal_0[count,:]), ((1)+(((2)*(kernel_size)))), 2), label="|<tx_cal_0>|")
plt.plot(scipy.signal.savgol_filter(np.abs(rx_cal_0[count,:]), ((1)+(((2)*(kernel_size)))), 2), label="|<rx_cal_0>|")
plt.plot(scipy.signal.savgol_filter(np.abs(epdn_cal_0[count,:]), ((1)+(((2)*(kernel_size)))), 2), label="|<epdn_cal_0>|")
plt.plot(scipy.signal.savgol_filter(np.abs(ta_cal_0[count,:]), ((1)+(((2)*(kernel_size)))), 2), label="|<ta_cal_0>|")
plt.plot(scipy.signal.savgol_filter(np.abs(np.mean(apdn_cal_0, axis=0)), ((1)+(((2)*(kernel_size)))), 2), label="|<apdn_cal_0>|")
plt.plot(scipy.signal.savgol_filter(np.abs(txh_iso_cal_0[count,:]), ((1)+(((2)*(kernel_size)))), 2), label="|<txh_iso_cal_0>|")
plt.grid()
plt.legend()
plt.figure()
plt.plot(scipy.signal.savgol_filter(np.abs(tx_cal_1[count,:]), ((1)+(((4)*(kernel_size)))), 2), label="|<tx_cal_1>|")
plt.plot(scipy.signal.savgol_filter(np.abs(rx_cal_1[count,:]), ((1)+(((4)*(kernel_size)))), 2), label="|<rx_cal_1>|")
plt.plot(scipy.signal.savgol_filter(np.abs(epdn_cal_1[count,:]), ((1)+(((4)*(kernel_size)))), 2), label="|<epdn_cal_1>|")
plt.plot(scipy.signal.savgol_filter(np.abs(ta_cal_1[count,:]), ((1)+(((4)*(kernel_size)))), 2), label="|<ta_cal_1>|")
plt.plot(scipy.signal.savgol_filter(np.abs(np.mean(apdn_cal_1, axis=0)), ((1)+(((4)*(kernel_size)))), 2), label="|<apdn_cal_1>|")
plt.plot(scipy.signal.savgol_filter(np.abs(txh_iso_cal_1[count,:]), ((1)+(((4)*(kernel_size)))), 2), label="|<txh_iso_cal_1>|")
plt.grid()
plt.legend()