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
plt.figure()
plt.gca().set_prop_cycle(None)
savgol_kernel_size=((1)+(((2)*(kernel_size))))
q=tx_cal_0[count,:]
sav_mag_q=scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)
sav_arg_q=scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)
sav_tx_cal_0=((sav_mag_q)*(np.exp(((1j)*(sav_arg_q)))))
scale=(1.e+0)
v=((scale)*(scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)))
plt.plot(np.linspace(0, ((((len(v))-(1)))/(1)), len(v)), v, linestyle="-", label="|tx_cal_0|")
savgol_kernel_size=((1)+(((2)*(kernel_size))))
q=rx_cal_0[count,:]
sav_mag_q=scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)
sav_arg_q=scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)
sav_rx_cal_0=((sav_mag_q)*(np.exp(((1j)*(sav_arg_q)))))
scale=(1.e+0)
v=((scale)*(scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)))
plt.plot(np.linspace(0, ((((len(v))-(1)))/(1)), len(v)), v, linestyle="-", label="|rx_cal_0|")
savgol_kernel_size=((1)+(((2)*(kernel_size))))
q=epdn_cal_0[count,:]
sav_mag_q=scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)
sav_arg_q=scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)
sav_epdn_cal_0=((sav_mag_q)*(np.exp(((1j)*(sav_arg_q)))))
scale=(1.e+0)
v=((scale)*(scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)))
plt.plot(np.linspace(0, ((((len(v))-(1)))/(1)), len(v)), v, linestyle="-", label="|epdn_cal_0|")
savgol_kernel_size=((1)+(((2)*(kernel_size))))
q=ta_cal_0[count,:]
sav_mag_q=scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)
sav_arg_q=scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)
sav_ta_cal_0=((sav_mag_q)*(np.exp(((1j)*(sav_arg_q)))))
scale=(1.e+0)
v=((scale)*(scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)))
plt.plot(np.linspace(0, ((((len(v))-(1)))/(1)), len(v)), v, linestyle="-", label="|ta_cal_0|")
savgol_kernel_size=((1)+(((2)*(kernel_size))))
q=np.mean(apdn_cal_0, axis=0)
sav_mag_q=scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)
sav_arg_q=scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)
sav_apdn_cal_0=((sav_mag_q)*(np.exp(((1j)*(sav_arg_q)))))
scale=(1.e+0)
v=((scale)*(scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)))
plt.plot(np.linspace(0, ((((len(v))-(1)))/(1)), len(v)), v, linestyle="-", label="|apdn_cal_0|")
savgol_kernel_size=((1)+(((2)*(kernel_size))))
q=txh_iso_cal_0[count,:]
sav_mag_q=scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)
sav_arg_q=scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)
sav_txh_iso_cal_0=((sav_mag_q)*(np.exp(((1j)*(sav_arg_q)))))
scale=(1.e+0)
v=((scale)*(scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)))
plt.plot(np.linspace(0, ((((len(v))-(1)))/(1)), len(v)), v, linestyle="-", label="|txh_iso_cal_0|")
plt.gca().set_prop_cycle(None)
savgol_kernel_size=((1)+(((4)*(kernel_size))))
q=tx_cal_1[count,:]
sav_mag_q=scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)
sav_arg_q=scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)
sav_tx_cal_1=((sav_mag_q)*(np.exp(((1j)*(sav_arg_q)))))
scale=(1.e+0)
v=((scale)*(scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)))
plt.plot(np.linspace(0, ((((len(v))-(1)))/(2)), len(v)), v, linestyle="--", label="|tx_cal_1|")
savgol_kernel_size=((1)+(((4)*(kernel_size))))
q=rx_cal_1[count,:]
sav_mag_q=scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)
sav_arg_q=scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)
sav_rx_cal_1=((sav_mag_q)*(np.exp(((1j)*(sav_arg_q)))))
scale=(1.e+0)
v=((scale)*(scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)))
plt.plot(np.linspace(0, ((((len(v))-(1)))/(2)), len(v)), v, linestyle="--", label="|rx_cal_1|")
savgol_kernel_size=((1)+(((4)*(kernel_size))))
q=epdn_cal_1[count,:]
sav_mag_q=scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)
sav_arg_q=scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)
sav_epdn_cal_1=((sav_mag_q)*(np.exp(((1j)*(sav_arg_q)))))
scale=(1.e+0)
v=((scale)*(scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)))
plt.plot(np.linspace(0, ((((len(v))-(1)))/(2)), len(v)), v, linestyle="--", label="|epdn_cal_1|")
savgol_kernel_size=((1)+(((4)*(kernel_size))))
q=ta_cal_1[count,:]
sav_mag_q=scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)
sav_arg_q=scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)
sav_ta_cal_1=((sav_mag_q)*(np.exp(((1j)*(sav_arg_q)))))
scale=(1.e+0)
v=((scale)*(scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)))
plt.plot(np.linspace(0, ((((len(v))-(1)))/(2)), len(v)), v, linestyle="--", label="|ta_cal_1|")
savgol_kernel_size=((1)+(((4)*(kernel_size))))
q=np.mean(apdn_cal_1, axis=0)
sav_mag_q=scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)
sav_arg_q=scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)
sav_apdn_cal_1=((sav_mag_q)*(np.exp(((1j)*(sav_arg_q)))))
scale=(1.e+0)
v=((scale)*(scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)))
plt.plot(np.linspace(0, ((((len(v))-(1)))/(2)), len(v)), v, linestyle="--", label="|apdn_cal_1|")
savgol_kernel_size=((1)+(((4)*(kernel_size))))
q=txh_iso_cal_1[count,:]
sav_mag_q=scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)
sav_arg_q=scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)
sav_txh_iso_cal_1=((sav_mag_q)*(np.exp(((1j)*(sav_arg_q)))))
scale=(1.e+0)
v=((scale)*(scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)))
plt.plot(np.linspace(0, ((((len(v))-(1)))/(2)), len(v)), v, linestyle="--", label="|txh_iso_cal_1|")
plt.grid()
plt.legend()
plt.figure()
plt.gca().set_prop_cycle(None)
savgol_kernel_size=((1)+(((2)*(kernel_size))))
q=tx_cal_0[count,:]
sav_mag_q=scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)
sav_arg_q=scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)
sav_tx_cal_0=((sav_mag_q)*(np.exp(((1j)*(sav_arg_q)))))
scale=(1.e+0)
v=((scale)*(scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)))
plt.plot(np.linspace(0, ((((len(v))-(1)))/(1)), len(v)), v, linestyle="-", label="arg tx_cal_0")
savgol_kernel_size=((1)+(((2)*(kernel_size))))
q=rx_cal_0[count,:]
sav_mag_q=scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)
sav_arg_q=scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)
sav_rx_cal_0=((sav_mag_q)*(np.exp(((1j)*(sav_arg_q)))))
scale=(1.e+0)
v=((scale)*(scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)))
plt.plot(np.linspace(0, ((((len(v))-(1)))/(1)), len(v)), v, linestyle="-", label="arg rx_cal_0")
savgol_kernel_size=((1)+(((2)*(kernel_size))))
q=epdn_cal_0[count,:]
sav_mag_q=scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)
sav_arg_q=scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)
sav_epdn_cal_0=((sav_mag_q)*(np.exp(((1j)*(sav_arg_q)))))
scale=(1.e+0)
v=((scale)*(scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)))
plt.plot(np.linspace(0, ((((len(v))-(1)))/(1)), len(v)), v, linestyle="-", label="arg epdn_cal_0")
savgol_kernel_size=((1)+(((2)*(kernel_size))))
q=ta_cal_0[count,:]
sav_mag_q=scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)
sav_arg_q=scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)
sav_ta_cal_0=((sav_mag_q)*(np.exp(((1j)*(sav_arg_q)))))
scale=(1.e+0)
v=((scale)*(scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)))
plt.plot(np.linspace(0, ((((len(v))-(1)))/(1)), len(v)), v, linestyle="-", label="arg ta_cal_0")
savgol_kernel_size=((1)+(((2)*(kernel_size))))
q=np.mean(apdn_cal_0, axis=0)
sav_mag_q=scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)
sav_arg_q=scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)
sav_apdn_cal_0=((sav_mag_q)*(np.exp(((1j)*(sav_arg_q)))))
scale=(1.e+0)
v=((scale)*(scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)))
plt.plot(np.linspace(0, ((((len(v))-(1)))/(1)), len(v)), v, linestyle="-", label="arg apdn_cal_0")
savgol_kernel_size=((1)+(((2)*(kernel_size))))
q=txh_iso_cal_0[count,:]
sav_mag_q=scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)
sav_arg_q=scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)
sav_txh_iso_cal_0=((sav_mag_q)*(np.exp(((1j)*(sav_arg_q)))))
scale=(1.e+0)
v=((scale)*(scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)))
plt.plot(np.linspace(0, ((((len(v))-(1)))/(1)), len(v)), v, linestyle="-", label="arg txh_iso_cal_0")
plt.gca().set_prop_cycle(None)
savgol_kernel_size=((1)+(((4)*(kernel_size))))
q=tx_cal_1[count,:]
sav_mag_q=scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)
sav_arg_q=scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)
sav_tx_cal_1=((sav_mag_q)*(np.exp(((1j)*(sav_arg_q)))))
scale=(5.e-1)
v=((scale)*(scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)))
plt.plot(np.linspace(0, ((((len(v))-(1)))/(2)), len(v)), v, linestyle="--", label="arg tx_cal_1")
savgol_kernel_size=((1)+(((4)*(kernel_size))))
q=rx_cal_1[count,:]
sav_mag_q=scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)
sav_arg_q=scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)
sav_rx_cal_1=((sav_mag_q)*(np.exp(((1j)*(sav_arg_q)))))
scale=(5.e-1)
v=((scale)*(scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)))
plt.plot(np.linspace(0, ((((len(v))-(1)))/(2)), len(v)), v, linestyle="--", label="arg rx_cal_1")
savgol_kernel_size=((1)+(((4)*(kernel_size))))
q=epdn_cal_1[count,:]
sav_mag_q=scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)
sav_arg_q=scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)
sav_epdn_cal_1=((sav_mag_q)*(np.exp(((1j)*(sav_arg_q)))))
scale=(5.e-1)
v=((scale)*(scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)))
plt.plot(np.linspace(0, ((((len(v))-(1)))/(2)), len(v)), v, linestyle="--", label="arg epdn_cal_1")
savgol_kernel_size=((1)+(((4)*(kernel_size))))
q=ta_cal_1[count,:]
sav_mag_q=scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)
sav_arg_q=scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)
sav_ta_cal_1=((sav_mag_q)*(np.exp(((1j)*(sav_arg_q)))))
scale=(5.e-1)
v=((scale)*(scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)))
plt.plot(np.linspace(0, ((((len(v))-(1)))/(2)), len(v)), v, linestyle="--", label="arg ta_cal_1")
savgol_kernel_size=((1)+(((4)*(kernel_size))))
q=np.mean(apdn_cal_1, axis=0)
sav_mag_q=scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)
sav_arg_q=scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)
sav_apdn_cal_1=((sav_mag_q)*(np.exp(((1j)*(sav_arg_q)))))
scale=(5.e-1)
v=((scale)*(scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)))
plt.plot(np.linspace(0, ((((len(v))-(1)))/(2)), len(v)), v, linestyle="--", label="arg apdn_cal_1")
savgol_kernel_size=((1)+(((4)*(kernel_size))))
q=txh_iso_cal_1[count,:]
sav_mag_q=scipy.signal.savgol_filter(np.abs(q), savgol_kernel_size, 2)
sav_arg_q=scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)
sav_txh_iso_cal_1=((sav_mag_q)*(np.exp(((1j)*(sav_arg_q)))))
scale=(5.e-1)
v=((scale)*(scipy.signal.savgol_filter(np.unwrap(np.angle(q)), savgol_kernel_size, 2)))
plt.plot(np.linspace(0, ((((len(v))-(1)))/(2)), len(v)), v, linestyle="--", label="arg txh_iso_cal_1")
plt.grid()
plt.legend()
plt.figure()
plt.suptitle("single pulse")
pl=(5,1,)
ax0=plt.subplot2grid(pl, (0,0,))
ax1=plt.subplot2grid(pl, (1,0,))
ax2=plt.subplot2grid(pl, (2,0,))
ax3=plt.subplot2grid(pl, (3,0,))
ax4=plt.subplot2grid(pl, (4,0,))
reps=np.zeros(tx_cal_0.shape, dtype=np.complex64)
for count in range(reps.shape[0]):
    top=((np.fft.fft(tx_cal_0[count,:]))*(np.fft.fft(rx_cal_0[count,:]))*(np.fft.fft(ta_cal_0[count,:])))
    bot=((np.fft.fft(np.mean(apdn_cal_0, axis=0)))*(np.fft.fft(epdn_cal_0[count,:])))
    win=np.fft.fftshift(scipy.signal.tukey(tx_cal_0.shape[1], alpha=(1.0000000149011612e-1)))
    reps[count,:]=np.fft.ifft(((win)*(((top)/(bot)))))
for count in range(3):
    # i want to compute rep_vv according to page 36 (detailed alg definition)
    top=((np.fft.fft(tx_cal_0[count,:]))*(np.fft.fft(rx_cal_0[count,:]))*(np.fft.fft(ta_cal_0[count,:])))
    bot=((np.fft.fft(np.mean(apdn_cal_0, axis=0)))*(np.fft.fft(epdn_cal_0[count,:])))
    win=np.fft.fftshift(scipy.signal.tukey(tx_cal_0.shape[1], alpha=(1.0000000149011612e-1)))
    reps[count,:]=np.fft.ifft(((win)*(((top)/(bot)))))
    xs=np.fft.fftfreq(len(top))
    ax0.plot(xs, np.abs(top), label="top")
    ax0.grid()
    ax0.legend()
    ax1.plot(xs, np.abs(bot), label="bot")
    ax1.legend()
    ax1.grid()
    ax2.plot(xs, win, label="tukey")
    ax2.legend()
    ax2.grid()
    ax3.plot(xs, np.abs(((top)/(bot))), label="top/bot")
    ax3.plot(xs, ((win)*(np.abs(((top)/(bot))))), label="top/bot*win")
    ax3.legend()
    ax3.grid()
    ax4.plot(np.real(np.fft.ifft(((top)/(bot)))), label="ifft top/bot")
    ax4.plot(np.real(np.fft.ifft(((win)*(((top)/(bot)))))), label="ifft top/bot*win")
    ax4.legend()
    ax4.grid()
# page 38, (4-36) compress replicas using the first extracted replica
repsc=np.zeros((((reps.shape[0])-(1)),reps.shape[1],), dtype=np.complex64)
for i in range(1, reps.shape[0]):
    repsc[((i)-(1)),:]=np.fft.ifft(((np.fft.fft(reps[i]))*(np.conj(np.fft.fft(reps[0])))))