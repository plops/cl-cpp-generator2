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
# %% echo packet information
df=pd.read_csv("./o_range.csv")
cal_type_desc=["tx_cal", "rx_cal", "epdn_cal", "ta_cal", "apdn_cal", "na_0", "na_1", "txh_iso_cal"]
pol_desc=["txh", "txh_rxh", "txh_rxv", "txh_rxvh", "txv", "txv_rxh", "txv_rxv", "txv_rxvh"]
rx_desc=["rxv", "rxh"]
signal_type_desc=["echo", "noise", "na2", "na3", "na4", "na5", "na6", "na7", "tx_cal", "rx_cal", "epdn_cal", "ta_cal", "apdn_cal", "na13", "na14", "txhiso_cal"]
# %% calibration packet information
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
# %% sample rate computation using 3.2.5.4, tab5.1-1, tab5.1-2, txpl and swl description
decimation_filter_bandwidth=[100, (8.770999908447266e+1), -1, (7.424999999999999e+1), (5.943999862670898e+1), (5.061999893188476e+1), (4.4889999389648433e+1), (2.220000076293945e+1), (5.659000015258788e+1), (4.286000061035156e+1), (1.5100000381469728e+1), (4.834999847412109e+1)]
decimation_filter_L=[3, 2, -1, 5, 4, 3, 1, 1, 3, 5, 3, 4]
decimation_filter_M=[4, 3, -1, 9, 9, 8, 3, 6, 7, 16, 26, 11]
decimation_filter_length_NF=[28, 28, -1, 32, 40, 48, 52, 92, 36, 68, 120, 44]
decimation_filter_output_offset=[87, 87, -1, 88, 90, 92, 93, 103, 89, 97, 110, 91]
decimation_filter_swath_desc=["full_bandwidth", "s1_wv1", "n/a", "s2", "s3", "s4", "s5", "ew1", "iw1", "s6_iw3", "ew2_ew3_ew4_ew5", "iw2_wv2"]
# table 5.1-1 D values as a function of rgdec and C. first index is rgdec
decimation_filter_D=[[1, 1, 2, 3], [1, 1, 2], [-1], [1, 1, 2, 2, 3, 3, 4, 4, 5], [0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 1, 1, 2, 2, 3, 3], [0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 1, 1, 2, 2, 3, 3], [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3], [0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4]]
dfc["decimation_filter_bandwidth"]=list(map(lambda x: decimation_filter_bandwidth[x], dfc.rgdec))
dfc["decimation_filter_L"]=list(map(lambda x: decimation_filter_L[x], dfc.rgdec))
dfc["decimation_filter_M"]=list(map(lambda x: decimation_filter_M[x], dfc.rgdec))
dfc["decimation_filter_length_NF"]=list(map(lambda x: decimation_filter_length_NF[x], dfc.rgdec))
dfc["decimation_filter_output_offset"]=list(map(lambda x: decimation_filter_output_offset[x], dfc.rgdec))
dfc["decimation_filter_swath_desc"]=list(map(lambda x: decimation_filter_swath_desc[x], dfc.rgdec))
df["decimation_filter_bandwidth"]=list(map(lambda x: decimation_filter_bandwidth[x], df.rgdec))
df["decimation_filter_L"]=list(map(lambda x: decimation_filter_L[x], df.rgdec))
df["decimation_filter_M"]=list(map(lambda x: decimation_filter_M[x], df.rgdec))
df["decimation_filter_length_NF"]=list(map(lambda x: decimation_filter_length_NF[x], df.rgdec))
df["decimation_filter_output_offset"]=list(map(lambda x: decimation_filter_output_offset[x], df.rgdec))
df["decimation_filter_swath_desc"]=list(map(lambda x: decimation_filter_swath_desc[x], df.rgdec))
fref=(3.7534721374511715e+1)
dfc["fdec"]=((4)*(fref)*(((dfc.decimation_filter_L)/(dfc.decimation_filter_M))))
dfc["N3_tx"]=np.ceil(((dfc.fdec)*(dfc.txpl))).astype(np.int)
dfc["decimation_filter_B"]=((((2)*(dfc.swl)))-(((dfc.decimation_filter_output_offset)+(17))))
dfc["decimation_filter_C"]=((dfc.decimation_filter_B)-(((dfc.decimation_filter_M)*(((dfc.decimation_filter_B)//(dfc.decimation_filter_M))))))
dfc["N3_rx"]=list(map(lambda idx_row: ((2)*(((((idx_row[1].decimation_filter_L)*(((idx_row[1].decimation_filter_B)//(idx_row[1].decimation_filter_M)))))+(decimation_filter_D[idx_row[1].rgdec][idx_row[1].decimation_filter_C])+(1)))), dfc.iterrows()))
df["fdec"]=((4)*(fref)*(((df.decimation_filter_L)/(df.decimation_filter_M))))
df["N3_tx"]=np.ceil(((df.fdec)*(df.txpl))).astype(np.int)
df["decimation_filter_B"]=((((2)*(df.swl)))-(((df.decimation_filter_output_offset)+(17))))
df["decimation_filter_C"]=((df.decimation_filter_B)-(((df.decimation_filter_M)*(((df.decimation_filter_B)//(df.decimation_filter_M))))))
df["N3_rx"]=list(map(lambda idx_row: ((2)*(((((idx_row[1].decimation_filter_L)*(((idx_row[1].decimation_filter_B)//(idx_row[1].decimation_filter_M)))))+(decimation_filter_D[idx_row[1].rgdec][idx_row[1].decimation_filter_C])+(1)))), df.iterrows()))
s=np.memmap(next(pathlib.Path("./").glob("o_cal*.cf")), dtype=np.complex64, mode="r", shape=(700,6000,))
ss=np.memmap(next(pathlib.Path("./").glob("o_r*.cf")), dtype=np.complex64, mode="r", shape=(5533,13177,))
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