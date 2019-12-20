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
import scipy.ndimage
import scipy.signal
import numpy.polynomial
# %% echo packet information
df=pd.read_csv("./o_range.csv")
dfa=pd.read_csv("./o_all.csv")
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
dfa["decimation_filter_bandwidth"]=list(map(lambda x: decimation_filter_bandwidth[x], dfa.rgdec))
dfa["decimation_filter_L"]=list(map(lambda x: decimation_filter_L[x], dfa.rgdec))
dfa["decimation_filter_M"]=list(map(lambda x: decimation_filter_M[x], dfa.rgdec))
dfa["decimation_filter_length_NF"]=list(map(lambda x: decimation_filter_length_NF[x], dfa.rgdec))
dfa["decimation_filter_output_offset"]=list(map(lambda x: decimation_filter_output_offset[x], dfa.rgdec))
dfa["decimation_filter_swath_desc"]=list(map(lambda x: decimation_filter_swath_desc[x], dfa.rgdec))
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
dfa["fdec"]=((4)*(fref)*(((dfa.decimation_filter_L)/(dfa.decimation_filter_M))))
dfa["N3_tx"]=np.ceil(((dfa.fdec)*(dfa.txpl))).astype(np.int)
dfa["decimation_filter_B"]=((((2)*(dfa.swl)))-(((dfa.decimation_filter_output_offset)+(17))))
dfa["decimation_filter_C"]=((dfa.decimation_filter_B)-(((dfa.decimation_filter_M)*(((dfa.decimation_filter_B)//(dfa.decimation_filter_M))))))
dfa["N3_rx"]=list(map(lambda idx_row: ((2)*(((((idx_row[1].decimation_filter_L)*(((idx_row[1].decimation_filter_B)//(idx_row[1].decimation_filter_M)))))+(decimation_filter_D[idx_row[1].rgdec][idx_row[1].decimation_filter_C])+(1)))), dfa.iterrows()))
# %% get pulse configuration that is rank pri_counts in the past
dfap=dfa.set_index("pri_count")
dfa["ranked_txprr"]=dfap.loc[((dfa.pri_count)-(dfa["rank"]))].reset_index().txprr
dfa["ranked_txprr_"]=dfap.loc[((dfa.pri_count)-(dfa["rank"]))].reset_index().txprr_
dfa["ranked_txpl"]=dfap.loc[((dfa.pri_count)-(dfa["rank"]))].reset_index().txpl
dfa["ranked_txpl_"]=dfap.loc[((dfa.pri_count)-(dfa["rank"]))].reset_index().txpl_
dfa["ranked_txpsf"]=dfap.loc[((dfa.pri_count)-(dfa["rank"]))].reset_index().txpsf
dfa["ranked_ses_ssb_tx_pulse_number"]=dfap.loc[((dfa.pri_count)-(dfa["rank"]))].reset_index().ses_ssb_tx_pulse_number
s=np.memmap(next(pathlib.Path("./").glob("o_cal*.cf")), dtype=np.complex64, mode="r", shape=(800,6000,))
ss=np.memmap(next(pathlib.Path("./").glob("o_r*.cf")), dtype=np.complex64, mode="r", shape=(1000,24890,))
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
th_level=(6.000000238418579e-1)
th=((th_level)*(np.max(a)))
mask=((th)<(a))
start=np.argmax(mask)
end=((len(mask))-(np.argmax(mask[::-1])))
cut=a[start:end]
fdec=dfc.iloc[0].fdec
start_us=((start)/(fdec))
end_us=((end)/(fdec))
xs_a_us=((np.arange(len(a)))/(fdec))
xs=xs_a_us[start:end]
cba, cba_diag=np.polynomial.chebyshev.chebfit(xs, cut, 2, full=True)
plt.figure()
pl=(2,1,)
plt.subplot2grid(pl, (0,0,))
plt.plot(xs_a_us, a)
plt.plot(xs_a_us, np.real(reps[0]), label="re reps0")
xs_off=((xs_a_us)-((((5.e-1))*(dfc.txpl[0])))-((5.e-1)))
xs_mask=(((((((-5.e-1))*(dfc.txpl[0])))<(xs_off))) & (((xs_off)<((((5.e-1))*(dfc.txpl[0]))))))
arg_nomchirp=((-2)*(np.pi)*(((((xs_off)*(((dfc.txpsf[0])+((((5.e-1))*(dfc.txpl[0])*(dfc.txprr[0])))))))+(((((xs_off)**(2)))*((5.e-1))*(dfc.txprr[0]))))))
def fun_nomchirp(xs, delta_t, ph, p0, p1):
    delta_t=(0.0e+0)
    amp=(((0.0e+0))+((7.5e+2)))
    xs_off=((xs)-(((delta_t)+((((((5.e-1))*(dfc.txpl[0])))+((5.e-1)))))))
    xs_mask=(((((((5.e-1))+((((-5.e-1))*(dfc.txpl[0])))))<(xs_off))) & (((xs_off)<((((-5.e-1))+((((5.e-1))*(dfc.txpl[0]))))))))
    arg_nomchirp=((((((np.pi)/((1.8e+2))))*(ph)))+(((-2)*(np.pi)*(((((xs_off)*(p0)*(((dfc.txpsf[0])+((((5.e-1))*(dfc.txpl[0])*(dfc.txprr[0])))))))+(((((xs_off)**(2)))*(p1)*((5.e-1))*(dfc.txprr[0]))))))))
    z=((amp)*(xs_mask)*(np.exp(((1j)*(arg_nomchirp)))))
    return np.concatenate((np.real(z),np.imag(z),))
def fun_nomchirp_polar(xs, delta_t, ph, p0, p1):
    delta_t=(0.0e+0)
    amp=(((0.0e+0))+((7.5e+2)))
    xs_off=((xs)-(((delta_t)+((((((5.e-1))*(dfc.txpl[0])))+((5.e-1)))))))
    xs_mask=(((((((5.e-1))+((((-5.e-1))*(dfc.txpl[0])))))<(xs_off))) & (((xs_off)<((((-5.e-1))+((((5.e-1))*(dfc.txpl[0]))))))))
    arg_nomchirp=((((((np.pi)/((1.8e+2))))*(ph)))+(((-2)*(np.pi)*(((((xs_off)*(p0)*(((dfc.txpsf[0])+((((5.e-1))*(dfc.txpl[0])*(dfc.txprr[0])))))))+(((((xs_off)**(2)))*(p1)*((5.e-1))*(dfc.txprr[0]))))))))
    z=((amp)*(xs_mask)*(np.exp(((1j)*(arg_nomchirp)))))
    return np.concatenate((np.abs(z),np.unwrap(np.angle(z)),))
def fun_nomchirpz(xs, delta_t, ph, p0, p1):
    delta_t=(0.0e+0)
    amp=(((0.0e+0))+((7.5e+2)))
    xs_off=((xs)-(((delta_t)+((((((5.e-1))*(dfc.txpl[0])))+((5.e-1)))))))
    xs_mask=(((((((5.e-1))+((((-5.e-1))*(dfc.txpl[0])))))<(xs_off))) & (((xs_off)<((((-5.e-1))+((((5.e-1))*(dfc.txpl[0]))))))))
    arg_nomchirp=((((((np.pi)/((1.8e+2))))*(ph)))+(((-2)*(np.pi)*(((((xs_off)*(p0)*(((dfc.txpsf[0])+((((5.e-1))*(dfc.txpl[0])*(dfc.txprr[0])))))))+(((((xs_off)**(2)))*(p1)*((5.e-1))*(dfc.txprr[0]))))))))
    z=((amp)*(xs_mask)*(np.exp(((1j)*(arg_nomchirp)))))
    return z
def fun_nomchirparg(xs, delta_t, ph, p0, p1):
    delta_t=(0.0e+0)
    amp=(((0.0e+0))+((7.5e+2)))
    xs_off=((xs)-(((delta_t)+((((((5.e-1))*(dfc.txpl[0])))+((5.e-1)))))))
    xs_mask=(((((((5.e-1))+((((-5.e-1))*(dfc.txpl[0])))))<(xs_off))) & (((xs_off)<((((-5.e-1))+((((5.e-1))*(dfc.txpl[0]))))))))
    arg_nomchirp=((((((np.pi)/((1.8e+2))))*(ph)))+(((-2)*(np.pi)*(((((xs_off)*(p0)*(((dfc.txpsf[0])+((((5.e-1))*(dfc.txpl[0])*(dfc.txprr[0])))))))+(((((xs_off)**(2)))*(p1)*((5.e-1))*(dfc.txprr[0]))))))))
    z=((amp)*(xs_mask)*(np.exp(((1j)*(arg_nomchirp)))))
    return arg_nomchirp
p0=((0.0e+0),(-9.85e+1),(1.e+0),(1.e+0),)
opt, opt2=scipy.optimize.curve_fit(fun_nomchirp_polar, xs_a_us, np.concatenate((np.abs(reps[0]),np.unwrap(np.angle(reps[0])),)), p0=p0)
# %% fit polynomial to phase
a=np.abs(reps[0])
arg=np.unwrap(np.angle(reps[0]))
th=((th_level)*(np.max(a)))
mask=((th)<(a))
start=np.argmax(mask)
end=((len(mask))-(np.argmax(mask[::-1])))
cut=arg[start:end]
fdec=dfc.iloc[0].fdec
start_us=((start)/(fdec))
end_us=((end)/(fdec))
xs_a_us=((np.arange(len(a)))/(fdec))
xs=xs_a_us[start:end]
cbarg, cbarg_diag=np.polynomial.chebyshev.chebfit(xs, cut, 2, full=True)
plt.plot(xs, np.real(((np.polynomial.chebyshev.chebval(xs, cba))*(np.exp(((1j)*(np.polynomial.chebyshev.chebval(xs, cbarg))))))), label="cheb_full")
plt.plot(xs, np.polynomial.chebyshev.chebval(xs, cba))
plt.axvline(x=start_us, color="r")
plt.axvline(x=end_us, color="r")
plt.xlim(((start_us)-(10)), ((end_us)+(10)))
plt.xlabel("time (us)")
plt.legend()
plt.subplot2grid(pl, (1,0,))
plt.plot(xs, np.abs(((reps[0][start:end])-(((np.polynomial.chebyshev.chebval(xs, cba))*(np.exp(((1j)*(np.polynomial.chebyshev.chebval(xs, cbarg))))))))), label="cheb full")
plt.legend()
plt.xlim(((start_us)-(10)), ((end_us)+(10)))
plt.axvline(x=start_us, color="r")
plt.axvline(x=end_us, color="r")
plt.xlabel("time (us)")
# %% show phase fit
plt.figure()
pl=(3,1,)
plt.subplot2grid(pl, (0,0,))
plt.plot(xs_a_us, ((arg)-(0)), label="arg_meas")
polyarg=np.polynomial.polynomial.polyfit(xs, np.polynomial.chebyshev.chebval(xs, cbarg), 2)
xs_extreme=np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(polyarg))
xs_off=((xs)-(xs_extreme))
polyarg2_a=((dfc.txpsf[0])+((((5.e-1))*(dfc.txpl[0])*(dfc.txprr[0]))))
polyarg2_b=(((5.e-1))*(dfc.txprr[0]))
polyarg2=((-2)*(np.pi)*(((((xs_off)*(polyarg2_a)))+(((((xs_off)**(2)))*(polyarg2_b))))))
polyarg3_a=dfc.txpsf[0]
polyarg3_b=(((5.e-1))*(dfc.txprr[0]))
xs3=((xs)-(((xs_extreme)-((((5.e-1))*(dfc.iloc[0].txpl))))))
polyarg3=((-2)*(np.pi)*(((((xs3)*(polyarg3_a)))+(((((xs3)**(2)))*(polyarg3_b))))))
nomchirp_xs=((np.arange(len(a)))/(dfc.iloc[0].fdec))
nomchirp_mask=((nomchirp_xs)<(dfc.txpl[0]))
nomchirp_im=(((((7.5e+2))*(nomchirp_mask)))*(np.exp(((1j)*(((-2)*(np.pi)*(((((nomchirp_xs)*(dfc.txpsf[0])))+(((((nomchirp_xs)**(2)))*((((5.e-1))*(dfc.txprr[0])))))))))))))
def make_chirp(row, n=6000):
    nomchirp_xs=((np.arange(n))/(row.fdec))
    nomchirp_mask=((nomchirp_xs)<(row.txpl))
    nomchirp_im=(((((7.5e+2))*(nomchirp_mask)))*(np.exp(((1j)*(((-2)*(np.pi)*(((((nomchirp_xs)*(row.txpsf)))+(((((nomchirp_xs)**(2)))*((((5.e-1))*(row.txprr)))))))))))))
    return (nomchirp_im,nomchirp_xs,)
nomchirp_0=make_chirp(dfc.iloc[0])
nomchirp_im_0=make_chirp(df.iloc[0], n=ss.shape[1])
plt.plot(xs, ((np.polynomial.chebyshev.chebval(xs, cbarg))-(0)), label="arg_cheb")
plt.plot(xs, ((np.polynomial.polynomial.polyval(xs, polyarg))-(0)), label="arg_poly")
plt.plot(xs, polyarg3, label="arg_poly3")
plt.plot(nomchirp_xs, np.unwrap(np.angle(nomchirp_im)), label="nomchirp_im")
plt.axvline(x=start_us, color="r")
plt.axvline(x=end_us, color="r")
plt.xlim(((start_us)-(10)), ((end_us)+(10)))
plt.xlabel("time (us)")
plt.legend()
plt.subplot2grid(pl, (1,0,))
scale=(((3.6e+2))/((((2.e+0))*(np.pi))))
plt.ylabel("phase error (deg)")
plt.plot(xs, ((scale)*(((cut)-(np.polynomial.chebyshev.chebval(xs, cbarg))))), label="cbarg")
plt.plot(xs, ((scale)*(((cut)-(np.polynomial.polynomial.polyval(xs, polyarg))))), label="arg_poly")
plt.plot(xs, ((scale)*(((((cut)-(0)))-(((-2)*(np.pi)*(((((xs3)*(polyarg3_a)))+(((((xs3)**(2)))*(polyarg3_b)))))))))), label="arg_poly3")
plt.legend()
plt.xlim(((start_us)-(10)), ((end_us)+(10)))
plt.xlabel("time (us)")
plt.axvline(x=start_us, color="r")
plt.axvline(x=end_us, color="r")
plt.subplot2grid(pl, (2,0,))
plt.ylabel("phase wrapped (deg)")
plt.plot(xs_a_us, np.angle(np.exp(((1j)*(arg)))), label="orig")
plt.plot(xs, np.angle(np.exp(((1j)*(np.polynomial.chebyshev.chebval(xs, cbarg))))), label="cheb_full")
plt.xlim(((start_us)-(10)), ((end_us)+(10)))
plt.legend()
plt.xlabel("time (us)")
plt.axvline(x=start_us, color="r")
plt.axvline(x=end_us, color="r")