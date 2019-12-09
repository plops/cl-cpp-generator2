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
dfc=pd.read_csv("./o_cal_range.csv")
dfc["cal_type_desc"]=list(map(lambda x: cal_type_desc[x], dfc.cal_type))
dfc.cal_type_desc=dfc.cal_type_desc.astype("category")
s=np.memmap(next(pathlib.Path("./").glob("o_cal*.cf")), dtype=np.complex64, mode="r", shape=(800,6000,))
ss=np.memmap(next(pathlib.Path("./").glob("o_r*.cf")), dtype=np.complex64, mode="r", shape=(16516,24695,))
fref=(3.7534721374511715e+1)
input=(((5.e-1))*(((s[1,:3000])-(s[0,:3000]))))
xs=((np.arange(len(input)))/(fref))
plt.figure()
chirp_phase=np.unwrap(np.angle(input))
dfchirp=pd.DataFrame({("xs"):(xs),("mag"):(np.abs(input))}).set_index("xs")
dfchirpon=((150)<(dfchirp))
chirp_start=dfchirpon[((dfchirpon.mag)==(True))].iloc[0].name
chirp_end=dfchirpon[((dfchirpon.mag)==(True))].iloc[-1].name
chirp_poly=np.polynomial.polynomial.Polynomial.fit(xs, chirp_phase, 2, domain=[chirp_start, chirp_end])
plt.plot(xs, np.unwrap(np.angle(input)))
plt.plot(xs, chirp_poly(xs))
plt.figure()
plt.plot(xs, np.real(input))
plt.plot(xs, np.real(((175)*(np.exp(((1j)*(chirp_poly(xs))))))))