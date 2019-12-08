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
cal_type_dtype=pd.api.types.CategoricalDtype(ordered=True, categories=["tx_cal", "rx_cal", "epdn_cal", "ta_cal", "apdn_cal", "na_0", "na_1", "txh_cal_iso"])
dfc=pd.read_csv("./o_cal_range.csv")
s=np.memmap(next(pathlib.Path("./").glob("o_cal*.cf")), dtype=np.complex64, mode="r", shape=(800,6000,))
fref=(3.7534721374511715e+1)
input=(((5.e-1))*(((s[0,:3000])-(s[1,:3000]))))
xs=((np.arange(len(input)))/(fref))
plt.plot(xs, np.real(input), label="real")
plt.plot(xs, np.imag(input), label="imag")
row=dfc.iloc[0]
txprr=row.txprr
txprr_=row.txprr_
txpsf=row.txpsf
txpl=row.txpl
txpl_=row.txpl_
steps=((-50)+(np.linspace(0, 3000, 3001)))
tn=((steps)*((((1.e+0))/(fref))))
p1=((txpsf)-(((txprr)*((-5.e-1))*(txpl))))
p2=(((5.e-1))*(txprr))
arg=((((p1)*(tn)))+(((p2)*(tn)*(tn))))
ys=((175)*(np.exp(((-2j)*(np.pi)*(arg)))))
def chirp(tn, amp, p1, p2, xoffset, xscale):
    tns=((xscale)*(tn))
    tnso=((tns)-(xoffset))
    arg=((((p1)*(tnso)))+(((p2)*(tnso)*(tnso))))
    z=((amp)*(np.exp(((-2j)*(np.pi)*(arg)))))
    return np.concatenate((np.real(z),np.imag(z),))
p0=((1.75e+2),((txpsf)-(((txprr)*((-5.e-1))*(txpl)))),(((5.e-1))*(txprr)),(0.0e+0),(1.e+0),)
opt, opt2=scipy.optimize.curve_fit(chirp, ((np.arange(len(input)))/(fref)), np.concatenate((np.real(input),np.imag(input),)), p0=p0)
plt.plot(xs, chirp(xs, *p0)[:3000], label="init_re")
plt.plot(xs, chirp(xs, *opt)[:3000], label="fit_re")
plt.legend()
fig=plt.figure()
ax=fig.add_subplot("111")
ax.imshow(np.angle(s))
ax.set_aspect("auto")