from numpy import *
import matplotlib.pyplot as plt
plt.ion()

x = array([1, 2, 3])
y = array([2.1, 2.3, 2.6])
sig = array([1,1,1])
N = len(x) # 3

bb,aa = polyfit(x,y,1) # 1.83 .249

S = sum(1/sig**2) #3
Sx = sum(x/sig**2) #6
Sxx = sum((x/sig)**2) # 14
Delta = S*Sxx - Sx**2 # 6
Sy = sum(y/sig**2) #7
t = (x-Sx/S)/sig # -1 0 1
Stt = sum(t**2) # 2

b = sum(t * y  / sig) / Stt # .25
a = (Sy - Sx * b) / S # 1.83


chi2 = sum(((y-a-b*x)/sig)**2) # .0017
fac = sqrt(chi2/(N-2)) # .04
siga = Sxx / Delta * fac # .09
sigb = S / Delta * fac #.02
# uncertainties of a and b estimated by assuming equal errors
# and that line is a good model

plt.plot(x,y)
