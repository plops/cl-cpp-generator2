# sudo emerge pyserial
import time
import numpy as np
import serial
# %%
con=serial.Serial(port="/dev/ttyACM0", baudrate=1000000, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=(0.50    ), xonxoff=False, rtscts=False, writeTimeout=(5.00e-2), dsrdtr=False, interCharTimeout=(5.00e-2))