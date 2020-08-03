# sudo emerge pyserial
import matplotlib
import matplotlib.pyplot as plt
plt.ion()
import sys
import time
import numpy as np
import serial
import pandas as pd
import xarray as xr
import xarray.plot as xrp
sys.path.append("/home/martin/src/nanopb/b/")
import simple_pb2 as pb
# %%
con=serial.Serial(port="/dev/ttyACM0", baudrate=115200, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=(0.50    ), xonxoff=False, rtscts=False, writeTimeout=(5.00e-2), dsrdtr=False, interCharTimeout=(5.00e-2))
msg=pb.SimpleMessage()
d0=con.read(((40)*(180)))
d=d0
res=[]
for i in range(3):
    try:
        pattern=b"\xff\xff\xff\xff\xff\x55\x55\x55\x55\x55"
        start_idx=d.find(pattern)
        d=d[((start_idx)+(len(pattern))):]
        end_idx=d.find(pattern)
        d1=d[0:end_idx]
        pbr=msg.ParseFromString(d1)
        res.append({("sample_nr"):(0),("sample"):(msg.sample00),("phase"):(msg.phase)})
        res.append({("sample_nr"):(1),("sample"):(msg.sample01),("phase"):(msg.phase)})
        res.append({("sample_nr"):(2),("sample"):(msg.sample02),("phase"):(msg.phase)})
        res.append({("sample_nr"):(3),("sample"):(msg.sample03),("phase"):(msg.phase)})
        res.append({("sample_nr"):(4),("sample"):(msg.sample04),("phase"):(msg.phase)})
        res.append({("sample_nr"):(5),("sample"):(msg.sample05),("phase"):(msg.phase)})
        res.append({("sample_nr"):(6),("sample"):(msg.sample06),("phase"):(msg.phase)})
        res.append({("sample_nr"):(7),("sample"):(msg.sample07),("phase"):(msg.phase)})
        res.append({("sample_nr"):(8),("sample"):(msg.sample08),("phase"):(msg.phase)})
        res.append({("sample_nr"):(9),("sample"):(msg.sample09),("phase"):(msg.phase)})
    except Exception as e:
        print(e)
        pass
last_len=msg.ByteSize()
df=pd.DataFrame(res)
dfi=df.set_index(["sample_nr", "phase"])
class Uart():
    def __init__(self, connection, debug=False):
        self._con=connection
        self._debug=debug
    def _write(self, cmd):
        
        self._con.write("{}\n".format(cmd).encode("utf-8"))
    def _read(self):
        # read all response lines from uart connections
        try:
            line=self._con.read_until()
            res=line.decode("ISO-8859-1")
            while (self._con.in_waiting):
                print(res)
                line=self._con.read_until()
                print("AW: {}".format(line))
                res=line.decode("ISO-8859-1")
        except Exception as e:
            print("warning in _read: {}. discarding the remaining input buffer {}.".format(e, self._con.read(size=self._con.in_waiting)))
            self._con.reset_input_buffer()
            return np.nan
        return res
    def close(self):
        self._con.close()
u=Uart(con)