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
try:
    start_idx=d.find(b"\x55\x55\x55\x55\x55")
    end_idx=((start_idx)+(5)+(d[start_idx+5:].find(b"\x55\x55\x55\x55\x55")))
    d1=d[start_idx+5:end_idx]
    pbr=msg.ParseFromString(d1)
except Exception as e:
    print(e)
    pass
last_len=msg.ByteSize()
res=[]
for i in range(30):
    try:
        msg2=pb.SimpleMessage()
        start_idx2=d[end_idx+5:].find(b"\x55\x55\x55\x55\x55")
        end_idx2=((start_idx2)+(5)+(d[start_idx2+5:].find(b"\x55\x55\x55\x55\x55")))
        d2=d0[start_idx2:end_idx2]
        pbr2=msg2.ParseFromString(d2)
        res.append({("sample_nr"):(0),("sample"):(msg2.sample00),("phase"):(msg2.phase)})
        res.append({("sample_nr"):(1),("sample"):(msg2.sample01),("phase"):(msg2.phase)})
        res.append({("sample_nr"):(2),("sample"):(msg2.sample02),("phase"):(msg2.phase)})
        res.append({("sample_nr"):(3),("sample"):(msg2.sample03),("phase"):(msg2.phase)})
        res.append({("sample_nr"):(4),("sample"):(msg2.sample04),("phase"):(msg2.phase)})
        res.append({("sample_nr"):(5),("sample"):(msg2.sample05),("phase"):(msg2.phase)})
        res.append({("sample_nr"):(6),("sample"):(msg2.sample06),("phase"):(msg2.phase)})
        res.append({("sample_nr"):(7),("sample"):(msg2.sample07),("phase"):(msg2.phase)})
        res.append({("sample_nr"):(8),("sample"):(msg2.sample08),("phase"):(msg2.phase)})
        res.append({("sample_nr"):(9),("sample"):(msg2.sample09),("phase"):(msg2.phase)})
        start_idx=start_idx2
        end_idx=end_idx2
        last_len=msg2.ByteSize()
    except Exception as e:
        print("e={} i={}".format(e, i))
        pass
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