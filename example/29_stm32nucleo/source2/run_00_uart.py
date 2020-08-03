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
d0=con.read(180)
time.sleep((1.00e-2))
d1=con.read_all()
d=d1
try:
    start_idx=d.find(b"\x08\xd5\xaa")
    d1=d[start_idx:start_idx+180]
    pbr=msg.ParseFromString(d1)
except Exception as e:
    print(e)
    pass
last_len=msg.ByteSize()
res=[]
for i in range(30):
    try:
        time.sleep((3.00e-2))
        d0=con.read_all()
        start_idx2=d0.find(b"\x08\xd5\xaa")
        msg2=pb.SimpleMessage()
        d2=d0[start_idx2:start_idx2+180]
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
        last_len=msg2.ByteSize()
    except Exception as e:
        print(e)
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