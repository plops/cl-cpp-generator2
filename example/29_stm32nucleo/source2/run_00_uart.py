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
d0=con.read_all()
time.sleep((1.00e-2))
d1=con.read_all()
d=d1
print(d)
try:
    start_idx=d.find(b"\x08\xd5\xaa")
    d1=d[start_idx:start_idx+180]
    pbr=msg.ParseFromString(d1)
except Exception as e:
    print(e)
    pass
last_len=msg.ByteSize()
res=[]
for i in range(10):
    try:
        time.sleep((0.10    ))
        d2=con.read_all()
        msg2=pb.SimpleMessage()
        start_idx2=((start_idx)+(last_len))
        d2=d[start_idx2:start_idx2+180]
        pbr2=msg2.ParseFromString(d2)
        d={("samples"):([msg2.sample00, msg2.sample01, msg2.sample02, msg2.sample03, msg2.sample04, msg2.sample05, msg2.sample06, msg2.sample07, msg2.sample08, msg2.sample09, msg2.sample10, msg2.sample11, msg2.sample12, msg2.sample13, msg2.sample14, msg2.sample15, msg2.sample16, msg2.sample17, msg2.sample18, msg2.sample19, msg2.sample20, msg2.sample21, msg2.sample22, msg2.sample23, msg2.sample24, msg2.sample25, msg2.sample26, msg2.sample27, msg2.sample28, msg2.sample29, msg2.sample30, msg2.sample31, msg2.sample32, msg2.sample33, msg2.sample34, msg2.sample35, msg2.sample36, msg2.sample37, msg2.sample38, msg2.sample39]),("phase"):(msg2.phase)}
        res.append(d)
        start_idx=start_idx2
        last_len=msg2.ByteSize()
    except Exception as e:
        print(e)
        pass
df=pd.DataFrame(res)
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