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
d0=con.read(((30)*(180)))
d=d0
res=[]
for i in range(30):
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
        res.append({("sample_nr"):(10),("sample"):(msg.sample10),("phase"):(msg.phase)})
        res.append({("sample_nr"):(11),("sample"):(msg.sample11),("phase"):(msg.phase)})
        res.append({("sample_nr"):(12),("sample"):(msg.sample12),("phase"):(msg.phase)})
        res.append({("sample_nr"):(13),("sample"):(msg.sample13),("phase"):(msg.phase)})
        res.append({("sample_nr"):(14),("sample"):(msg.sample14),("phase"):(msg.phase)})
        res.append({("sample_nr"):(15),("sample"):(msg.sample15),("phase"):(msg.phase)})
        res.append({("sample_nr"):(16),("sample"):(msg.sample16),("phase"):(msg.phase)})
        res.append({("sample_nr"):(17),("sample"):(msg.sample17),("phase"):(msg.phase)})
        res.append({("sample_nr"):(18),("sample"):(msg.sample18),("phase"):(msg.phase)})
        res.append({("sample_nr"):(19),("sample"):(msg.sample19),("phase"):(msg.phase)})
        res.append({("sample_nr"):(20),("sample"):(msg.sample20),("phase"):(msg.phase)})
        res.append({("sample_nr"):(21),("sample"):(msg.sample21),("phase"):(msg.phase)})
        res.append({("sample_nr"):(22),("sample"):(msg.sample22),("phase"):(msg.phase)})
        res.append({("sample_nr"):(23),("sample"):(msg.sample23),("phase"):(msg.phase)})
        res.append({("sample_nr"):(24),("sample"):(msg.sample24),("phase"):(msg.phase)})
        res.append({("sample_nr"):(25),("sample"):(msg.sample25),("phase"):(msg.phase)})
        res.append({("sample_nr"):(26),("sample"):(msg.sample26),("phase"):(msg.phase)})
        res.append({("sample_nr"):(27),("sample"):(msg.sample27),("phase"):(msg.phase)})
        res.append({("sample_nr"):(28),("sample"):(msg.sample28),("phase"):(msg.phase)})
        res.append({("sample_nr"):(29),("sample"):(msg.sample29),("phase"):(msg.phase)})
        res.append({("sample_nr"):(30),("sample"):(msg.sample30),("phase"):(msg.phase)})
        res.append({("sample_nr"):(31),("sample"):(msg.sample31),("phase"):(msg.phase)})
        res.append({("sample_nr"):(32),("sample"):(msg.sample32),("phase"):(msg.phase)})
        res.append({("sample_nr"):(33),("sample"):(msg.sample33),("phase"):(msg.phase)})
        res.append({("sample_nr"):(34),("sample"):(msg.sample34),("phase"):(msg.phase)})
        res.append({("sample_nr"):(35),("sample"):(msg.sample35),("phase"):(msg.phase)})
        res.append({("sample_nr"):(36),("sample"):(msg.sample36),("phase"):(msg.phase)})
        res.append({("sample_nr"):(37),("sample"):(msg.sample37),("phase"):(msg.phase)})
        res.append({("sample_nr"):(38),("sample"):(msg.sample38),("phase"):(msg.phase)})
        res.append({("sample_nr"):(39),("sample"):(msg.sample39),("phase"):(msg.phase)})
    except Exception as e:
        print(e)
        pass
last_len=msg.ByteSize()
df=pd.DataFrame(res)
dfi=df.set_index(["sample_nr", "phase"])
xs=dfi.to_xarray()
xrp.imshow(xs.sample)
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