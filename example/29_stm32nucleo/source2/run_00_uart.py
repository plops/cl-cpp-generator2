# sudo emerge pyserial
import sys
import time
import numpy as np
import serial
sys.path.append("/home/martin/src/nanopb/b/")
import simple_pb2 as pb
# %%
con=serial.Serial(port="/dev/ttyACM0", baudrate=1000000, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=(0.50    ), xonxoff=False, rtscts=False, writeTimeout=(5.00e-2), dsrdtr=False, interCharTimeout=(5.00e-2))
msg=pb.SimpleMessage()
time.sleep(1)
d=con.read_all()
print(d)
pbr=msg.ParseDelimitedFromString(d)
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