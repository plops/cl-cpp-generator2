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
define-automaton(quote(parse_serial_packet), quasiquote((START((current_char=con.read().decode())((if ( ((current_char)==("U")) ):
    result=((current_char)+(con.read().decode()))
    state=State_FSM.START_CHAR0
else:
    state=State_FSM.ERROR))))((FINISH((return (0,result,result_comment,))()), ERROR((raise(Exception("error in parse_module_response")))())))))
# %%
con=serial.Serial(port="/dev/ttyACM0", baudrate=115200, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=(0.50    ), xonxoff=False, rtscts=False, writeTimeout=(5.00e-2), dsrdtr=False, interCharTimeout=(5.00e-2))
msg=pb.SimpleMessage()
d0=con.read(((10)*(180)))
d=d0
d1=d0
res=[]
starting_point_found=False
starting_point_found_again=False
count=0
while (((not(starting_point_found_again)))):
    try:
        if ( ((200)<(len(d))) ):
            d=con.read(((10)*(180)))
        pattern=b"\xff\xff\xff\xff\xff\x55\x55\x55\x55\x55"
        start_idx=d.find(pattern)
        pkt_len_lsb=d[((5)+(5)+(0)+(start_idx))]
        pkt_len_msb=d[((5)+(5)+(1)+(start_idx))]
        pkt_len=((pkt_len_lsb)+(((256)*(pkt_len_msb))))
        pktfront=d[start_idx:((start_idx)+(5)+(5)+(2))]
        d=d[((5)+(5)+(2)+(start_idx)):]
        count=((count)+(1))
        dpkt=d[0:pkt_len]
        pktend=d[pkt_len:((pkt_len)+(5))]
        pbr=msg.ParseFromString(dpkt)
        if ( ((not(starting_point_found)) and (((msg.phase)==(3)))) ):
            starting_point_found=True
        if ( ((not(starting_point_found_again)) and (((msg.phase)==(3)))) ):
            starting_point_found_again=True
        if ( ((starting_point_found) and (not(starting_point_found_again))) ):
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
        print("count={} msg.phase={}".format(count, msg.phase))
    except Exception as e:
        print("exception while processing packet {}: {}".format(count, e))
        print("""start_idx={}
pktfront={}
pktend={}
dpkt={}
(len dpkt)={}
pkt_len={}""".format(start_idx, pktfront, pktend, dpkt, len(dpkt), pkt_len))
        f=open("/home/martin/stage/cl-cpp-generator2/example/29_stm32nucleo//source2/run_00_uart.py")
        content=f.readlines()
        f.close()
        lineno=sys.exc_info()[-1].tb_lineno
        for l in range(((lineno)-(3)), ((lineno)+(2))):
            print("{} {}".format(l, content[l][0:-1]))
        print("Error in line {}: {} '{}'".format(lineno, type(e).__name__, e))
        pass
last_len=msg.ByteSize()
df=pd.DataFrame(res)
dfi=df.set_index(["sample_nr", "phase"])
xs=dfi.to_xarray()
xrp.imshow(np.log(xs.sample))