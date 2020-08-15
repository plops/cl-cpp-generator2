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
from enum import Enum
class State_FSM(Enum):
    START=0
    START_CHAR0=1
    START_CHAR1=2
    START_CHAR2=3
    START_CHAR3=4
    PACKET_LEN_LSB=5
    PACKET_LEN_MSB=6
    PAYLOAD=7
    END_CHAR0=8
    END_CHAR1=9
    END_CHAR2=10
    END_CHAR3=11
    END_CHAR4=12
    STOP=13
    ERROR=14
    FINISH=15
#  http://www.findinglisp.com/blog/2004/06/basic:automaton:macro.html
state=State_FSM.START
def parse_serial_packet_reset():
    global state
    state=State_FSM.START
def parse_serial_packet(con, accum={}):
    # returns tuple with 3 values (val, result, comment). If val==1 call again, if val==0 then fsm is in finish state. If val==:1 then FSM is in error state.
    global state
    result=""
    result_comment=accum
    if ( ((state)==(State_FSM.START)) ):
        current_char=con.read()
        result_comment["parsed_bytes"]=0
        print("{} current_state=START next-state=START_CHAR0 char={}".format(result_comment["parsed_bytes"], current_char))
        if ( ((current_char)==(b'U')) ):
            state=State_FSM.START_CHAR0
        else:
            state=State_FSM.START
    if ( ((state)==(State_FSM.START_CHAR0)) ):
        current_char=con.read()
        result_comment["parsed_bytes"]=((1)+(result_comment["parsed_bytes"]))
        print("{} current_state=START_CHAR0 next-state=START_CHAR1 char={}".format(result_comment["parsed_bytes"], current_char))
        if ( ((current_char)==(b'U')) ):
            state=State_FSM.START_CHAR1
        else:
            state=State_FSM.START
    if ( ((state)==(State_FSM.START_CHAR1)) ):
        current_char=con.read()
        result_comment["parsed_bytes"]=((1)+(result_comment["parsed_bytes"]))
        print("{} current_state=START_CHAR1 next-state=START_CHAR2 char={}".format(result_comment["parsed_bytes"], current_char))
        if ( ((current_char)==(b'U')) ):
            state=State_FSM.START_CHAR2
        else:
            state=State_FSM.START
    if ( ((state)==(State_FSM.START_CHAR2)) ):
        current_char=con.read()
        result_comment["parsed_bytes"]=((1)+(result_comment["parsed_bytes"]))
        print("{} current_state=START_CHAR2 next-state=START_CHAR3 char={}".format(result_comment["parsed_bytes"], current_char))
        if ( ((current_char)==(b'U')) ):
            state=State_FSM.START_CHAR3
        else:
            state=State_FSM.START
    if ( ((state)==(State_FSM.START_CHAR3)) ):
        current_char=con.read()
        result_comment["parsed_bytes"]=((1)+(result_comment["parsed_bytes"]))
        print("{} current_state=START_CHAR3 next-state=PACKET_LEN_LSB char={}".format(result_comment["parsed_bytes"], current_char))
        if ( ((current_char)==(b'U')) ):
            state=State_FSM.PACKET_LEN_LSB
        else:
            state=State_FSM.START
    if ( ((state)==(State_FSM.PACKET_LEN_LSB)) ):
        current_char=con.read()
        result_comment["parsed_bytes"]=((1)+(result_comment["parsed_bytes"]))
        print("{} current_state=PACKET_LEN_LSB char={}".format(result_comment["parsed_bytes"], current_char))
        result_comment["packet_len"]=current_char[0]
        state=State_FSM.PACKET_LEN_MSB
    if ( ((state)==(State_FSM.PACKET_LEN_MSB)) ):
        current_char=con.read()
        result_comment["parsed_bytes"]=((1)+(result_comment["parsed_bytes"]))
        result_comment["packet_len"]=((result_comment["packet_len"])+(((256)*(current_char[0]))))
        result_comment["packet_payload_bytes_read"]=0
        print("{} current_state=PACKET_LEN_MSB char={} packet_len={}".format(result_comment["parsed_bytes"], current_char, result_comment["packet_len"]))
        state=State_FSM.PAYLOAD
    if ( ((state)==(State_FSM.PAYLOAD)) ):
        current_char=con.read()
        print("{} current_state=PAYLOAD char={}".format(result_comment["parsed_bytes"], current_char))
        result_comment["parsed_bytes"]=((1)+(result_comment["parsed_bytes"]))
        result_comment["packet_payload_bytes_read"]=((result_comment["packet_payload_bytes_read"])+(1))
        if ( ((result_comment["packet_payload_bytes_read"])<(result_comment["packet_len"])) ):
            state=State_FSM.PAYLOAD
        else:
            state=State_FSM.END_CHAR0
    if ( ((state)==(State_FSM.END_CHAR0)) ):
        current_char=con.read()
        result_comment["parsed_bytes"]=((1)+(result_comment["parsed_bytes"]))
        print("{} current_state=END_CHAR0 next-state=END_CHAR1 char={}".format(result_comment["parsed_bytes"], current_char))
        if ( ((current_char)==(b'xff')) ):
            state=State_FSM.END_CHAR1
        else:
            state=State_FSM.ERROR
    if ( ((state)==(State_FSM.END_CHAR1)) ):
        current_char=con.read()
        result_comment["parsed_bytes"]=((1)+(result_comment["parsed_bytes"]))
        print("{} current_state=END_CHAR1 next-state=END_CHAR2 char={}".format(result_comment["parsed_bytes"], current_char))
        if ( ((current_char)==(b'xff')) ):
            state=State_FSM.END_CHAR2
        else:
            state=State_FSM.ERROR
    if ( ((state)==(State_FSM.END_CHAR2)) ):
        current_char=con.read()
        result_comment["parsed_bytes"]=((1)+(result_comment["parsed_bytes"]))
        print("{} current_state=END_CHAR2 next-state=END_CHAR3 char={}".format(result_comment["parsed_bytes"], current_char))
        if ( ((current_char)==(b'xff')) ):
            state=State_FSM.END_CHAR3
        else:
            state=State_FSM.ERROR
    if ( ((state)==(State_FSM.END_CHAR3)) ):
        current_char=con.read()
        result_comment["parsed_bytes"]=((1)+(result_comment["parsed_bytes"]))
        print("{} current_state=END_CHAR3 next-state=END_CHAR4 char={}".format(result_comment["parsed_bytes"], current_char))
        if ( ((current_char)==(b'xff')) ):
            state=State_FSM.END_CHAR4
        else:
            state=State_FSM.ERROR
    if ( ((state)==(State_FSM.END_CHAR4)) ):
        current_char=con.read()
        result_comment["parsed_bytes"]=((1)+(result_comment["parsed_bytes"]))
        print("{} current_state=END_CHAR4 next-state=FINISH char={}".format(result_comment["parsed_bytes"], current_char))
        if ( ((current_char)==(b'xff')) ):
            state=State_FSM.FINISH
        else:
            state=State_FSM.ERROR
    if ( ((state)==(State_FSM.FINISH)) ):
        return (0,result,result_comment,)
    if ( ((state)==(State_FSM.ERROR)) ):
        raise(Exception("error in parse_module_response"))
    return (1,result,result_comment,)
# %%
con=serial.Serial(port="/dev/ttyACM0", baudrate=115200, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=(0.50    ), xonxoff=False, rtscts=False, writeTimeout=(5.00e-2), dsrdtr=False, interCharTimeout=(5.00e-2))
class Listener():
    def __init__(self, connection):
        self._con=connection
    def _fsm_read(self):
        parse_serial_packet_reset()
        res=(1,"",{},)
        while (((1)==(res[0]))):
            res=parse_serial_packet(self._con, accum=res[2])
        response=res[1]
        return response
l=Listener(con)
l._fsm_read()