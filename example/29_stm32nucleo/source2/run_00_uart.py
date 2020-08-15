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
    START_CHAR4=5
    PACKET_LEN_LSB=6
    PACKET_LEN_MSB=7
    PAYLOAD=8
    END_CHAR0=9
    END_CHAR1=10
    END_CHAR2=11
    END_CHAR3=12
    END_CHAR4=13
    STOP=14
    ERROR=15
    FINISH=16
#  http://www.findinglisp.com/blog/2004/06/basic:automaton:macro.html
state=State_FSM.START
def parse_serial_packet_reset():
    global state
    state=State_FSM.START
def parse_serial_packet(con):
    # returns tuple with 3 values (val, result, comment). If val==1 call again, if val==0 then fsm is in finish state. If val==:1 then FSM is in error state.
    global state
    result=""
    result_comment=""
    if ( ((state)==(State_FSM.START)) ):
        current_char=con.read()
        print("current_state=START next-state=START_CHAR0 char={}".format(current_char))
        if ( ((current_char)==(b'U')) ):
            result=((current_char)+(con.read()))
            state=State_FSM.START_CHAR0
        else:
            state=State_FSM.START
    if ( ((state)==(State_FSM.START_CHAR0)) ):
        current_char=con.read()
        print("current_state=START_CHAR0 next-state=START_CHAR1 char={}".format(current_char))
        if ( ((current_char)==(b'U')) ):
            result=((current_char)+(con.read()))
            state=State_FSM.START_CHAR1
        else:
            state=State_FSM.START
    if ( ((state)==(State_FSM.START_CHAR1)) ):
        current_char=con.read()
        print("current_state=START_CHAR1 next-state=START_CHAR2 char={}".format(current_char))
        if ( ((current_char)==(b'U')) ):
            result=((current_char)+(con.read()))
            state=State_FSM.START_CHAR2
        else:
            state=State_FSM.START
    if ( ((state)==(State_FSM.START_CHAR2)) ):
        current_char=con.read()
        print("current_state=START_CHAR2 next-state=START_CHAR3 char={}".format(current_char))
        if ( ((current_char)==(b'U')) ):
            result=((current_char)+(con.read()))
            state=State_FSM.START_CHAR3
        else:
            state=State_FSM.START
    if ( ((state)==(State_FSM.START_CHAR3)) ):
        current_char=con.read()
        print("current_state=START_CHAR3 next-state=START_CHAR4 char={}".format(current_char))
        if ( ((current_char)==(b'U')) ):
            result=((current_char)+(con.read()))
            state=State_FSM.START_CHAR4
        else:
            state=State_FSM.START
    if ( ((state)==(State_FSM.START_CHAR4)) ):
        current_char=con.read()
        print("current_state=START_CHAR4 next-state=PACKET_LEN_LSB char={}".format(current_char))
        if ( ((current_char)==(b'U')) ):
            result=((current_char)+(con.read()))
            state=State_FSM.PACKET_LEN_LSB
        else:
            state=State_FSM.START
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
        res=(1,"","",)
        while (((1)==(res[0]))):
            res=parse_serial_packet(self._con)
        response=res[1]
        return response
l=Listener(con)