#|default_exp p03_pb_server
# pip3 install --user protobuf
import time
import socket
import struct
from data_pb2 import Packet, DataRequest, DataResponse
start_time=time.time()
debug=True
_code_git_version="f7bb5bcb05c5c17918e101e36421be83e473b968"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/121_nanopb/source/"
_code_generation_time="20:05:17 of Wednesday, 2023-04-12 (GMT+1)"
def listen():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost",1234,))
        s.listen()
        print("{} listening on localhost:1234 ".format(((time.time())-(start_time))))
        while (True):
            conn, addr=s.accept()
            with conn:
                print("{} connection addr={}".format(((time.time())-(start_time)), addr))
                print("{} wait for DataResponse message ".format(((time.time())-(start_time))))
                data=conn.recv(1024)
                buf=data
                while (data):
                    data=conn.recv(1024)
                    buf += data
                print("{} finished reading buf={}".format(((time.time())-(start_time)), buf))
                imsg=DataResponse()
                print("{} nil imsg.ParseFromString(buf)={}".format(((time.time())-(start_time)), imsg.ParseFromString(buf)))
                print("{} nil imsg.index={} imsg.datetime={} imsg.pressure={} imsg.humidity={} imsg.temperature={} imsg.co2_concentration={}".format(((time.time())-(start_time)), imsg.index, imsg.datetime, imsg.pressure, imsg.humidity, imsg.temperature, imsg.co2_concentration))
                print("{} send DataRequest message ".format(((time.time())-(start_time))))
                omsg=DataRequest(start_index=123, count=42).SerializeToString()
                conn.sendall(omsg)
                print("{} connection closed ".format(((time.time())-(start_time))))
listen()