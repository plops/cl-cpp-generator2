#|default_exp p03_pb_server
# pip3 install --user protobuf
import time
import socket
import struct
from data_pb2 import Packet, DataRequest, DataResponse
start_time=time.time()
debug=True
_code_git_version="513bb457a30c3a9eab6520bd23d703822dc6a3bc"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/121_nanopb/source/"
_code_generation_time="08:40:53 of Wednesday, 2023-04-12 (GMT+1)"
def listen():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost",1234,))
        s.listen()
        print("{} listening on localhost:1234 ".format(((time.time())-(start_time))))
        while (True):
            conn, addr=s.accept()
            with conn:
                print("{} connection addr={}".format(((time.time())-(start_time)), addr))
                data=conn.recv(1024)
                buf=data
                while (data):
                    data=conn.recv(1024)
                    buf += data
                request=DataResponse().ParseFromString(buf)
                print("{} nil request.temperature={} request.co2_concentration={}".format(((time.time())-(start_time)), request.temperature, request.co2_concentration))
                reply=DataRequest(start_index=123).SerializeToString()
                conn.sendall(reply)
                print("{} connection closed ".format(((time.time())-(start_time))))
listen()