#|default_exp p02_pb_client
# pip3 install --user protobuf
import time
import socket
import struct
from data_pb2 import Packet, DataRequest, DataResponse
start_time=time.time()
debug=True
_code_git_version="b8aaa0ef6bddd98b599388373f5b8d9541a6c3e1"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/121_nanopb/source/"
_code_generation_time="07:20:51 of Tuesday, 2023-04-11 (GMT+1)"
s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("localhost",1234,))
request=DataRequest(count=123, start_index=12345)
request_string=request.SerializeToString()
print("{} nil request_string={}".format(((time.time())-(start_time)), request_string))
opacket=Packet(length=len(request_string), payload=request_string)
opacket_string=opacket.SerializeToString()
s.sendall(opacket_string)
time.sleep((0.20    ))
data=s.recv(9600)
print("{} nil data={}".format(((time.time())-(start_time)), data))
response_packet=Packet()
response_packet.ParseFromString(data)
response=DataResponse()
response.ParseFromString(response_packet.payload)
print("{} nil response={}".format(((time.time())-(start_time)), response))
s.close()