#|default_exp p02_pb_client
# pip3 install --user protobuf
import time
import socket
import struct
from data_pb2 import Packet, DataRequest, DataResponse
start_time=time.time()
debug=True
_code_git_version="1f8ae4347ee3ba58f05c4234789b2103bd6b9b0e"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/121_nanopb/source/"
_code_generation_time="07:17:03 of Tuesday, 2023-04-11 (GMT+1)"
s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("localhost",1234,))
request=DataRequest(count=123, start_index=12345)
request_string=request.SerializeToString()
print("{} nil request_string={}".format(((time.time())-(start_time)), request_string))
opacket=Packet(length=length(request_string), payload=request_string)
s.sendall(opacket)
time.sleep((0.20    ))
data=s.recv(9600)
print("{} nil data={}".format(((time.time())-(start_time)), data))
response_packet=Packet()
response_packet.ParseFromString(data)
response=DataResponse()
response.ParseFromString(response_packet.payload)
print("{} nil response={}".format(((time.time())-(start_time)), response))
s.close()