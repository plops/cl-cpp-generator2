#|default_exp p02_pb_client
# pip3 install --user protobuf
import time
import socket
import struct
from data_pb2 import DataRequest, DataResponse
start_time=time.time()
debug=True
_code_git_version="93479861c2b11e84b9c0d41b344fe0f71fc806ce"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/121_nanopb/source/"
_code_generation_time="23:43:49 of Monday, 2023-04-10 (GMT+1)"
s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("localhost",1234,))
request=DataRequest(count=123, start_index=12345)
request_string=request.SerializeToString()
print("{} nil request_string={}".format(((time.time())-(start_time)), request_string))
s.sendall(request_string)
time.sleep((0.20    ))
data=s.recv(1024)
print("{} nil data={}".format(((time.time())-(start_time)), data))
response=DataResponse()
response.ParseFromString(data)
print("{} nil response={}".format(((time.time())-(start_time)), response))
s.close()