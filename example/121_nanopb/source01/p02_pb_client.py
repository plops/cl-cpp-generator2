#|default_exp p02_pb_client
# pip3 install --user protobuf
import time
import socket
import struct
from data_pb2 import Packet, DataRequest, DataResponse
start_time=time.time()
debug=True
_code_git_version="360bdd43fb4c044a0c098a4f576d0df6be390d32"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/121_nanopb/source/"
_code_generation_time="22:07:30 of Tuesday, 2023-04-11 (GMT+1)"
def talk():
    s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("localhost",1234,))
    request=DataRequest(count=123, start_index=12345)
    request_string=request.SerializeToString()
    print("{} nil request_string={}".format(((time.time())-(start_time)), request_string))
    s.sendall(request_string)
    # close the write channel of the socket, so that the server receives EOF and knows the request has finished
    s.shutdown(socket.SHUT_WR)
    # this sends a FIN. The server will respond with ACK once it received all remaining bytes of the request
    # https://stackoverflow.com/questions/4160347/close-vs-shutdown-socket/23483487#23483487
    # what remains is to wait until recv returns 0 (the server finished its response and closed his socket)
    buf=b''
    while (True):
        data=s.recv(1024)
        if ( not(data) ):
            # recv return 0 (EOF), so we break out of loop
            break
        buf += data
    print("{} nil buf={}".format(((time.time())-(start_time)), buf))
    response=DataResponse()
    response.ParseFromString(buf)
    print("{} nil response.index={} response.datetime={} response.pressure={} response.humidity={} response.temperature={} response.co2_concentration={}".format(((time.time())-(start_time)), response.index, response.datetime, response.pressure, response.humidity, response.temperature, response.co2_concentration))
    s.close()
talk()