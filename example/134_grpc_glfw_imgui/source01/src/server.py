
# pip install grpcio grpcio-tools


# Then, compile your `glgui.proto` file to generate Python code:
#```bash
#python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. glgui.proto
#```

#Now, you can create the server:

import random
import grpc
from concurrent import futures
import glgui_pb2
import glgui_pb2_grpc

class GLGuiService(glgui_pb2_grpc.GLGuiServiceServicer):

    def GetRandomRectangle(self, request, context):
        x1, y1 = random.uniform(0, 100), random.uniform(0, 100)
        x2, y2 = x1 + random.uniform(1, 20), y1 + random.uniform(1, 20)
        return glgui_pb2.RectangleResponse(x1=x1, y1=y1, x2=x2, y2=y2)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    glgui_pb2_grpc.add_GLGuiServiceServicer_to_server(GLGuiService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()


# This server listens on port 50051 and returns random rectangle
# coordinates when the `GetRandomRectangle` method is called. To test
# it, you'd also create a gRPC client that calls this method.
