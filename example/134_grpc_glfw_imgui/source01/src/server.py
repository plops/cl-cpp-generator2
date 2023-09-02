
# pip install grpcio grpcio-tools


# Then, compile your `glgui.proto` file to generate Python code:
#```bash
# python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. glgui.proto
#```

#Now, you can create the server:

import random
import grpc
from concurrent import futures
import glgui_pb2
import glgui_pb2_grpc
import numpy as np
import cv2

frame_counter = 0

class GLGuiService(glgui_pb2_grpc.GLGuiServiceServicer):

    def GetRandomRectangle(self, request, context):
        x1, y1 = random.uniform(0, 100), random.uniform(0, 100)
        x2, y2 = x1 + random.uniform(1, 20), y1 + random.uniform(1, 20)
        return glgui_pb2.RectangleResponse(x1=x1, y1=y1, x2=x2, y2=y2)

    def GetImage(self, request, context):
        global frame_counter

        w = request.width
        h = request.height
        image = np.zeros((h,w,3),dtype=np.uint8)

        M = min(w,h)
        circle_diameter = M/10.0 + (M/2.0-M/10.0) * np.sin(2*np.pi*frame_counter/20)/2

        cv2.circle(image, (w//2,h//2), int(circle_diameter//2), (255,255,255), -1)
        frame_counter += 1;
        return glgui_pb2.GetImageResponse(width=w,height=h,data=image.tobytes())

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
