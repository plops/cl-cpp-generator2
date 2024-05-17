#!/usr/bin/env python3
# sudo emerge -av dev-python/flatbuffers
# flatc --python image.fbs
import os
import time
import flatbuffers
import numpy as np
from MyImage.image_generated import Image
with open("image.bin", "rb") as f:
    buf=f.read()
image=Image.GetRootAsImage(bytes(buf), 0)
img=np.frombuffer(image.DataAsNumpy(), dtype=np.uint8)
img=img.reshape((image.Height(),image.Width(),))