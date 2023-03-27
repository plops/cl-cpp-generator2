CXX = g++

# https://developers.redhat.com/blog/2018/03/21/compiler-and-linker-flags-gcc
CXXFLAGS = -ggdb -O1 -Wall -Wextra \
-std=gnu++20 \
-I/usr/include/opencascade \
-D_FORTIFY_SOURCE=2 \
-D_GLIBCXX_ASSERTIONS \
-fasynchronous-unwind-tables \
-fexceptions \
-fstack-clash-protection \
-fstack-protector-all \
-fstack-protector-strong \
-grecord-gcc-switches \
-fcf-protection \
-pipe \
-Werror=format-security \
-Wl,-z,defs \
-Wl,-z,now \
-Wl,-z,relro

#-mcet
#-Werror=implicit-function-declaration 
# -fplugin=annobin

LDFLAGS = -L/usr/lib/opencascade -lTKBin -lTKBinL -lTKBinXCAF -lTKBO -lTKBool -lTKBRep -lTKCAF -lTKCDF -lTKDCAF -lTKDraw -lTKernel -lTKFeat -lTKFillet -lTKG2d -lTKG3d -lTKGeomAlgo -lTKGeomBase -lTKHLR -lTKIGES -lTKLCAF -lTKMath -lTKMesh -lTKMeshVS -lTKOffset -lTKOpenGl -lTKOpenGlTest -lTKPrim -lTKQADraw -lTKRWMesh -lTKService -lTKShHealing -lTKStd -lTKStdL -lTKSTEP -lTKSTEP209 -lTKSTEPAttr -lTKSTEPBase -lTKSTL -lTKTopAlgo -lTKTopTest -lTKV3d -lTKVCAF -lTKViewerTest -lTKVRML -lTKXCAF -lTKXDEDRAW -lTKXDEIGES -lTKXDESTEP -lTKXMesh -lTKXml -lTKXmlL -lTKXmlXCAF -lTKXSBase -lTKXSDRAW -lGLEW

all: opencascade_example

opencascade_example: main.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f opencascade_example main.o
