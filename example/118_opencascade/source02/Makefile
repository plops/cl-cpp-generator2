CXX = g++
CXXFLAGS = -ggdb -O0 -Wall -Wextra -std=c++11 -I/usr/include/opencascade
LDFLAGS = -L/usr/lib/opencascade -lTKBin -lTKBinL -lTKBinXCAF -lTKBO -lTKBool -lTKBRep -lTKCAF -lTKCDF -lTKDCAF -lTKDraw -lTKernel -lTKFeat -lTKFillet -lTKG2d -lTKG3d -lTKGeomAlgo -lTKGeomBase -lTKHLR -lTKIGES -lTKLCAF -lTKMath -lTKMesh -lTKMeshVS -lTKOffset -lTKOpenGl -lTKOpenGlTest -lTKPrim -lTKQADraw -lTKRWMesh -lTKService -lTKShHealing -lTKStd -lTKStdL -lTKSTEP -lTKSTEP209 -lTKSTEPAttr -lTKSTEPBase -lTKSTL -lTKTopAlgo -lTKTopTest -lTKV3d -lTKVCAF -lTKViewerTest -lTKVRML -lTKXCAF -lTKXDEDRAW -lTKXDEIGES -lTKXDESTEP -lTKXMesh -lTKXml -lTKXmlL -lTKXmlXCAF -lTKXSBase -lTKXSDRAW -lGLEW

all: opencascade_example

opencascade_example: main.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f opencascade_example main.o
