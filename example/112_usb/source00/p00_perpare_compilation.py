#!/usr/bin/env python3
import subprocess
cflags=["-std=c++20", "-ggdb", "-O1"]
cmd=((["clang++", "-###", "main.cpp", "-c"])+(cflags))
print("calling : subprocess.run {}".format(" ".join(cmd)))
out=subprocess.run(cmd, capture_output=True)
count=0
start=-1
clang_line0=None
for line in out.stderr.decode("utf-8").split("\n"):
    if ( ("(in-process)" in line) ):
        start=count
    if ( ((((0)<(start))) and (((((1)+(start)))==(count)))) ):
        clang_line0=line
    count += 1
clang_line1=clang_line0.replace("\"-o\" \"main.o\"", "").replace("\"-main-file-name\" \"main.cpp\"", "").replace("\"main.cpp\"", "")
with open("compile01.sh", "w") as f:
    f.write((("time ")+(clang_line1)+(" module.modulemap -o fatheader.pcm -emit-module -fmodules -fmodule-name=fatheader")))
with open("compile02.sh", "w") as f:
    for file in ["main", "UsbError"]:
        flags=" ".join(((cflags)))
        f.write(f"time clang++ {flags} -fmodule-file=fatheader.pcm {file}.cpp -c -o {file}.o\n")
    f.write("time clang++ {} main.o UsbError.o -o main `pkg-config libusb-1.0 --libs`\n".format(" ".join(((cflags)))))