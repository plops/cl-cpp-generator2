import subprocess
qtflags=subprocess.run(["pkg-config", "Qt5Gui", "Qt5Widgets", "--cflags"], capture_output=True)
cflags=["-std=c++20", "-ggdb", "-O1"]
cmd=((["clang++", "-###", "main.cpp", "-c"])+(cflags)+(qtflags.stdout.decode("utf-8").split(" ")))
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
    f.write(((clang_line1)+(" module.modulemap -o std_mod.pcm -emit-module -fmodules -fmodule-name=std_mod ")))
with open("compile02.sh", "w") as f:
    f.write("clang++ {} main.cpp -o main `pkg-config Qt5Gui Qt5Widgets --cflags --libs`".format(" ".join(cflags)))