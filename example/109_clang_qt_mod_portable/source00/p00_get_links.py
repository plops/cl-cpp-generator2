import subprocess
cmd=["clang++", "-###", "main.cpp", "-c", "-std=c++20", "-ggdb", "-O1"]
print("calling: {}".format(" ".join(cmd)))
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
print(clang_line1)