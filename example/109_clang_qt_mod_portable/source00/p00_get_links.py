import subprocess
cmd=["clang++", "-###", "main.cpp", "-c", "-std=c++20", "-ggdb", "-O1"]
print("calling: {}".format(" ".join(cmd)))
out=subprocess.run(cmd, capture_output=True)
count=0
start=-1
for line in out.stderr.decode("utf-8").split("\n"):
    if ( ("(in-process)" in line) ):
        start=count
    if ( ((((0)<(start))) and (((((1)+(start)))==(count)))) ):
        print("{}: '{}'".format(count, line))
    count += 1