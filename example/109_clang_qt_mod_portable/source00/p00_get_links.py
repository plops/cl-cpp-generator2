import subprocess
out=subprocess.run(["clang++", "-###", "main.cpp", "-c", "-std=c++20", "-ggdb", "-O1"], capture_output=True)
count=0
start=0
for line in out.split("\n"):
    if ( line.contains("(in-process)") ):
        start=((count)+(1))
    if ( ((count)<=(start)) ):
        print("{}: {}".format(count, line))
    count += 1