#!/usr/bin/env python3
# python -m venv ~/pytorch_env
# . ~/pytorch_env/bin/activate
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu11
# pip install lmfit
import os
import time
import json
import pathlib
import re
directory=pathlib.Path("/home/martin/stage/cl-cpp-generator2/example")
training_data=[]
for f in directory.rglob("gen*.lisp"):
    # genXX.lisp -> sourceXX
    output_dir=((f.parent)/("source{}".format(f.stem[3:5])))
    if ( output_dir.exists() ):
        output_files=((list(output_dir.glob("*.cpp")))+(list(output_dir.glob("*.c")))+(list(output_dir.glob("*.h"))))
        if ( ((0)<(len(output_files))) ):
            print(f"Info 1: Found match {f} {len(output_files)}.")
        else:
            print(f"Warning 1: No matches in output directory for {f}.")
            continue
    else:
        content=f.read_text()
        match=re.search(r"""\(defparameter \*source-dir\* .*\"(.*)\"\)""", content)
        if ( match ):
            output_dir=((pathlib.Path(" /home/martin/stage/cl-cpp-generator2/"))/(match.group(1)))
            output_files=((list(output_dir.glob("*.cpp")))+(list(output_dir.glob("*.c")))+(list(output_dir.glob("*.h"))))
            if ( ((0)<(len(output_files))) ):
                print(f"Info 2: Found match {f} {len(output_files)}.")
            else:
                print(f"Warning 2: Not enough files for {f} in {output_dir} gp1={match.group(1)}.")
                print(f"Warning 4: match={match}.")
                continue
        else:
            print(f"Warning 3: Could not determine output directory for {f}.")
            continue