#!/usr/bin/env python3
# python -m venv ~/pytorch_env
# . ~/pytorch_env/bin/activate
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu11
# pip install lmfit
import os
import time
import pathlib
import re
import sys
import pandas as pd
directory=pathlib.Path("/home/martin/stage/cl-cpp-generator2")
training_data=[]
for f in ((directory)/("example")).rglob("gen*.lisp"):
    # exclude python generating files
    content=f.read_text()
    if ( re.search(r"""\(ql:quickload "cl-py-generator"\)""", content) ):
        print(f"(string Info 0: Skip python generator {f}.)")
        continue
    # genXX.lisp -> sourceXX
    output_dir=((f.parent)/("source{}".format(f.stem[3:5])))
    if ( output_dir.exists() ):
        output_files=((list(output_dir.rglob("*.cpp")))+(list(output_dir.rglob("*.c")))+(list(output_dir.rglob("*.h")))+(list(output_dir.rglob("*.hpp")))+(list(output_dir.rglob("*.cu")))+(list(output_dir.rglob("*.cl")))+(list(output_dir.rglob("*CMakeLists.txt"))))
        if ( ((0)<(len(output_files))) ):
            print(f"Info 1: Found match {f} {len(output_files)}.")
            lisp_content=f.read_text()
            text_input=""
            for output_file in output_files:
                text_input += f"// {output_file}\n{output_file.read_text()}\n\n"
            training_data.append(dict(text_input=text_input, output=lisp_content))
        else:
            print(f"Warning 1: No matches in output directory for {f}.")
            continue
    else:
        content=f.read_text()
        match=re.search(r"""\(defparameter \*source-dir\* .*\"(.*)\"\)""", content)
        if ( match ):
            output_dir=((directory)/(match.group(1)))
            output_files=((list(output_dir.rglob("*.cpp")))+(list(output_dir.rglob("*.c")))+(list(output_dir.rglob("*.h")))+(list(output_dir.rglob("*.hpp")))+(list(output_dir.rglob("*.cu")))+(list(output_dir.rglob("*.cl")))+(list(output_dir.rglob("*CMakeLists.txt"))))
            if ( ((0)<(len(output_files))) ):
                print(f"Info 2: Found match {f} {len(output_files)}.")
                lisp_content=f.read_text()
                text_input=""
                for output_file in output_files:
                    text_input += f"// {output_file}\n{output_file.read_text()}\n\n"
                training_data.append(dict(text_input=text_input, output=lisp_content))
            else:
                print(f"Warning 2: Not enough files for {f} in {output_dir} gp1={match.group(1)}.")
                print(f"Warning 4: match={match} ls {output_dir}={output_files}.")
                continue
        else:
            print(f"Warning 3: Could not determine output directory for {f}.")
            continue
df=pd.DataFrame(training_data)
df["text_input_len"]=df.text_input.str.len()
df["output_len"]=df.output.str.len()
df1=df[((((df.text_input_len)<(40000))) & (((df.output_len)<(5000))))]
df1.to_csv("training_data.csv", index=False)