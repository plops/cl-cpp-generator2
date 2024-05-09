# Intro

- this directory contains python code to collect the input
  s-expressions and corresponding output for all the examples in
  cl-cpp-generator2
  
- the data is collected in a json file of this format:

```
[
        {
             'text_input': '1',
             'output': '2',
        },{
             'text_input': '3',
             'output': '4',
        },{
...
}
]
```

- this will serve as finetuning input for a large language model

- the llm shall transform c++ input code to s-expressions 


# Python code

i have a directory with many examples of input lisp file (always named gen[0-9][0-9].lisp) and corresponding output:
```
/home/martin/stage/cl-cpp-generator2/example/44_asio/source/simple_01_server.cpp
/home/martin/stage/cl-cpp-generator2/example/44_asio/source/vis_03_connection.cpp
/home/martin/stage/cl-cpp-generator2/example/44_asio/source/simple_00_client.cpp
/home/martin/stage/cl-cpp-generator2/example/44_asio/source/vis_02_tsqueue.cpp
/home/martin/stage/cl-cpp-generator2/example/44_asio/source/utils.h
/home/martin/stage/cl-cpp-generator2/example/44_asio/source/vis_00_base.cpp
/home/martin/stage/cl-cpp-generator2/example/44_asio/source/vis_05_server.cpp
/home/martin/stage/cl-cpp-generator2/example/44_asio/source/globals.h
/home/martin/stage/cl-cpp-generator2/example/12_business_card_ray/gen00.lisp
/home/martin/stage/cl-cpp-generator2/example/12_business_card_ray/source/proto2.h
/home/martin/stage/cl-cpp-generator2/example/12_business_card_ray/source/utils.h
/home/martin/stage/cl-cpp-generator2/example/12_business_card_ray/source/globals.h
/home/martin/stage/cl-cpp-generator2/example/92_pipewire/util.lisp
/home/martin/stage/cl-cpp-generator2/example/92_pipewire/gen00.lisp
/home/martin/stage/cl-cpp-generator2/example/64_opencv_star_video/gen00.lisp
/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar/gen01py.lisp
/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar/gen02.lisp
/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar/gen03py.lisp
/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar/source_doppler/doppler_01_mmap.cpp
/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar/source_doppler/doppler_00_main.cpp
/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar/source_doppler/doppler_03_draw.cpp
/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar/source_doppler/proto2.h
/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar/source_doppler/doppler_02_glfw_window.cpp
/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar/source_doppler/utils.h
/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar/source_doppler/doppler_05_gui.cpp
/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar/source_doppler/globals.h
/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar/gen00.lisp
/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar/source/copernicus_02_collect_packet_headers.cpp
/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar/source/proto2.h
/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar/source/copernicus_00_main.cpp
/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar/source/copernicus_04_decode_packet.cpp
/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar/source/copernicus_01_mmap.cpp
/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar/source/copernicus_06_decode_sub_commutated_data.cpp
/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar/source/copernicus_03_process_packet_headers.cpp
/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar/source/utils.h
/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar/source/copernicus_05_decode_type_ab_packet.cpp
/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar/source/copernicus_07_decode_type_c_packet.cpp
/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar/source/globals.h
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/Egl.h
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/Vertex.h
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/format-inl.h
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/Geometry.h
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/Program.h
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/AttribPointer.cpp
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/App.h
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/format.h
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/Cube.h
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/main.cpp
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/Framebuffer.cpp
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/AttribPointer.h
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/App.cpp
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/core.h
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/Renderer.h
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/android_native_app_glue.h
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/Framebuffer.h
/home/martin/stage/cl-cpp-generator2/example/95_vr/util.lisp
/home/martin/stage/cl-cpp-generator2/example/95_vr/gen00.lisp
```

the output can be cpp, c or h files. typically they are placed into a `source<a><b>` folder if the input file was called `gen<a><b>.lisp`.

however, sometimes this mapping to the output directory is not so clear. e.g.
/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar/gen02.lisp

is written into the folder source_doppler. this can be seen inside of gen02.lisp:
```common-lisp
  (defparameter *source-dir* #P"example/08_copernicus_radar/source_doppler/")
```

create python code that

collects all input *.lisp files and stores them together with the output files in the following json datastructure:

```
[
        {
             'text_input': '<content of the files generated by gen00.lisp (potentially muliple .cpp .c and .h files) >',
             'output': '<content of gen00.lisp>',
        },{
             'text_input': '3',
             'output': '4',
        },{
...
}
]
```

the json shall be used to train a large language model to create s-expression *.lisp files based on C++ input


here is the corresponding abreviated entry for 
/home/martin/stage/cl-cpp-generator2/example/95_vr/gen00.lisp


```json
        {
             'text_input': '// source00/App.cpp
  // no preamble
#include "App.h"
#include "VrApi.h"
#include "VrApi_Helpers.h"
#include "VrApi_Input.h"
#include "VrApi_SystemUtils.h"
#include "android_native_app_glue.h"
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
...

// source00/App.h
#ifndef APP_H
#define APP_H

#include "Egl.h"
#include "Renderer.h"
#include <iostream>
#include "VrApi.h"
...
',
  'output': '(eval-when (:compile-toplevel :execute :load-toplevel)
	    (ql:quickload "cl-cpp-generator2")
	    (ql:quickload "cl-change-case")
	    (ql:quickload "cl-ppcre"))

(in-package :cl-cpp-generator2)

(progn
  ;; for classes with templates use write-source and defclass+
  ;; for cpp files without header use write-source
  ...
  ',
}
```

note that the input may contain a listing of multiple files.

create python code that performs the following steps:

- iterate through directory and list input .lisp files with corresponding output files (if relationship is clear from filenames)
- list input .lisp files where output not clear, search for *source-dir* definition in those files and extract the directory.
- create the json file with training examples
- use pathlib


# code generated by gemini 1.5 pro:

- Import Libraries: Imports necessary libraries for file handling, regular expressions, and JSON manipulation.
- create_training_data(directory) Function:
    - Iterates through the directory and its subdirectories to find all .lisp files.
    - For each .lisp file, it tries to determine the corresponding output directory:
			- Checks if a sourceXX folder exists (where XX is the number from the Lisp filename).
            - If not, it searches for the *source-dir* definition within the Lisp file using regular expressions.
    - If the output directory is found, it gathers all .cpp, .c, and .h files within that directory.
    - It reads the content of the Lisp file and the combined content of the output files.
    - A training example is created with text_input (C++ code) and output (Lisp code) and added to the training_data list.
	- Save JSON: The collected training examples are saved to a JSON file named training_data.json.
	
```Python

import json
from pathlib import Path

def create_training_data(directory):
  """
  Creates a JSON file with training examples for a large language model using pathlib.

  Args:
    directory: The Path object representing the directory containing the input Lisp files and corresponding output files.
  """
  training_data = []

  for lisp_file in directory.rglob("*.lisp"):
        # Check for clear output file mapping (genXX.lisp -> sourceXX)
        output_dir = lisp_file.parent / f"source{lisp_file.stem[3:5]}"
        if output_dir.exists():
          output_files = list(output_dir.glob("*.cpp")) + list(output_dir.glob("*.c")) + list(output_dir.glob("*.h"))
        else:
          # Search for *source-dir* definition in Lisp file
          content = lisp_file.read_text()
          match = re.search(r'\(defparameter \*source-dir\* #P"(.*)"\)', content)
          if match:
              output_dir = Path(match.group(1))
              output_files = list(output_dir.glob("*.cpp")) + list(output_dir.glob("*.c")) + list(output_dir.glob("*.h"))
          else:
              print(f"Warning: Could not determine output directory for {lisp_file}")
              continue

        # Read content of Lisp and output files
        lisp_content = lisp_file.read_text()
        text_input = ""
        for output_file in output_files:
            text_input += f"// {output_file}\n{output_file.read_text()}\n\n"

        # Add training example to list
        training_data.append({
            "text_input": text_input,
            "output": lisp_content,
        })

  # Save training data to JSON file
  with open("training_data.json", "w") as f:
      json.dump(training_data, f, indent=2)

# Example usage
directory_path = "/home/martin/stage/cl-cpp-generator2/example"  # Replace with your actual directory path
directory = Path(directory_path)
create_training_data(directory)

```
