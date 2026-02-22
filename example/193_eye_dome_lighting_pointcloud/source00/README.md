An experimental simple point cloud viewer for XYZ text files

I generated the code and documentation in gemini deep research using the following prompt:

```
deliverable to produce: a modular modern C++ example using AAA style to display a dense (assume all screen pixels covered) point cloud that is being loaded from an XYZ (.txt) file. the 3d scan has no colors, just surface position. use shaders similar to potree to enance surface undulations and features using Eye dome lighting. keep implementation as simple and straight forward as possible. if non-physical shortcuts can simplify code or enhance performce take them. use imgui to to allow parameter (e.g. strength, radius, offset, ...) changes at runtime. the code has to compile and run on linux. this is the primary platform but the code shall still be prepared for cross-platform deployment.

the report shall discuss the rationale for the apporaches


It is not needed to research or explain in detail the AAA idiom (just use auto). 
You don't have to research the fastest way to read the text files. Just use a simple way. 
the full source code shall be shown in one piece at the end of the report. 



the documentation needs more infomration on the dependencies. e.g. glad provides an online generator with which to create a specific version. you have to write instructions for that (what extensions are required):
this is written in the glad documentation:
```
Generate the bindings you need with the online generator, most commonly you just want to select an OpenGL version, add the OpenGL extensions you want and click generate.
If you're not using GLFW, SDL, Qt or a windowing library that provides some kind of GetProcAddress function, you also need to tick the loader option.
Add the generated files to your project and compile with them:

src

└── main.c

glad

├── src

│  └── gl.c

├── include

│  ├── glad

│  │  └── gl.h

│  └── KHR

│     └── khrplatform.h



# gcc src/main.c glad/src/gl.c -Iglad/include -lglfw -ldl


in addition to the c++ source show a CMakefile.txt that would configure the project. i created a draft but it needs more work:

note that on my linux development machine i placed the dependencies in ~/src/glfw ~/src/imgui ~/src/glad (though i may have to download a generated glad zip from the website also), ~/src/glm
```