# cl-cpp-generator2

## Description
The cl-cpp-generator2 is a powerful Lisp package aimed at leveraging
the robust macro system of Common Lisp to enhance languages that bear
similarity to C or C++. This code generator incorporates features that
are reminiscent of Common Lisp conventions, such as type declarations
and operator names. It has also been designed to handle implicit
function calls. This tool strives to provide a bridge between the
worlds of Lisp and C-like languages, enabling users to reap the
benefits of both paradigms in their coding projects.


## Prerequisites
Before you start, ensure you have `quicklisp` installed on your
system in the folder `~/quicklisp`.

## Installation
Follow these steps to install the cl-cpp-generator2 package:

```bash
cd ~
mkdir stage
git clone https://github.com/plops/cl-cpp-generator2
ln -s ~/stage/cl-cpp-generator2 ~/quicklisp/local-projects
```

It is recommended to extract the repo in the `~/stage` directory.

## Getting Started
To start, open `gen01.lisp` in emacs/slime and call the s-expressions
in sequence. Modify the last large s-expression and press C-c C-c to
regenerate all necessary C files.

## Examples
For examples, check the project's example directory:
https://github.com/plops/cl-cpp-generator2/tree/master/example

## FAQs
- **Why doesn't this library generate LLVM?**  
The main interest lies in experimenting with Cuda, OpenCL, Vulkan, and
some Microcontrollers that have C compilers, such as Arduino, Altera
Nios in FPGA, and TI C28x DSP.

## Documentation
In this domain-specific language, I try to follow Common Lisp
conventions as much as possible. However, conditional expressions do
not return a value to keep the C code simpler and more readable. For a
complete list of supported expressions, refer to the documentation.

## Project Status

This project is under active development, with improvements and
features being added occasionally. One of the key areas of focus is
minimizing the number of parentheses in the generated expressions, as
they may currently appear in higher than necessary amounts.

The ideal solution for handling this would be to utilize a separate
tool, such as clang-format, to manage it outside of the
project. However, no suitable tool has been found yet. While Clion
does offer some relevant capabilities, its usage is not free and it
can be cumbersome to execute.

Another area of active experimentation is the separation of headers
and implementation for C++ classes in a user-friendly way. This
exploration is visible in the more recent examples, typically defined
in a file named 'util.lisp'. We encourage users and contributors to
check these examples out for a deeper understanding of the current
development direction.

## History
cl-cpp-generator2 is the tenth in a series of code generators. It
builds on the learnings and experiences from various other projects
like cl-cpp-generator, cl-ada-generator, cl-py-generator, and more.

## License
The project is open-source, free to use, modify, and distribute under
the [MIT License](LICENSE).
