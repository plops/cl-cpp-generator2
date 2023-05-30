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

| short name                                                          | lisp                                                           | C++                                                      |
|---------------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------|
| defun name lambda-list [declaration*] form*                         | (defun foo (a) (declare (type int a) (values int)) (return 2)) | int foo(int a){ return 2;}                               |
| let ({var \vert (var [init-form])}*) declaration* form*"            | (let (a (b 3) (c 3)) (declare (type int a b)) ...              | int a; int b=3; auto c=3;                                |
| setf {pair}*                                                        | (setf a 3 b (+ a 3))                                           | a=3; b=a+3;                                              |
| + {summands}*, /, *,  -                                             | (+ a b c)                                                      | a+b+c                                                    |
| logior {arg}*                                                       | (logior a b)                                                   | a \vert b                                                |
| logand {arg}*                                                       | (logand a b)                                                   | a & b                                                    |
| or {arg}*                                                           | (or a b)                                                       | a \vert \vert b                                          |
| and {arg}*                                                          | (and a b)                                                      | a && b                                                   |
| /= a b, *=, <=, !=, ==, ^=                                          | (/= a b)                                                       | a /= b                                                   |
| <<, >>, <                                                           | (<< a b)                                                       | a << b                                                   |
| incf a [b=1], decf                                                  | (incf a 2)                                                     | a+=2                                                     |
| when                                                                | (when a b)                                                     | if(a) { b; }                                             |
| unless                                                              | (unless a b)                                                   | if(!a) { b; }                                            |
| if                                                                  | (if a (do0 b) (do0 c))                                         | if(a) { b; } else {c;}                                   |
| case                                                                | (case a (b (return 3)) (t (return 4)))                         | switch a ..                                              |
| string                                                              | (string "a")                                                   | "a"                                                      |
| char                                                                | (char "a")                                                     | 'a'                                                      |
| aref                                                                | (aref a 2 3)                                                   | a[2][3]                                                  |
| dot                                                                 | (dot b (f 3))                                                  | b.f(3)                                                   |
| lambda                                                              | (lambda (x) y)                                                 | [&]() { return 3; }                                      |
| defclass  name ({superclass}*) ({slot-specifier}*) [[class-option]] | (defclass Employee (Person) ... TBD                            | class Employee : Person { ...                            |
| for start end iter                                                  | (for ((= a 0) (< a 12) (incf a)) ...)                          | for (a=0; a<12;a++){ ...                                 |
| dotimes i n                                                         | (dotimes (i 12) ...)                                           | for (int i=0; i<12; i++) { ...                           |
| while cond                                                          | (while (== a 1) ...)                                           | while (a==1) { ...                                       |
| foreach item collection                                             | (foreach (a data) ...)                                         | for (auto& a: data) { ...                                |
| deftype name lambda-list {form}*                                    | (deftype csf64 () "complex float")                             | typedef complex float csf64                              |
| defstruct0 name {slot-description}*                                 | (defstruct0 Point (x int) (y int))                             | struct { int x; int y} Point; typedef sruct Point Point; |
| throw                                                               |                                                                |                                                          |
| return                                                              |                                                                |                                                          |
| (uint32_t*) 42                                                      | (cast uint32_t* 42)                                            |                                                          |


## Documentation

In cl-cpp-generator2, several operators can interpret a declare
statement. These include `for-range`, `dotimes`, `let`, `defun`,
`defmethod`, and `lambda`. Similar to Common Lisp, this feature can be
utilized for defining variable types, function parameter types, and
function return values.

### Variable Types

Variables types can be defined using `let` as demonstrated below:

```lisp
(let ((outfile))
  (declare (type "std::ofstream" outfile))
  ...
)
```

### Function Parameter Types

The type of a function parameter can be defined within the function's
declare statement.

```lisp
(defun open (dev)
  (declare (type device& dev))
  ...
)
```

### Function Return Values

Similarly, function return values can be specified using declare:

```lisp
(defun try_release ()
  (declare (values int))
  ...
)
```

### C/C++ Specifics

In addition to the basic uses, cl-cpp-generator2 uses the declare
statement to define C/C++ specifics such as lambda captures,
constructors, constructor initializer lists, and attributes like
`static`, `inline`, `virtual`, `final`, and `override`. This enables
users to have finer control over their code generation.


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
