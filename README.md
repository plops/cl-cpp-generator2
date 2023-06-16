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

This project is continually evolving with occasional new features
being added to enhance its functionality. One of the main ongoing
improvements is the reduction of unnecessary parentheses in the
generated expressions. The ideal scenario would be to use an external
tool such as clang-format to address this issue, but no suitable
options have been identified thus far.

One such tool, StyleCop.Analyzers, which is part of the
[StyleCopAnalyzers](https://github.com/DotNetAnalyzers/StyleCopAnalyzers)
project, does a great job of handling these cases, but unfortunately,
it only works for C# and not for our context of C or C++
languages. The use of paid solutions like Clion, despite its
capabilities, remains less preferred due to the cost and the
cumbersome process involved. It's worth mentioning that SonarLint
could potentially serve as an option. Licensed under LGPL, SonarLint
isn't a standalone tool, necessitating operation within an IDE, like
Visual Studio Code.


Recently, exploratory work has been initiated on separating headers
and implementation for C++ classes in a user-friendly manner, which
can be found in more recent examples (usually defined in a file named
util.lisp).

Looking ahead, one of the project's long-term goals is to develop a
comprehensive test suite to ensure the quality and reliability of the
code. However, this is a complex endeavor that requires stable output,
specifically, minimized parentheses and elimination of superfluous
semicolons. At this stage, such stabilization is yet to be achieved,
and the task remains a future goal. The inherent high information
density of the code, as illustrated by the for loop code generator,
adds to the complexity of this effort, making it a challenging yet
exciting future prospect.


```
 (for (destructuring-bind ((start end iter) &rest body) (cdr code)
			 (format nil "for (~@[~a~];~@[~a~];~@[~a~]) ~a"
				 (emit start)
				 (emit end)
				 (emit iter)
				 (emit `(progn ,@body)))))
```

The conditional operator `~@[` of format is used to only print the
start parameter if it is not nil. A thorough test of can require a lot
of cases.

## Avoid redundant parentheses (Work in progress)

I can't just return strings anymore. I will also have to return the
most recent operator, so that I can compare precedence with the
operator that is in the next higher level of the abstract syntax tree.


## History
cl-cpp-generator2 is the tenth in a series of code generators. It
builds on the learnings and experiences from various other projects
like cl-cpp-generator, cl-ada-generator, cl-py-generator, and more.

## License
The project is open-source, free to use, modify, and distribute under
the [MIT License](LICENSE).
