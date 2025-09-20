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

Create a new Common Lisp file  (e.g. `demo.lisp`) that loads the cl-cpp-generator2 package using quicklisp
and define your own package that uses cl-cpp-generator2:

```
(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(defpackage #:my-cpp-project
  (:use #:cl #:cl-cpp-generator2)) 

(in-package #:my-cpp-project)


(format t "~a:~%~a~%"
        *package*
        (cl-cpp-generator2:emit-c :code
          `(do0
             (include <stdio.h>) 
             (defstruct0 struct_a (a int)))))
```

Execute the lisp file: `sbcl --load demo.lisp --quit`. The output will look like this:

```
sbcl --load demo.lisp --quit
```

## Examples
For examples, check the project's example directory:
https://github.com/plops/cl-cpp-generator2/tree/master/example

For a long time I used to work in the `cl-cpp-generator2` package. So most of the examples 
contain gen<n>.lisp file(s)  are still structured in this way.

To start one of the examples, open `gen01.lisp` in emacs/slime and call the s-expressions
in sequence. Modify the last large s-expression and press C-c C-c to
regenerate all necessary C files.

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
| aref                                                                | (aref a 2 3)                                                   | `a[2][3]`                                                  |
| dot                                                                 | (dot b (f 3))                                                  | b.f(3)                                                   |
| lambda                                                              | (lambda (x) y)                                                 | `[&]() { return 3; }`                                      |
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

### List of supported s-expression forms

Here is the list of supported forms:  
   
- comma .. Comma separated list. Example: `(comma 1 2 3)` => `1, 2, 3`  
- semicolon .. Semicolon separated list. Example `(semicolon 1 2 3)` => `1; 2; 3`  
- scope .. Merge a C++ name using scopes. Example: `(scope std vector)` => `std::vector`  
- space .. Merge several objects with space in between. Example: `(space TEST (progn))` => `TEST {}`  
- space-n .. Like `space` but without semicolons. Example: `(space-n "TEST" "XYZ")` => `TEST XYZ`  
- comments .. C++ style comments. Example: `(comments "This is a comment")` => `// This is a comment`  
- lines .. Like comments but without the comment syntax. Example: `(lines "line1" "line2")` => `line1\nline2`  
- doc .. JavaDoc style comments. Example: `(doc "Brief description" "Detailed description")` => `/** Brief description\n * Detailed description\n */`  
- paren* .. Place parentheses only when needed. Example: `(paren* + 5)` => `5`  
- paren .. Parentheses with comma separated values. Example: `(paren 1 2 3)` => `(1, 2, 3)`  
- angle .. Angle brackets with comma separated values. Example: `(angle "typename T" "int N")` => `<typename T, int N>`  
- bracket .. Square brackets with comma separated values. Example: `(bracket 1 2 3)` => `[1, 2, 3]`  
- curly .. Curly braces with comma separated values. Example: `(curly "public:" "void func()")` => `{public: void func()}`  
- designated-initializer .. C designated initializer syntax. Example: `(designated-initializer key1 val1 key2 val2)` => `{.key1 = val1, .key2 = val2}`  
- new .. C++ new operator. Example: `(new int)` => `new int`  
- indent .. Increase indentation. Example: `(indent "code")` => `    code`  
- split-header-and-code .. Split header and code block.  
- do0 .. Execute forms, each in a new line.  
- pragma .. C pragma directive. Example: `(pragma once)` => `#pragma once`  
- include .. C include directive. Example: `(include "myheader.h")` => `#include "myheader.h"`  
- include<> .. C include directive with angle brackets. Example: `(include<> "stdio.h")` => `#include <stdio.h>`  
- progn .. Group a sequence of forms. Example: `(progn (stmt1) (stmt2))` => `{stmt1; stmt2;}`  
- namespace .. C++ namespace definition. Example: `(namespace ns (code))` => `namespace ns {code}`  
- defclass+ .. C++ class definition (force emission of defintion). Example: `(defclass+ name (parent) (code))`  
- defclass .. C++ class definition with only headers (allows to split implementation and declaration). Example: `(defclass name (parent) (code))`  
- protected .. C++ protected section in class. Example: `(protected "void func()")` => `protected: void func();`  
- public .. C++ public section in class. Example: `(public "void func()")` => `public: void func();`  
- defmethod .. C++ class method definition. Example: `(defmethod type "name" (args) (code))`  
- defun .. C++ function definition. Example: `(defun type "name" (args) (code))`  
- return .. C++ return statement. Example: `(return value)` => `return value;`  
- co_return .. C++ coroutine return statement. Example: `(co_return value)` => `co_return value;`  
- co_await .. C++ coroutine await statement. Example: `(co_await expression)` => `co_await expression;`  
- co_yield .. C++ coroutine yield statement. Example: `(co_yield expression)` => `co_yield expression;`  
- throw .. C++ throw statement. Example: `(throw expression)` => `throw expression;`  
- cast .. C++ cast operation. Example: `(cast type value)` => `(type) value`  
- let .. Lisp-like let construct. Example: `(let ((x 5)) (use x))`  
- setf .. Assign values to variables. Example: `(setf x 5)` => `x = 5;`  
- using .. Alias declaration or type alias. Example: `(using alias type)` => `using alias = type;`  
- not .. C++ logical not operation. Example: `(not x)` => `!x`  
- bitwise-not .. C++ bitwise not operation. Example: `(bitwise-not x)` => `~x`  
- deref .. C++ pointer dereference. Example: `(deref ptr)` => `*ptr`  
- ref .. C++ address-of operation. Example: `(ref var)` => `&var`  
- + .. C++ addition operation. Example: `(+ x y)` => `x + y`  
- - .. C++ subtraction operation. Example: `(- x y)` => `x - y`  
- * .. C++ multiplication operation. Example: `(* x y)` => `x * y`  
- ^ .. C++ bitwise XOR operation. Example: `(^ x y)` => `x ^ y`  
- xor .. C++ bitwise XOR operation. Example: `(xor x y)` => `x ^ y`  
- & .. C++ bitwise AND operation. Example: `(& x y)` => `x & y`  
- / .. C++ division operation. Example: `(/ x y)` => `x / y`  
- or .. C++ bitwise OR operation. Example: `(or x y)` => `x | y`  
- and .. C++ bitwise AND operation. Example: `(and x y)` => `x & y`  
- logior .. C++ logical OR operation. Example: `(logior x y)` => `x || y`  
- logand .. C++ logical AND operation. Example: `(logand x y)` => `x && y`  
- = .. C++ assignment operation. Example: `(= x y)` => `x = y`  
- /= .. C++ division assignment operation. Example: `(/= x y)` => `x /= y`  
- *= .. C++ multiplication assignment operation. Example: `(*= x y)` => `x *= y`  
- ^= .. C++ XOR assignment operation. Example: `(^= x y)` => `x ^= y`  
- <=> .. C++ spaceship (three-way comparison) operator. Example: `(<=> x y)` => `x <=> y`  
- <= .. C++ less than or equal to comparison. Example: `(<= x y)` => `x <= y`  
- < .. C++ less than comparison. Example: `(< x y)` => `x < y`  
- != .. C++ not equal to comparison. Example: `(!= x y)` => `x != y`  
- == .. C++ equality comparison. Example: `(== x y)` => `x == y`  
- % .. C++ modulo operation. Example: `(% x y)` => `x % y`  
- << .. C++ left shift operation. Example: `(<< x y)` => `x << y`  
- >> .. C++ right shift operation. Example: `(>> x y)` => `x >> y`  
- incf .. C++ increment operation. Example: `(incf x)` => `x++`  
- decf .. C++ decrement operation. Example: `(decf x)` => `x--`  
- string .. C++ string literal. Example: `(string "hello")` => `"hello"`  
- string-r .. C++ raw string literal. Example: `(string-r "hello")` => `R"(hello)"`  
- string-u8 .. C++ UTF-8 string literal. Example: `(string-u8 "hello")` => `u8"(hello)"`  
- char .. C++ char literal. Example: `(char 'a')` => `'a'`  
- hex .. C++ hexadecimal literal. Example: `(hex 255)` => `0xff`  
- ? .. C++ ternary conditional operator. Example: `( ? x y z)` => `x ? y : z`  
- if .. C++ if statement. Example: `(if condition true-branch false-branch)`  
- when .. C++ if statement without else branch. Example: `(when condition body)`  
- unless .. C++ if not statement. Example: `(unless condition body)`  
- cond .. C++ switch-case structure. Example: `(cond (cond1 body1) (cond2 body2) (t default))`  
- dot .. C++ member access operator. Example: `(dot object member)` => `object.member`  
- aref .. C++ array access operator. Example: `(aref array index)` => `array[index]`  
- -> .. C++ pointer member access operator. Example: `(-> object member)` => `object->member`  
- lambda .. C++ lambda expression. Example: `(lambda (args) (body))`  
- case .. C++ switch-case statement. Example: `(case key (case1 body1) (case2 body2))`  
- for .. C++ for loop. Example: `(for (init cond iter) body)`  
- for-range .. C++ range-based for loop. Example: `(for-range (var range) body)`  
- dotimes .. C++ for loop with fixed iterations. Example: `(dotimes (i n step) body)`  
- foreach .. C++ range-based for loop. Example: `(foreach (item collection) body)`  
- while .. C++ while loop. Example: `(while condition body)`  
- deftype .. C++ typedef statement. Example: `(deftype name (type))`  
- struct .. C++ struct keyword. Example: `(struct name)`  
- defstruct0 .. C++ struct definition without initialization. Example: `(defstruct0 name slots)`  
- handler-case .. C++ try-catch block. Example: `(handler-case body (exception-type handler) ...)`  
   
Some entries like `defun*`, `defun+`, and certain variants of
expressions were omitted. The are variations to separate implementation from declaration.


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

## Efficient Parentheses Management (Under Development)

In this section, I discuss the paren* operator which inspects its
arguments and adds parentheses only when necessary. Alternatively, you
can use the paren operator to enforce the inclusion of parentheses,
leading to potential redundancy but ensuring that the syntax tree,
constructed from the s-expression input, is precisely mirrored by the
resulting C++ output string.


Implementing the paren* operator means we can't just return strings
anymore. Instead, we must return the most recent operator so that we
can evaluate its precedence against the operator at the next higher
level in the abstract syntax tree. To facilitate this, I have defined
a string-op class in c.lisp. Along with the helper functions m
<operator> <string> and m-of <string-op>, the string-op class is used
throughout c.lisp to represent both the string and the current
operator.

Testing for the paren* operator has begun in the 't/01_paren'
directory. Here, I establish an s-expression, the expected C++ string,
and a Common Lisp function that yields the same value. Each test
generates a C++ file to confirm that the C++ code, derived from the
s-expression, matches the result of the Common Lisp code. 

Additionally, we draw a comparison between code that employs the newly
introduced paren* operator, which eliminates superfluous parentheses,
and code that still includes the full set of parentheses.


As a general guideline, if given a choice, I lean towards
over-employing parentheses. For instance, I prefer
(3+4)*(13/4)*((171+2)/5) over (3+4)*13/4*(171+2)/5.

However, it is crucial to properly handle complex cases such as
(static_cast<int>(((1) & ((v) >> (1))))).



A lingering question is whether paren* needs to know if the operators
are positioned to the left, right, or both sides of the current
element. In response to this, ChatGPT-4 says:

    You generally only need to know the precedence of each operator to
    correctly order the operations, and you only need to know the
    associativity of an operator when dealing with multiple instances of
    the same operator.

Despite this, I encountered a contradiction in the Lisp expression (-
(+ 3 4) (- 7 3)), which translates to the infix expression 3+4-7-3 but
should ideally be 3+4-(7-3).

ChatGPT-4 provided this explanation:

    In this case, we have the subtraction operator -, which is left
    associative, being used twice. The right operand of the first
    subtraction is another subtraction. Even though the - operator is left
    associative, the right operand needs to be parenthesized to correctly
    represent the original expression. This is because the subtraction in
    the right operand should be evaluated before the first subtraction.

Although this is insightful, I am uncertain of the implications. It
may suggest the need to modify the precedence table, differentiating
between the - and + operators.

 

## Remarks 2023-12-30

I started reading Stroustrup: A Tour of C++. In this section I note
some ideas how the Lisp code could be improved.

### Initialization Preferences

- Bjarne Stroustrup, the creator of C++, recommends using braces `{}`
  for initialization over the equals sign `=`. This approach, known as
  brace initialization, offers advantages such as uniform syntax and
  prevention of narrowing conversions (if the type is not auto).

### `consteval` Functions

- Functions declared as `consteval` must be evaluated at compile time
  and thus should be defined in a way that makes them accessible to
  all relevant translation units. In traditional C++ with header
  files, this means declaring `consteval` methods in the header. In
  the context of C++20 modules, `consteval` functions are defined
  within a module interface unit, which functions similarly to a
  header but is integrated into the module system.

### Variable Introduction in Control Structures

- C++ allows the introduction of a variable within the condition of
  control structures like `for` and `if`. This feature enables both
  the declaration of a variable and its conditional evaluation in a
  single statement. For example:
  
  ```cpp
  if (auto n = v.size(); n != 0) { ...
  ```

#### Mapping to Lisp

- When considering how to map this C++ feature to Lisp, a
  straightforward approach would be using `if` with `let`:
  
  ```lisp
  (if (let ((n (v.size))) (!= n 0)) ...)
  ```
- However, this Lisp translation might only necessitate one comparison
  and one semicolon. Given its infrequent occurrence in the code,
  implementing a direct equivalent may not be sufficiently beneficial
  to justify the effort.


## History
cl-cpp-generator2 is the tenth in a series of code generators. It
builds on the learnings and experiences from various other projects
like cl-cpp-generator, cl-ada-generator, cl-py-generator, and more.

## License
The project is open-source, free to use, modify, and distribute under
the [MIT License](LICENSE).
