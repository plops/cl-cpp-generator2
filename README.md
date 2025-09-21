# cl-cpp-generator2
A Common Lisp DSL (domain specific language) and code generator for producing readable C/C++ (and CUDA/OpenCL) from s-expressions.    

## Quick summary  
cl-cpp-generator2 is a Lisp package that maps Common Lisp s-expressions to C/C++ code. It exposes a compact DSL (operators like defun, let, setf, for-range, paren*, include<>, etc.) that you write in Lisp; the package emits corresponding C/C++ source that you can write to disk. It is built to be used interactively in Common Lisp (SBCL recommended) and can be used to generate code for desktop C++ projects, embedded toolchains, CUDA/OpenCL, or to experiment with language transformations.  
  
## Overview 
cl-cpp-generator2 lets you express C/C++ constructs as Common Lisp s-expressions and then emit human-readable C/C++ source. The motivation is to use Lisp's macros and syntax manipulation strengths to generate C/C++ program skeletons, helper code, or host-side code for GPU/embedded targets (CUDA, OpenCL, microcontroller C). It emphasizes readability of the generated C++ while allowing you to annotate types, constructors, lambda captures, and function attributes through Lisp declare forms.  
  


## Who is this for?
- Lisp developers who want to produce C/C++ code programmatically.  
- Engineers generating host code for GPU kernels (CUDA/OpenCL) and embedded C targets.  
- People exploring language-design, code-generation, or macro-based code transformation.  

## Key features  
- Declarative mapping of Lisp s-exprs to C/C++ constructs (functions, classes, structs, loops, control flow).  
- Support for using Lisp declare forms for variable types, function parameter types, return types, lambda captures, constructors, and attributes (static, inline, virtual, noexcept, override, final).  
- Mechanisms to separate headers and implementations (split-header-and-code, defclass / defclass+).  
- Optional emission formatting using clang-format and clang-tidy (SBCL convenience wrappers included).  
- paren* operator — attempts to avoid redundant parentheses by inspecting operator precedence (work in progress).  

## Prerequisites  
- quicklisp installed (recommended location: ~/quicklisp)  
- SBCL (Steel Bank Common Lisp) — the project is developed and tested primarily with SBCL.  
- Optional: clang-format and clang-tidy for automatic formatting/fixes when using write-source (write-source calls them via sb-ext:run-program — SBCL only).  
- cl-unicode required (installable via quicklisp).  
  
Install cl-unicode:  
```bash  
sbcl --eval "(ql:quickload :cl-unicode)" --quit  
```  
  
Notes:  
- Some helper code uses SBCL-specific facilities (sb-ext:run-program). You can disable formatting/tidy in write-source if you use another Lisp.  
- If clang-format/clang-tidy aren't available, either install them or call write-source with :format nil and :tidy nil.  
  
## Installation  
Recommended workflow: extract the repo into a staging area and link it into Quicklisp's local projects.  
  
In your shell:  
```bash  
cd ~  
mkdir -p stage  
cd stage  
git clone https://github.com/plops/cl-cpp-generator2  
ln -s ~/stage/cl-cpp-generator2 ~/quicklisp/local-projects  
```  
  
Load the system in Lisp:  
```lisp  
(load "~/quicklisp/setup.lisp")  
(ql:quickload "cl-cpp-generator2")  
```  

## Quick Start
This minimal example shows how to load the package and emit a small C++ program from s-expressions.  
  
1. Create a `demo.lisp` file:  
  
```lisp  
;; demo.lisp  
(load "~/quicklisp/setup.lisp")  
(eval-when (:compile-toplevel :execute :load-toplevel)  
  (ql:quickload "cl-cpp-generator2"))  
  
(defpackage #:my-cpp-project  
  (:use #:cl #:cl-cpp-generator2))  
(in-package #:my-cpp-project)  
  
(format t "~a~%" (cl-cpp-generator2:emit-c :code  
  `(do0  
     (include<> "iostream")  
     (defun main (argc argv)  
       (declare (type int argc) (type char** argv) (values int))  
       (return 0)))))  
```  
  
2. Run it with SBCL:  
```bash  
sbcl --load demo.lisp --quit  
```  
  
You will see the generated C++ printed to stdout (and you can use `write-source` to write files).

```
$ sbcl --load demo.lisp --quit  
This is SBCL 2.5.4, an implementation of ANSI Common Lisp.
[..]
To load "cl-cpp-generator2":
  Load 1 ASDF system:
    cl-cpp-generator2
; Loading "cl-cpp-generator2"
.................
#include <iostream> 

int main (int argc, char** argv)        {
        return 0;
}

```

## How it works — at a glance 
- You describe C/C++ structures as Lisp s-expressions using the provided operators.  
- `emit-c` traverses these s-expressions and produces a string (or a string-op object that remembers operator precedence).  
- `write-source` wraps emit-c, optionally formats the output with clang-format and clang-tidy, and writes files to disk. It also avoids rewriting files when content is unchanged (using sxhash).  
- You can embed type information and other C++ nuances in Lisp using `declare` forms inside `defun`, `lambda`, `let`, `dotimes`, `for-range`, etc. The helper `consume-declare` parses this declare info (types, values, capture, construct, and attributes).  
- The DSL exposes both low-level operations (paren, angle, bracket, curly, cast, aref, dot, ->) and high-level constructs (defun, defmethod, defclass, defstruct0, for-range).  
  
## Reading the command reference
The operator reference below lists Lisp operator names (left column) and shows how typical Lisp forms are translated to C/C++. Each mapping is a shorthand — the DSL is richer and supports many combinations and options.  
  
How to interpret the operator examples:  
- Lisp forms are shown as s-expressions: (operator args...)  
- Right column shows typical C/C++ output that operator produces.  
- Special forms like `defun`, `defmethod`, `defclass`, and `defstruct0` use declare statements to determine types and attributes.  
- `paren*` is an intelligent parenthesizer that attempts to add parentheses only when needed — precedence-aware (still under active development).  
  
If you are new: start with the `do0` wrapper to sequence statements; use `include<>` or `include` for headers; define functions with `defun` and specify types with (declare (type ...)) and return types with (declare (values ...)).  
  
## List of supported s-expression forms  
(Short introductory text: this is a compact feature reference. After the short list we include more detailed forms and notes.)  
  
- comma .. Comma separated list. Example: (comma 1 2 3) => 1, 2, 3    
- semicolon .. Semicolon separated list. Example (semicolon 1 2 3) => 1; 2; 3    
- scope .. Merge C++ scopes. Example: (scope std vector) => std::vector    
- space .. Merge several objects with space in between. Example: (space TEST (progn)) => TEST {}    
- space-n .. Like space but without semicolons. Example: (space-n "TEST" "XYZ") => TEST XYZ    
- comments .. C++ style comments. Example: (comments "This is a comment") => // This is a comment    
- lines .. Like comments but without the comment syntax. Example: (lines "line1" "line2") => line1\nline2    
- doc .. JavaDoc style comments. Example: (doc "Brief" "Detailed") => /** Brief \n * Detailed \n */    
- paren* .. Add parentheses only when necessary (precedence-aware). Example: (paren* + 5) => 5    
- paren .. Parentheses with comma separated values. Example: (paren 1 2 3) => (1, 2, 3)    
- angle .. Angle brackets with comma separated values. Example: (angle "typename T" "int N") => <typename T, int N>    
- bracket .. Square brackets. Example: (bracket 1 2 3) => [1, 2, 3]    
- curly .. Curly braces with comma separated values. Example: (curly "public:" "void func()") => { public: void func(); }    
- designated-initializer .. C designated initializer. Example: (designated-initializer key1 val1 key2 val2) => {.key1 = val1, .key2 = val2}    
- new .. C++ new. Example: (new int) => new int    
- indent .. Increase indentation for nested code generation. Example: (indent "code") => "    code"    
- split-header-and-code .. Split header and code block emission (hooks)    
- do0 .. Sequence statements; each statement on its own line.    
- pragma .. C pragma directive. Example: (pragma once) => #pragma once    
- include .. #include "myheader.h" (string form)    
- include<> .. #include <stdio.h> (angle-bracket include)    
- progn .. Block expression grouped with braces. Example: (progn (stmt1) (stmt2)) => {stmt1; stmt2;}    
- namespace .. C++ namespace definition. Example: (namespace ns (code)) => namespace ns { code }    
- defclass+ .. Force emission of class definition (header+implementation)    
- defclass .. Class definition (supports producing headers + separating implementations)    
- protected / public .. C++ class visibility specifiers. Example: (protected "void func()") => protected: void func();    
- defmethod / defun .. C++ method and function definitions (support many declare options)    
- return / co_return / co_await / co_yield / throw .. Corresponding C++ statements    
- cast .. C style cast. Example: (cast type value) => (type) value    
- let .. Lisp let -> variable declarations (uses types from declare if present)    
- setf .. Assignments. Example: (setf x 5) => x = 5;    
- using .. Type alias. Example: (using alias type) => using alias = type;    
- not / bitwise-not / deref / ref .. Unary operators: ! ~ * &    
- `+ - * / ^ & | << >>` (and many more arithmetic/bitwise operators)    
- logior / logand .. logical || and &&    
- incf / decf .. increments/in-decrements (a++ / a-- or a += n)    
- string, string-r, string-u8, char, hex .. literal forms    
- ? .. ternary operator: ( ? cond then else ) => cond ? then : else    
- if / when / unless / cond / case .. control flow -> if/else / switch / etc.    
- dot / aref / -> .. member/array access: object.member, object[index], object->member    
- lambda .. C++ lambda expression, supports capture via (declare (capture ...))    
- for / for-range / dotimes / foreach / while .. loop constructs    
- deftype .. typedef mapping    
- struct / defstruct0 .. struct generation    
- handler-case .. try-catch mapping  

(For a full, exhaustive operator list and examples, see the detailed reference below or the implementation file c.lisp in the repository. The DSL supports many combinations and optional declare attributes such as const, inline, static, virtual, noexcept, final, override, pure, template, and template-instance.)  

## Declare statements, types, and attributes  
cl-cpp-generator2 supports parsing declare statements attached to function or block forms. These declare forms are consumed early (using `consume-declare`) and can supply:  
  
- Variable/parameter types: (declare (type int a b)) — used by let, defun, lambda, dotimes, for-range, etc.  
- Function return types: (declare (values int)) — used by defun/defmethod.  
- Lambda captures: (declare (capture x y)) — used by lambda forms; placed into the capture brackets.  
- Constructor initializers: (declare (construct ...)) — used in defmethods/constructors.  
- Function/spec attributes: (declare inline static virtual noexcept final override pure explicit const template ...) — used when emitting function or method signatures.  
  
Examples:  
- Variable type in let:  
```lisp  
(let ((outfile))  
  (declare (type "std::ofstream" outfile))  
  ...)  
```  
- Function parameter type:  
```lisp  
(defun open (dev)  
  (declare (type device& dev))  
  ...)  
```  
- Function return type:  
```lisp  
(defun try_release ()  
  (declare (values int))  
  ...)  
```

## Examples & Running examples  
See `example/` in the repository. Many examples are organized as `gen<n>.lisp` scripts; the generation process is typically interactive:  
  
- Open an example `gen01.lisp` in Emacs/SLIME and evaluate expressions or the final big s-expression sequence to regenerate C++ files.  
 
The examples directory contains more complex examples (GLFW, rendering, etc.)—look at `example/162_glfwpp_grating` for a non-trivial example that uses many parts of the DSL.  

## write-source and formatting  
`write-source` is a convenience wrapper that:  
- Calls `emit-c` with your code  
- Computes a hash and only writes the file if the new content differs  
- Optionally runs `clang-format -i` and `clang-tidy --fix`  
  
Example:  
```lisp  
(write-source "main.cpp"  
  `(do0  
     (include<> "iostream")  
     (defun main (argc argv)  
       (declare (values int))  
       (return 0)))  
  :dir #P"/tmp/"  
  :format t      ;; run clang-format  
  :tidy nil)     ;; disable clang-tidy if you want  
```  
  
If you do not want SBCL to run external programs, set `:format nil` and `:tidy nil`, or edit the code to remove sb-ext calls (if you use another Lisp implementation).  
  
## Project status & development notes  
- The project is actively evolving. Major ongoing work centers on improving the `paren*` operator and reducing redundant parentheses in emitted code.  
- Progress has been made toward splitting headers and implementation for C++ classes via `defclass` / `defclass+` and helper utilities in later examples.  
- The author plans to add a formal test-suite in the future. Stable, minimal output (reduced parentheses and redundant semicolons) will be important for that effort.  
  
Paren* and precedence:  
- A precedence table is included and `paren*` attempts to add parentheses only when necessary by comparing operator precedence between parent and child nodes.  
- This is an intricate area — some edge cases may remain. The design favors safety (more parentheses) where ambiguity exists.
- If clang-format ever supports parentheses reduction, it could be a better solution.

## Limitations & Known issues  
- SBCL-specific helper calls (sb-ext:run-program) are used; those can be disabled but SBCL is the primary tested environment.  
- Parentheses elimination (`paren*`) is still under development and may not always produce ideal results. The generator errs on the side of safety.  
- Some variations and convenience forms (defun*, defun+, defmethod* etc.) exist or were omitted in the short reference — consult `c.lisp` for details.  
- The emitted code is not guaranteed to be perfectly canonical C++ style; using clang-format and clang-tidy helps improve style, but exact spacing/paren consistency may vary.  

## Troubleshooting & practical tips 
- If write-source fails because clang-format or clang-tidy are missing, run write-source with `:format nil :tidy nil` or install those tools.  
- If you use a Lisp other than SBCL, search for `sb-ext:run-program` calls in `write-source` and remove or adapt them.  
- When experimenting interactively, prefer `emit-c` first to inspect strings before writing files with `write-source`.  
- If types are missing (emit functions complaining about unknown types), add `(declare (type ...))` in the surrounding `defun`/`let` forms — the generator uses `consume-declare` to collect types.  
  
## Contributing & Contact
Contributions are welcome. If you want to:  
- Report bugs: open an issue on the repository with a minimal reproduction.  
- Contribute examples or patches: fork, create a branch, and submit a pull request.  
- Discuss design changes or precedence details: open an issue or contact the maintainer.  
  
Maintainer: <kielhorn.martin@gmail.com>  
  
## FAQs
- **Why doesn't this library generate LLVM?**
The main interest lies in experimenting with Cuda, OpenCL, Vulkan, and
some Microcontrollers that have C compilers, such as Arduino, Altera
Nios in FPGA, and TI C28x DSP.

# Reference

Below is a categorized operator reference table for cl-cpp-generator2. Each row shows the DSL operator, a short purpose statement, a compact Lisp s-expression example, and the typical emitted C/C++ output. Use this as a quick lookup; many operators accept variants or additional declare options — see c.lisp for full details or search in examples/ for use cases.  
  
Note: Lisp examples are s-expressions you pass to emit-c; C/C++ output is a representative snippet (not always full expanded code).  
  
---  
  
## How to read the table  
- Operator: cl-cpp-generator2 DSL operator name.  
- Purpose: short explanation.  
- Lisp example: canonical s-expression usage.  
- C/C++ output: the typical or representative emitted code.  
  
---  
  
### Basic separators & joining  
| Operator | Purpose | Lisp example | C/C++ output |  
|---|---|---|---|  
| `comma` | Join values with commas | `(comma 1 2 3)` | `1, 2, 3` |  
| `semicolon` | Join values with semicolons | `(semicolon a b c)` | `a; b; c` |  
| `scope` | Join with C++ scope `::` | `(scope std vector)` | `std::vector` |  
| `space` | Join tokens with spaces (and keep semicolons when used in statements) | `(space TEST (progn))` | `TEST {}` |  
| `space-n` | Join tokens with spaces (no semicolon handling) | `(space-n "T" "U")` | `T U` |  
  
---  
  
### Brackets / grouping  
| Operator | Purpose | Lisp example | C/C++ output |  
|---|---|---:|---|  
| `paren` | Parentheses with comma-separated items | `(paren 1 2 3)` | `(1, 2, 3)` |  
| `paren*` | Precedence-aware parentheses (add only if needed) | `(paren* + ( * 1 2 ))` | `1 * 2` (or parenthesized if needed) |  
| `angle` | Angle brackets, comma-separated | `(angle "typename T" "int N")` | `<typename T, int N>` |  
| `bracket` | Square brackets | `(bracket i j)` | `[i, j]` |  
| `curly` | Curly braces | `(curly "public:" "void f()")` | `{ public: void f(); }` |  
| `designated-initializer` | C designated initializer | `(designated-initializer Width w Height h)` | `{ .Width = w, .Height = h }` |  
  
---  
  
### Strings, chars, numbers  
| Operator | Purpose | Lisp example | C/C++ output |  
|---|---|---|---|  
| `string` | Quoted C++ string literal | `(string "hello")` | `"hello"` |  
| `string-r` | Raw string literal | `(string-r "x\n)")` | `R"(x\n))"` |  
| `string-u8` | UTF-8 string literal | `(string-u8 "hi")` | `u8"hi"` |  
| `char` | Character literal | `(char "a")` | `'a'` |  
| `hex` | Hex literal | `(hex 255)` | `0xff` |  
  
---  
  
### Comments, docs, preprocessor, includes  
| Operator | Purpose | Lisp example | C/C++ output |  
|---|---|---|---|  
| `comments` | C++ `//` comments | `(comments "note")` | `// note` |  
| `lines` | Plain lines (no `//`) | `(lines "line1" "line2")` | `line1` <br> `line2` |  
| `doc` | JavaDoc/Doxygen block comment | `(doc "Brief" "Detail")` | `/** Brief\n * Detail\n */` |  
| `pragma` | Preprocessor pragma | `(pragma once)` | `#pragma once` |  
| `include` | `#include "file"` | `(include "my.h")` | `#include "my.h"` |  
| `include<>` | `#include <file>` | `(include<> "stdio.h")` | `#include <stdio.h>` |  
  
---  
  
### Statements, blocks & indentation  
| Operator | Purpose | Lisp example | C/C++ output |  
|---|---|---|---|  
| `do0` | Sequence of statements (each on its own line, semicolons handled) | `(do0 (setf x 1) (setf y 2))` | `x = 1;` <br> `y = 2;` |  
| `progn` | Block grouped with braces | `(progn (stmt1) (stmt2))` | `{ stmt1; stmt2; }` |  
| `do` | Like `do0` but used for inner grouped forms | `(do (stmt1) (stmt2))` | `stmt1` on separate lines (no extra braces) |  
| `indent` | Increase indentation for nested pieces | `(indent (do0 (stmt)))` | `    stmt` |  
  
---  
  
### Includes, namespace & header split  
| Operator | Purpose | Lisp example | C/C++ output / note |  
|---|---|---|---|  
| `namespace` | Emit C++ namespace block | `(namespace myns (do0 ...))` | `namespace myns { ... }` |  
| `split-header-and-code` | Emit header form and implementation separately (hookable) | `(split-header-and-code header code)` | (Caller-supplied hook may write header and code into different files) |  
  
---  
  
### Types, typedefs, structs, classes  
| Operator | Purpose | Lisp example | C/C++ output |  
|---|---|---|---|  
| `deftype` | Typedef | `(deftype csf64 () "complex float")` | `typedef complex float csf64;` |  
| `struct` | Bare struct name | `(struct Point)` | `struct Point` |  
| `defstruct0` | Define a struct with slots | `(defstruct0 Point (x int) (y int))` | `struct Point { int x; int y; };` |  
| `defclass` | Class declaration (header-only option available) | `(defclass (My<T>) (Base) (public: (defmethod foo (a) ...)))` | `template<...> class My : Base { public: int foo(int a); };` |  
| `defclass+` | Class with full method implementations inline | `(defclass+ My (Base) (public: (defmethod foo (a) (return a))))` | `class My : Base { public: int foo(int a) { return a; } };` |  
| `public` / `protected` | Class visibility sections | `(public "void f()")` | `public: void f();` |  
  
---  
  
### Functions & methods  
| Operator | Purpose | Lisp example | C/C++ output |  
|---|---|---|---|  
| `defun` | Define a free function | `(defun foo (a) (declare (type int a) (values int)) (return (+ a 2)))` | `int foo(int a) { return a + 2; }` |  
| `defun*` | Alternate variant (e.g. declaration-only shorthand) | `(defun* foo (a) (declare (type int a) (values int)))` | e.g., header-only declaration `int foo(int a);` |  
| `defun+` | Alternate variant (force emit implementation) | `(defun+ foo (a) ...)` | force implementation inlined in header/class |  
| `defmethod` | Class method (emit as member or out-of-class definition) | inside class: `(defmethod foo (a) (declare (type int a) (values int)) (return a))` | `int Class::foo(int a) { return a; }` |  
| `return` | Return statement | `(return 0)` | `return 0;` |  
| `co_return` / `co_await` / `co_yield` | Coroutine statements | `(co_return val)` | `co_return val;` |  
| `throw` | Throw exception | `(throw "e")` | `throw "e";` |  
| `cast` | C-style cast | `(cast int x)` | `(int) x` |  
  
Notes: use `(declare (values ...))` in function bodies to specify return types; `(declare (type ...))` for parameter types.  
  
---  
  
### `let`, `setf`, `using` — variable & alias declarations  
| Operator | Purpose | Lisp example | C/C++ output |  
|---|---|---|---|  
| `let` | Declare variables, with optional initializers (uses `declare` types or `auto`) | `(let ((a 5) (b (std--vector<int> (curly 1 2)))) (use a b))` | `auto a = 5; auto b{std::vector<int>{1, 2}}; use(a, b);` |  
| `letc` / `letd` | `letc` => write `const` prefix; `letd` => use `decltype` for declarations | `(letc ((x 5)))` | `const auto x{5};` |  
| `setf` | Assignment(s) | `(setf a 3 b (+ a 3))` | `a = 3; b = a + 3;` |  
| `using` | Type alias / `using` declaration | `(using Vec std::vector<int>)` | `using Vec = std::vector<int>;` |  
  
---  
  
### Control flow: conditional & multi-branch  
| Operator | Purpose | Lisp example | C/C++ output |  
|---|---|---|---|  
| `if` | `if` with optional `else` | `(if cond (do0 then) (do0 else))` | `if (cond) { then } else { else }` |  
| `if-constexpr` | `if constexpr` (C++20) | `(if-constexpr cond (do0 a) (do0 b))` | `if constexpr (cond) { a } else { b }` |  
| `when` | `if (cond) { body; }` | `(when cond (do0 body))` | `if (cond) { body; }` |  
| `unless` | `if (!cond) { body; }` | `(unless cond (do0 body))` | `if (!cond) { body; }` |  
| `cond` | Multi-branch conditional (emits `if/else if` chains) | `(cond (c1 e1) (c2 e2) (t default))` | `if (c1) { e1 } else if (c2) { e2 } else { default }` |  
| `case` | Switch-case | `(case x (1 (do0 ...)) (t (do0 ...)))` | `switch (x) { case 1: ...; default: ... }` |  
| `handler-case` | `try`/`catch` mapping | `(handler-case (progn body) (int (e) (do0 ...)) (t () (do0 ...)))` | `try { body } catch (int e) { ... } catch (...) { ... }` |  
  
---  
  
### Loops & iteration  
| Operator | Purpose | Lisp example | C/C++ output |  
|---|---|---|---|  
| `for` | Classic C for loop | `(for ((= i 0) (< i n) (incf i)) (do0 ...))` | `for (i = 0; i < n; i += 1) { ... }` |  
| `for-range` | Range-based for | `(for-range ((x vec) (do0 ...)))` | `for (auto&& x : vec) { ... }` |  
| `dotimes` | Fixed iteration count | `(dotimes (i n) (do0 ...))` | `for (int i = 0; i < n; ++i) { ... }` |  
| `foreach` | C++ range-for alias | `(foreach (a vec) (do0 ...))` | `for (auto& a : vec) { ... }` |  
| `while` | While loop | `(while cond (do0 ...))` | `while (cond) { ... }` |  
  
---  
  
### Member access & arrays  
| Operator | Purpose | Lisp example | C/C++ output |  
|---|---|---|---|  
| `dot` | Member/accessor call or chained member access | `(dot obj member)` or `(dot obj (f 3))` | `obj.member` or `obj.f(3)` |  
| `->` | Pointer member access | `(-> obj member)` | `obj->member` |  
| `aref` | Array indexing (multi-dimensional possible) | `(aref arr 2 3)` | `arr[2][3]` |  
  
---  
  
### Lambdas & captures  
| Operator | Purpose | Lisp example | C/C++ output |  
|---|---|---|---|  
| `lambda` | C++ lambda with capture list (declare capture) | `(lambda (x) (declare (capture y) (type int x) (values int)) (return (+ x y)))` | `[&y](int x) -> int { return x + y; }` (capture & signature derived from declares) |  
  
---  
  
### Binary/unary operators (arithmetic / bitwise / logical / assignment)  
| Operator | Purpose | Lisp example | C/C++ output |  
|---|---|---|---|  
| `+` | Addition (multiple arguments) | `(+ a b c)` | `a + b + c` |  
| `-` | Subtraction or unary negation | `(- a b)` / `(- a)` | `a - b` / `-a` |  
| `*` | Multiplication | `(* a b)` | `a * b` |  
| `/` | Division | `(/ a b)` | `a / b` |  
| `%` | Modulo | `(% a b)` | `a % b` |  
| `<<` / `>>` | Shift left/right | `(<< a b)` | `a << b` |  
| `&` | Bitwise AND | `(& a b)` | `a & b` |  
| `|` / `or` | Bitwise OR (`or` is the DSL operator) | `(or a b)` | `a | b` |  
| `^` / `xor` | Bitwise XOR | `(^ a b)` or `(xor a b)` | `a ^ b` |  
| `logior` | Logical OR (||) | `(logior a b)` | `a || b` |  
| `logand` | Logical AND (&&) | `(logand a b)` | `a && b` |  
| `==` | Equality | `(== a b)` | `a == b` |  
| `!=` | Inequality | `(!= a b)` | `a != b` |  
| `<` / `<=` | Comparison (also supports three-arg chaining) | `(< a b)` / `(<= a b c)` | `a < b` / `a <= b && b <= c` |  
| `<=>` | Three-way comparison (C++20) | `(<=> a b)` | `a <=> b` |  
| `=` | Assignment expression (used inside expressions e.g., `(= x y)`) | `(= a 3)` or in do0 `((= a 3) ...)` | `a = 3` |  
| `/=` `*=` `^=` `<<=` `>>=` `&=` `|=` | Compound assignments | `(/= a b)` | `a /= b` |  
| `incf` / `decf` | Increment / decrement or add/subtract n | `(incf a)` / `(incf a 2)` | `a++` / `a += 2` |  
| `setf` | Multiple assignments (see above) | `(setf x 1 y 2)` | `x = 1; y = 2;` |  
  
---  
  
### Unary / pointer operators  
| Operator | Purpose | Lisp example | C/C++ output |  
|---|---|---|---|  
| `not` | Logical not | `(not x)` | `!x` |  
| `bitwise-not` | Bitwise not | `(bitwise-not x)` | `~x` |  
| `deref` | Pointer dereference | `(deref p)` | `*p` |  
| `ref` | Address-of | `(ref x)` | `&x` |  
| `new` | C++ new operator | `(new int)` | `new int` |  
  
---  
  
### Misc / advanced / helpers  
| Operator | Purpose | Lisp example | C/C++ output / note |  
|---|---|---|---|  
| `split-header-and-code` | Emit header and code separately via hook | `(split-header-and-code header code)` | Hooked emission of header and implementation files |  
| `using` | C++ using alias (see above) | `(using V std::vector<int>)` | `using V = std::vector<int>;` |  
| `handler-case` | Exception handling mapping | `(handler-case expr (Type (e) body) (t () default))` | `try { expr } catch(Type e) { ... } catch(...) { ... }` |  
  


### C/C++ Specifics

In addition to the basic uses, cl-cpp-generator2 uses the declare
statement to define C/C++ specifics such as lambda captures,
constructors, constructor initializer lists, and attributes like
`static`, `inline`, `virtual`, `final`, and `override`. This enables
users to have finer control over their code generation.


## Design Philosophy and Trade-offs

This section clarifies some of the design choices and trade-offs made in `cl-cpp-generator2` that might not be immediately obvious.

### Variable Declaration and Initialization

*   **Almost-Always-Auto:** The generator encourages an "almost-always-auto" style. You can introduce variables with `(let ((a (type ...))) ...)` without needing to specify the C++ type in the `let` binding itself. The type is inferred from the `declare` form, and if no type is declared, it defaults to `auto`.
*   **Brace Initialization `{}`:** C++ initialization with `{}` is preferred over `=`. This is to leverage more consistent syntax and avoid certain types of bugs, like narrowing conversions. However, be aware that C++ can have surprising behavior with brace initialization for `std::array` or `std::vector`, which has been a source of bugs.

### C++ Naming in Lisp

*   **Inverted Readtable:** To allow writing C++ identifiers (like `myVariable` or `MyClass`) directly as Common Lisp symbols, the readtable case is set to `:invert`. This is highly convenient but can have unforeseen side effects in complex Lisp environments.
*   **Handling Scoped Names (`::`):** The Common Lisp reader is confused by the colons in C++ scoped names like `std::cout`. To work around this, `cl-cpp-generator2` replaces all double-minus sequences (`--`) in symbols with `::` during code emission. This allows you to write `std--cout` in Lisp to generate `std::cout` in C++.
*   **Templates as Strings:** While you *can* write complex template types as s-expressions, e.g., `(space std--array (angle float 4))`, it is often more convenient and readable to provide them as a simple string: `"std::array<float,4>"`. The s-expression form is primarily useful when you need to programmatically generate types using Lisp macros, as shown in this example that creates several arrays of different sizes:

    ```lisp
     ,@(loop for i from 1 upto 3
    		collect
    		`(let ((,(format nil "a~a" i)
    			 (space std--array (angle float ,i))))))
    ```
    This generates:
    ```cpp
      auto a1{std::array<float, 1>};
      auto a2{std::array<float, 2>};
      auto a3{std::array<float, 3>};
    ```

### Semicolons and Parentheses: Heuristics and Complexity

*   **Semicolons:** A significant amount of complexity in the generator's source code comes from a heuristic designed to place semicolons correctly without requiring them explicitly in the Lisp code. A list of DSL operators that should not be followed by a semicolon is maintained, which works well but complicates the implementation compared to a language like Python that doesn't have this requirement.
*   **Parentheses:** A simple translation of Lisp forms to C++ would result in an excessive number of parentheses, making the code hard to read. A heuristic based on C++ operator precedence rules is used to remove unnecessary parentheses. However, this is a delicate balance. The author prefers to keep some "redundant" parentheses in complex comparisons (e.g., `if ((3==a) && (b == 7) || (c != (d << 3)))`) for clarity, as remembering operator precedence for things like `<<` can be difficult. The parenthesis-removal logic adds complexity and would ideally be replaced by a tool like `clang-format` if it supported such a feature (similar to `ruff` for Python). The current heuristic is not guaranteed to be perfect and would benefit from a comprehensive test suite.

### `defclass` and Code Generation

The examples often show a `utils.lisp` file with a `write-class` helper that can emit both a header (`.hpp`) and an implementation (`.cpp`) file from a single `defclass` expression. While this works for many cases, creating a truly general and robust solution has proven difficult. The challenge lies in handling the numerous prefixes and suffixes that can surround a class or method definition (`[[nodiscard]]`, `const`, `template<T>`, namespaces like `Bar<T>::`, etc.), which often differ between the header and the implementation. This makes it challenging to create a universal abstraction that covers all corner cases.

### Focus on Modern C++

The codebase used to contain a `*feature*` flag called `:generic-c` to handle differences between C and C++ standards (e.g., `___auto_type` in C vs. `auto` in C++). This feature has been deprecated to reduce complexity and focus the project on modern C++ (specifically C++17 and C++20). Similarly, special handling for corner cases in other C-like languages, such as the OpenGL Shading Language (GLSL), which has limitations on brace initialization and `auto`, is no longer a primary focus.


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
cl-cpp-generator2 is the tenth generator in a family of similar tools (cl-cpp-generator, cl-ada-generator, cl-py-generator, ...). The project reflects years of experimentation with code generation and macro-driven DSLs in Common Lisp.  
  

## License  
The project is open-source under the MIT license (see LICENSE). A few examples in the repository may be provided under GPLv3 — check example-specific LICENSE files where present.  


## Where to look in the codebase  
- package.lisp — package definition and exported symbols  
- c.lisp — the main emitter and DSL implementation (the authoritative list of supported forms and parser code)  
- example/ — many usage examples (start with gen01.lisp or read the README in examples)  
  

