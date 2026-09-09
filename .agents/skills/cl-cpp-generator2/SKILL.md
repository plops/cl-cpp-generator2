---
name: cl-cpp-generator2
description: Provides documentation and guides code generation using the cl-cpp-generator2 Common Lisp to C++ transpiler. Use when writing, modifying, or testing Common Lisp forms to be transpiled into C/C++ (including GLSL/HLSL shaders).
---

# CL-CPP-Generator2 Transpiler

This skill documents the Lisp-like S-Expression DSL used by the `cl-cpp-generator2` transpiler to emit C++ and shader code. Use this documentation to write correct Lisp forms, extend the transpiler, or understand generated C++ outputs.

## When to use this skill

- When implementing new features or writing new Lisp generator code in `example/` or at the workspace root.
- When working on Shadertoy-like shaders using transpiled Common Lisp.
- When debugging generated C++ or GLSL files or updating the core emitter in [c.lisp](file:///home/kiel/stage/cl-cpp-generator2/c.lisp).

## Language Architecture

The transpiler takes Lisp S-Expressions representing a C++ AST and compiles them into C++ code strings.
The package name is `:cl-cpp-generator2`.
The primary public entrypoints are:
- `emit-c`: Emits C++ code from a Lisp form.
- `write-source`: Transpiles code and writes the result to a C++ file (by default formatting it with `clang-format`).
- `write-notebook`: Writes the code to a Jupyter Notebook `.ipynb` file using xeus-cling.

The transpiler uses an `:invert` readtable-case, meaning that lower-case code is preserved and maps naturally to C/C++ identifiers.

> [!IMPORTANT]
> **Symbol Package Matching Alert**:
> The transpiler dispatcher uses exact symbol equality (`eq`) to match DSL keywords (like `curly`, `paren*`, `defun`, etc.).
> - Make sure the DSL symbols are either exported by `:cl-cpp-generator2` or package-qualified when writing generator files.
> - Unexported/incorrectly qualified symbols read in another package will fail to match the emitter dispatcher and default to a C++ function call representation (e.g., `symbol_name()`).

---

## DSL Reference & Mapping Guide

Below is a complete reference of the Lisp forms supported by the transpiler and their generated C++ syntax.

### 1. Variables & Assignments
- Direct assignment: `(= a b)` &rarr; `a = b`
- Multiple assignments: `(setf a 1 b 2)` &rarr; `a = 1;\nb = 2;`
- Increment / Decrement:
  - `(incf a 2)` &rarr; `a += 2`
  - `(decf a 3)` &rarr; `a -= 3`
- Compound Assignments:
  - `(*= a 2)` &rarr; `a *= 2`
  - `(/= a 3)` &rarr; `a /= 3`
  - `(^= a 4)` &rarr; `a ^= 4`

### 2. Variable Declarations (`let`, `letc`, `letd`)
- **`let` (normal variable, auto-typed by default if type is undeclared)**:
  ```lisp
  (let ((a 1)
        (b 2d0))
    (declare (type int a)
             (type double b)))
  ```
  Emits:
  ```cpp
  int a = 1;
  double b = 2.0;
  ```
- **`letc` (const variable declaration)**:
  `letc` acts like `let` but adds `const`.
- **`letd` (decltype variable declaration, no definition)**:
  Used to declare variables using `decltype` (e.g., `decltype(expr) var;`).

### 3. Basic Operators
All operators wrap their operands in parentheses to preserve operator precedence.
- **Arithmetic**:
  - `(+ a b)` &rarr; `((a) + (b))`
  - `(- a b)` &rarr; `((a) - (b))`
  - `(* a b c)` &rarr; `((a) * (b) * (c))`
  - `(/ a b)` &rarr; `((a) / (b))`
  - `(% a b)` &rarr; `((a) % (b))` (modulo)
- **Bitwise**:
  - `(& a b)` or `(logand a b)` &rarr; `((a) & (b))`
  - `(logior a b)` &rarr; `((a) | (b))`
  - `(<< a b)` &rarr; `((a) << (b))`
  - `(>> a b)` &rarr; `((a) >> (b))`
  - `(^ a b)` or `(xor a b)` &rarr; `((a) ^ (b))`
- **Comparison**:
  - `(== a b)` &rarr; `((a) == (b))`
  - `(!= a b)` &rarr; `((a) != (b))`
  - `(< a b)` &rarr; `((a) < (b))`
  - `(<= a b)` &rarr; `((a) <= (b))`
  - `(> a b)` &rarr; `((a) > (b))`
  - `(>= a b)` &rarr; `((a) >= (b))`
- **Logical**:
  - `(and a b)` &rarr; `((a) && (b))`
  - `(or a b)` &rarr; `((a) || (b))`
  - `(not a)` &rarr; `(!(a))`

### 4. Brackets & Accessors
- **Paren / Parentheses**: `(paren a b)` &rarr; `(a, b)`
- **Angle Brackets**: `(angle a b)` &rarr; `<a, b>`
- **Square Brackets**: `(bracket a b)` &rarr; `[a, b]`
- **Curly Braces**: `(curly a b)` &rarr; `{a, b}`
- **Designated Initializer**:
  `(designated-initializer x 1 y 2)` &rarr; `{.x = 1, .y = 2}`
- **Array Access**: `(aref arr idx)` &rarr; `arr[idx]`
- **Member Access**:
  - Dot member: `(dot obj attr)` &rarr; `obj.attr`
  - Pointer arrow: `(-> ptr attr)` &rarr; `ptr->attr`
- **Scope resolution (`::`)**: `(scope std string)` &rarr; `std::string`

### 5. Control Flow
- **If/Else / When / Unless**:
  - `(if condition true-stmt false-stmt)` &rarr; `if (condition) {\ntrue-stmt\n} else {\nfalse-stmt\n}`
  - `(when condition body*)` &rarr; `if (condition) {\nbody*\n}`
- **Cond (multi-branch conditional)**:
  Compiles to chained `if ... else if ... else`.
- **Loops**:
  - `for`: `(for (init cond step) body*)` &rarr; `for (init; cond; step) {\nbody*\n}`
  - `while`: `(while cond body*)` &rarr; `while (cond) {\nbody*\n}`
  - `dotimes`: `(dotimes (i 100) body*)` &rarr; C++ loop from 0 to 99.
- **Return & Coroutines**:
  - `(return expr)` &rarr; `return expr;`
  - `(co_return expr)` &rarr; `co_return expr;`
  - `(co_await expr)` &rarr; `co_await expr;`
  - `(co_yield expr)` &rarr; `co_yield expr;`

### 6. Functions & Classes
- **Function Definitions (`defun`, `defun*`, `defun+`)**:
  - `defun` uses the dynamic `header-only` setting of the compiler (e.g. for generating headers vs source implementations).
  - `defun*` generates a header declaration (declaration only).
  - `defun+` generates a full function implementation.
  ```lisp
  (defun foo (x y)
    (declare (type int x)
             (type float y)
             (values float)
             inline)
    (return (+ x y)))
  ```
  Emits:
  ```cpp
  inline float foo(int x, float y) {
      return ((x) + (y));
  }
  ```
  Declarations supported inside the `declare` form:
  - `(type <type> <vars>)` to declare variable types.
  - `(values <type>)` to declare function return type.
  - `inline`, `const`, `static`, `explicit`, `virtual`, `noexcept`, `final`, `override`, `pure` to apply those C++ specifiers/qualifiers.
  - `(template <args>)` to define a template function.

- **Classes / Structs**:
  - `defclass`: `(defclass ClassName (ParentClass) body*)` &rarr; class declaration.
  - `struct`: `(struct StructName body*)` &rarr; struct declaration.

### 7. Preprocessor Directives
- **Include**:
  - `(include<> stdio.h)` &rarr; `#include <stdio.h>`
  - `(include "my_header.h")` &rarr; `#include "my_header.h"`
- **Pragma**:
  - `(pragma once)` &rarr; `#pragma once`

---

## Compile-Time Code Generation & Templating

Since `cl-cpp-generator2` transpiles Lisp S-Expressions at compile time, you can leverage standard Common Lisp list processing and macro/loop constructs (like `loop`, `backquote`, and comma evaluation) to dynamically emit complex shader paths, coordinate mapping tables, or mathematical variations.

For example, when writing shaders, we can unroll loops or generate multiple SDF-distance functions dynamically.
