This Forth subset is a minimal, integer-only environment designed for basic arithmetic, stack manipulation, and simple conditional logic. Below is a summary of its core behavior and constraints.
## Core Constraints

* Case-Insensitivity: DUP, dup, and dUp are identical.
* Memory Limits: Max 64 dictionary entries (words + variables) and a 256-item data stack.
* No Redefinition: Unlike standard Forth, the first definition of a word is always used; you cannot shadow old words with new ones.
* Formatting: Tokens must be separated by spaces. Tabs and other whitespace are not recognized as separators.
* One-Line Conditionals: While you can define a word across multiple lines, IF...ELSE...THEN blocks must be completed on a single line.

## Quick Syntax Reference

| Category | Words / Syntax | Notes |
|---|---|---|
| Arithmetic | + - * | No division or modulo. |
| Stack | DUP DROP SWAP | Standard effects; ROT and OVER are missing. |
| Logic | < > = | True = -1, False = 0. |
| Variables | VARIABLE [name] | Use @ to fetch, ! to store. |
| Definitions | : [name] ... ; | Must be non-empty; name truncated at 31 chars. |
| Output | . | Pops and prints the top value. |

## Error Handling

* Unknown_Word: Token not found in the dictionary.
* Stack_Error: Underflow/overflow, arithmetic overflow, or exceeding the 10k instruction "fuel" limit.
* Compile_Error: Misused IF/THEN, unknown words during colon definition, or exceeding dictionary capacity.

## What's Missing?
There are no loops (DO/LOOP, BEGIN/UNTIL), no constants, no comments, and no strings. It is strictly a line-oriented tool for immediate integer calculation and simple branching logic.

