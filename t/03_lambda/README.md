# t/03_lambda

Unit tests for `lambda` emission (parameters, `(values ...)` return type,
`(capture ...)` lists). Proof of concept, mirroring `t/02_paren_precedence`.

```sh
./t/03_lambda/run.sh
```

Exit code 0 on success, 1 on failure. Three layers:

1. **string tests** — emitted C++ against hand-verified reference strings
   (whitespace-normalised). Pins the reported bug: a parameterless lambda
   with a return type must emit `[&]() -> int`, not `[&] -> int`.
2. **value tests** — one generated C++ program assigns every lambda to an
   `auto` variable, calls it and compares against an expected integer
   (entries with `:direct t`, e.g. an immediately invoked lambda, are
   emitted verbatim). Needs `g++` or `clang++`; without a compiler this
   layer is skipped with a message instead of failing. Generated C++ lands
   in `build/` (gitignored via `**/build/`).
3. **error tests** — forms that must signal (multiple `values` types,
   mirroring `parse-defun`) are checked for their message. `break` enters
   the debugger past `handler-case`, so the message is caught with a
   throwing `sb-ext:*invoke-debugger-hook*`.
