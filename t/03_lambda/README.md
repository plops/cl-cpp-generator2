# t/03_lambda

Unit tests for `lambda` emission: parameters (typed, `auto`, none),
`(values ...)` return types (including `void`, pointers and `&optional`
cutoff), and `(capture ...)` lists (explicit, default, by-value,
mixed, init-capture, `this`).

```sh
./t/03_lambda/run.sh
```

Exit code 0 on success, 1 on failure. Three layers:

1. **string tests** — emitted C++ against hand-verified reference strings
   (whitespace-normalised), in both modes: `:expected` pins the fully
   parenthesized output, `:omit` the `:omit-parens t` output. Pins the
   reported bugs: a parameterless lambda with a return type must emit
   `[&]() -> int`, not `[&] -> int`, and a capture-default sorts first
   (`[=,x]`, not `[x,=]`).
2. **value tests** — two generated C++ programs (fully parenthesized and
   elided, mirroring `t/02`'s vfull/vomit and `cl-rust-generator`'s dual
   runs) check every testable lambda: entries with `:value` bind `auto f`
   and compare the call result (entries with `:direct t`, e.g. an
   immediately invoked lambda, are emitted verbatim); entries with
   `:check` verify a custom condition instead (for `:void` results and
   pointer returns). Both programs must yield the same values. Needs
   `g++` or `clang++`; without a compiler this layer is skipped with a
   message instead of failing. Generated C++ lands in `build/` (gitignored
   via `**/build/`). Entries without `:value` or `:check` (`this`
   capture, lambda-as-argument) are string-tested only: no object or
   callee exists in the harness.
3. **error tests** — forms that must signal (multiple `values` types,
   mirroring `parse-defun`) are checked for their message. `break` enters
   the debugger past `handler-case`, so the message is caught with a
   throwing `sb-ext:*invoke-debugger-hook*`.

Cases double as documentation: `:description` fields render into
`SUPPORTED_FORMS.md` (see `./t/generate_docs.sh`).
