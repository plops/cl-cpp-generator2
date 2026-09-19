# t/03_lambda

Unit tests for `lambda` emission (parameters, `(values ...)` return type,
`(capture ...)` lists). Proof of concept, mirroring `t/02_paren_precedence`.

```sh
./t/03_lambda/run.sh
```

Exit code 0 on success, 1 on failure. Two layers:

1. **string tests** — emitted C++ against hand-verified reference strings
   (whitespace-normalised). Pins the reported bug: a parameterless lambda
   with a return type must emit `[&]() -> int`, not `[&] -> int`.
2. **value tests** — one generated C++ program assigns every lambda to an
   `auto` variable, calls it and compares against an expected integer.
   Needs `g++` or `clang++`; without a compiler this layer is skipped with
   a message instead of failing. Generated C++ lands in `build/`
   (gitignored via `**/build/`).
