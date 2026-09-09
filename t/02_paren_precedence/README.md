# t/02_paren_precedence

Unit tests for the parenthesis elision of `emit-c` (`write-source … :omit-parens t`).

```sh
./t/02_paren_precedence/run.sh
```

Exit code 0 on success, 1 on failure. Four layers:

1. **string tests** — emitted C++ against hand-verified reference strings, for
   the paren-eliding mode and (where given) the fully parenthesized mode. This
   is the only layer that catches bugs which are equally wrong in both modes.
2. **value tests** — one generated C++ program evaluates every expression in
   both modes and compares against an expected integer.
3. **helper tests** — `effective-operator` and `binds-looser-p`.
4. **randomized differential test** — random expressions, fully parenthesized
   output as the oracle, compared numerically in C++. Fixed seed, so failures
   are reproducible. Run more with

   ```sh
   sbcl --noinform --disable-debugger \
        --load t/02_paren_precedence/paren-tests.lisp \
        --eval '(cl-cpp-generator2::run-random-tests :count 2000 :depth 4 :seed 42)' \
        --quit
   ```

Layers 2 and 4 need `g++` or `clang++`; without a compiler they are skipped with
a message instead of failing. Generated C++ lands in `build/` (gitignored).

Background and the bugs these tests pin down:
`plan/20260830_01_omit_paren_bug/walkthrough.md`.
