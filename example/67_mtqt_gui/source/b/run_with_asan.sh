ASAN_OPTIONS=fast_unwind_on_malloc=0:symbolize=1 ASAN_SYMBOLIZER_PATH=/usr/bin/llvm-symbolizer ./mytest 2>&1 | tee mytest.out
