
* Investigate performance of using clang with C++20 modules

- try to include standard library using modules


** References
- https://stackoverflow.com/questions/66411157/import-std-lib-as-modules-with-clang
- https://clang.llvm.org/docs/PCHInternals.html


* Conventional clang call

- for comparison source00/use.cpp shows conventional include:

```
[martin@localhost source00]$ time ./compile0.sh

real	0m1.070s
user	0m0.960s
sys	0m0.098s
```
- it takes 1sec to include iostream and vector

* Conventional clang call with precompiled header

- code is in source01

- precompile header
```
[martin@localhost source01]$ time ./compile0.sh 

real	0m1.137s
user	0m1.011s
sys	0m0.088s
[martin@localhost source01]$ ls -tlrh *.gch
-rw-r--r--. 1 martin martin 6.4M Jan 28 14:16 std_mod.hpp.gch

```


-compile the cpp program:
```
[martin@localhost source01]$ time ./compile1.sh

real	0m1.095s
user	0m0.957s
sys	0m0.112s
```

I don't think clang uses the gch file. Adding -H to the command in
compile1.sh doesn't show it. I don't know how to fix this right now.


* C++20 modules
- code is in source02

- prepare the std_mod.pcm file (this only needs to be executed once and repeated whenever the headers change):  

```
[martin@localhost source]$ time ./compile1.sh 

real	0m1.239s
user	0m1.094s
sys	0m0.099s
```

- use the module:

```
[martin@localhost source]$ time ./compile2.sh

real	0m0.248s
user	0m0.165s
sys	0m0.052s

```

* Conclusion

- clang compiles in 0.25sec instead of >1sec when a module is used.