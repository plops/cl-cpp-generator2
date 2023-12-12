- discussion of cmake support for modules: https://www.kitware.com/import-cmake-c20-modules/
- cmake has to parse c++ to get dependencies (it uses c++ compiler for this)

-  CMake 3.25 or newer will have support for Modules.
-  CMake 3.28 has official support for C++ 20 named modules enabled
   without having to set the CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API
   variable
   https://cmake.org/cmake/help/latest/manual/cmake-cxxmodules.7.html

- as of now 3.28 is in testing of gentoo
- you will need a c++ compiler that has support for p1689r5
- the gcc fork has not been accepted upstream yet
- clang compiler version 16 or newer contains the p1689 implementation
  required for CMake

- i installed cmake 3.28 and use clang16 on gentoo. compilation works
