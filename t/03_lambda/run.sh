#!/bin/sh
# Run the lambda emission unit tests.
#
# Requires sbcl with quicklisp and cl-cpp-generator2 reachable via
# ~/quicklisp/local-projects.  A C++ compiler (g++ or clang++) is optional; the
# value layer is skipped without one.
set -e
cd "$(dirname "$0")/../.."
exec sbcl --noinform --disable-debugger \
     --load t/03_lambda/lambda-tests.lisp \
     --quit
