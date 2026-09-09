#!/bin/sh
# Run the parenthesis precedence unit tests.
#
# Requires sbcl with quicklisp and cl-cpp-generator2 reachable via
# ~/quicklisp/local-projects.  A C++ compiler (g++ or clang++) is optional; the
# value and the randomized differential layer are skipped without one.
set -e
cd "$(dirname "$0")/../.."
exec sbcl --noinform --disable-debugger \
     --load t/02_paren_precedence/paren-tests.lisp \
     --quit
