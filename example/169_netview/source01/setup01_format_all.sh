#!/bin/bash
clang-format -i `find .|egrep '\.(cpp|h)' `
