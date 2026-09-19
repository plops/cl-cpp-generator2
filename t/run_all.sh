#!/bin/sh
# Run every automated transpiler test suite and propagate the first failure.
#
# t/01_paren is intentionally excluded: it is a manual workflow with
# hardcoded absolute paths (/home/martin/stage/...), destructive `rm` steps
# and no exit-code contract, so it cannot run unattended.
set -e
cd "$(dirname "$0")/.."
./t/02_paren_precedence/run.sh
./t/03_lambda/run.sh
