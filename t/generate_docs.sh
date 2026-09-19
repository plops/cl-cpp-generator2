#!/bin/sh
# Generate SUPPORTED_FORMS.md from the executed test tables, or check it.
#
# Loading the suites runs them (t/02 fully including random tests, t/03
# fully); both quit nonzero on failure, so the file can never be generated
# from failing tests. The C++ value layers skip without a compiler.
#
# Usage: ./t/generate_docs.sh [--check]
# --check exits 1 when the committed file differs (CI freshness gate).
set -e
cd "$(dirname "$0")/.."
mode="${1:-}"
tmp="$(mktemp /tmp/supported_forms.XXXXXX.md)"
trap 'rm -f "$tmp"' EXIT INT TERM
sbcl --noinform --disable-debugger \
     --load t/generate-docs.lisp \
     --eval "(cl-cpp-generator2::write-docs \"$tmp\")" \
     --quit
if [ "$mode" = "--check" ]; then
  if cmp -s "$tmp" SUPPORTED_FORMS.md; then
    echo "SUPPORTED_FORMS.md is current"
  else
    echo "SUPPORTED_FORMS.md is stale; run ./t/generate_docs.sh" >&2
    exit 1
  fi
else
  mv "$tmp" SUPPORTED_FORMS.md
  echo "wrote SUPPORTED_FORMS.md"
fi
