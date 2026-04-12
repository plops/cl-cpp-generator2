#!/bin/bash
# Format gen04.lisp using emacs

cd /home/kiel/stage/cl-py-generator/example/143_helium_gemini

echo "Formatting gen04.lisp with emacs..."
emacs --batch -l ~/.emacs gen04.lisp \
       --eval "(package-initialize)" \
       --eval "(require 'slime)" \
       --eval "(slime-setup '(slime-cl-indent))" \
       --eval "(setq lisp-indent-function 'common-lisp-indent-function)" \
       --eval "(indent-region (point-min) (point-max))" \
       -f save-buffer

if [ $? -eq 0 ]; then
    echo "✓ gen04.lisp formatted successfully"
else
    echo "✗ Error formatting gen04.lisp"
    exit 1
fi
