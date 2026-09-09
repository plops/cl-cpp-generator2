# Parenthesis Debugging Guide

When working with deep S-Expression structures in this workspace, a single unmatched parenthesis can cause SBCL's reader to trigger `end of file on form` compiler errors. Use these three highly effective ways to locate structural mismatches in Lisp code.

---

## 1. Using a Python Nesting-Depth Tracker Script
A simple python one-liner can scan the file, count the open `(` and close `)` parentheses, track the balance per line, and list the exact line numbers where the unclosed blocks begin.

### How to Run:
Run this directly in your terminal in the directory of your Lisp file (replace `gen.lisp` with your target file name):
```bash
python3 -c "
with open('gen.lisp') as f:
    lines = f.readlines()

balance = 0
line_no = 0
stack = []
for idx, line in enumerate(lines):
    line_no = idx + 1
    # Count parens in the current line
    for col, char in enumerate(line):
        if char == '(':
            balance += 1
            stack.append((line_no, col + 1))
        elif char == ')':
            balance -= 1
            if balance < 0:
                print(f'Extra close paren \')\' at line {line_no}, col {col + 1}')
                balance = 0 # reset
            else:
                stack.pop()

print('Final Balance:', balance)
if balance > 0:
    print('Unclosed open parentheses start at:')
    for l, c in stack[-5:]:
        print(f'  line {l}, col {c}')
"
```

### Why it is helpful:
* Tells you the exact starting line/column of the unclosed `(` forms (usually the parent forms like `progn`, `let*`, or `defun`).
* Pinpoints exact positions of any extra closing parentheses.

---

## 2. Using the Common Lisp Pretty Printer (`pprint`)
Common Lisp's built-in pretty printer automatically formats code according to nesting depth. If there is a missing closing parenthesis, printing a form will result in either an error or a visual structure that instantly reveals the mismatch.

### Interactive Debugging in REPL:
Load the file into the Lisp REPL. If there is a read error, SBCL will enter the debugger. You can use read commands or load forms step-by-step:
```lisp
;; Read a file form-by-form and print it.
;; If a form has a mismatch, the printout will stop or indent wildly.
(with-open-file (stream "gen.lisp")
  (loop for form = (read stream nil :eof)
        until (eq form :eof)
        do (progn
             (format t \"~%--- Form ---~%\")
             (pprint form))))
```

### Why it is helpful:
* Since `read` parses entire forms, the last form printed successfully before the `read` error is the one preceding the broken form. The form where execution halts has the mismatch.

---

## 3. Emacs Batch Auto-Indentation
If you use Emacs, you can auto-indent the Lisp file from the shell using batch mode. A structural indentation shift immediately makes the missing parenthesis visible.

### How to Run:
```bash
emacs --batch gen.lisp --eval '(indent-region (point-min) (point-max))' -f save-buffer
```

### Why it is helpful:
* The indentation of the code will drift far to the right or left starting directly below the unclosed block.
* Opening the file in any text editor after auto-indentation makes the mismatched scope visually obvious.
