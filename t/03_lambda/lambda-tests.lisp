;;;; Unit tests for LAMBDA emission (parameters, return type, captures).
;;;;
;;;; Run with:
;;;;   sbcl --noinform --disable-debugger --load t/03_lambda/lambda-tests.lisp --quit
;;;; or via t/03_lambda/run.sh
;;;;
;;;; Three layers (proof of concept, mirroring t/02_paren_precedence):
;;;;
;;;;   1. STRING TESTS -- emitted C++ against hand-verified reference strings
;;;;      (whitespace is normalised before comparison).
;;;;
;;;;   2. VALUE TESTS -- one generated C++ program assigns every lambda to an
;;;;      `auto' variable, calls it and compares against an expected integer.
;;;;      A missing `()' or a wrong capture list fails to compile or fails
;;;;      the value check, so this layer is the semantic ground truth.
;;;;      Entries with :direct t hold a complete expression (e.g. an
;;;;      immediately invoked lambda) emitted verbatim instead.
;;;;
;;;;   3. ERROR TESTS -- forms that must signal (e.g. multiple `values'
;;;;      types, mirroring parse-defun) are checked for their message.
;;;;
;;;; Layer 2 is skipped (with a warning, not a failure) when no C++ compiler
;;;; is available. Generated C++ lands in build/ (gitignored via **/build/).

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2" :silent t)
)

(in-package :cl-cpp-generator2)

(defparameter *failures* 0)
(defparameter *checks* 0)

(defun report (ok name fmt &rest args)
  (incf *checks*)
  (unless ok
    (incf *failures*)
  )
  (format t "~:[FAIL~;ok  ~] ~a~@[ ~a~]~%"
    ok name (apply #'format nil fmt args))
)

(defun emit-str (code)
  (m-of (emit-c :code code))
)

(defun normalize (s)
  "Collapse every run of whitespace to a single space."
  (with-output-to-string (out)
    (loop with pending = nil
      for c across s
      do (if (member c '(#\Space #\Tab #\Newline #\Return))
        (setf pending t)
        (progn
          (when pending
            (write-char #\Space out)
            (setf pending nil)
          )
          (write-char c out)))
    )
  )
)

;;; :code     s-expression handed to emit-c
;;; :expected hand-verified C++ text (compared whitespace-normalised)
;;; :call     suffix appended to `f' in the value test, e.g. "()" or "(37)"
;;; :value    expected integer when the lambda is invoked
(defparameter *lambda-tests*
  '(;; the reported bug: no parameters plus (values ...) dropped the `()',
    ;; emitting the invalid `[&] -> int { ... }'
    (:name no-params-with-return
     :description "A parameterless lambda with a return type keeps its empty parameter list. Omitting the `()` is invalid C++."
     :code (lambda () (declare (values int)) (return 42))
     :expected "[&]() -> int { return 42; }"
     :call "()"
     :value 42)
    (:name no-params-no-return
     :description "Without a return type the parameter list is still emitted; the type is deduced from `return`."
     :code (lambda () (return 42))
     :expected "[&]() { return 42; }"
     :call "()"
     :value 42)
    (:name explicit-capture-with-return
     :description "An explicit by-copy capture appears in the first bracket pair."
     :code (lambda () (declare (capture x) (values int)) (return x))
     :expected "[x]() -> int { return x; }"
     :call "()"
     :value 5)
    (:name default-capture-sees-outer-x
     :description "With no capture declaration the lambda defaults to `[&]` and sees outer variables."
     :code (lambda (a) (declare (type int a) (values int)) (return (+ a x)))
     :expected "[&](int a) -> int { return (a)+(x); }"
     :call "(37)"
     :value 42)
    (:name two-params-no-return
     :description "Typed parameters render as a C++ parameter list; no `->` without `values`."
     :code (lambda (a b) (declare (type int a) (type int b)) (return (+ a b)))
     :expected "[&](int a, int b) { return (a)+(b); }"
     :call "(40, 2)"
     :value 42)
    (:name two-captures-with-return
     :description "Multiple explicit captures are comma-separated."
     :code (lambda () (declare (capture x y) (values int)) (return (+ x y)))
     :expected "[x,y]() -> int { return (x)+(y); }"
     :call "()"
     :value 12)
    (:name by-value-default
     :description "A `=` capture-default captures outer variables by copy."
     :code (lambda () (declare (capture =) (values int)) (return 42))
     :expected "[=]() -> int { return 42; }"
     :call "()"
     :value 42)
    (:name capture-default-moves-first
     :description "A capture-default must come first in C++; the generator reorders `(capture x =)` to `[=,x]`."
     :code (lambda () (declare (capture x =) (values int)) (return (+ x 1)))
     :expected "[=,x]() -> int { return (x)+(1); }"
     :call "()"
     :value 6)
    (:name mixed-captures-with-params
     :description "Explicit by-copy and by-reference captures mix freely alongside parameters."
     :code (lambda (a b) (declare (capture x &y)
                                  (type int a) (type int b)
                                  (values int))
             (return (+ (+ a b) (+ x y))))
     :expected "[x,&y](int a, int b) -> int { return ((a)+(b))+((x)+(y)); }"
     :call "(1, 2)"
     :value 15)
    (:name untyped-param-is-auto
     :description "Parameters without a declared type become `auto` (a C++14 generic lambda)."
     :code (lambda (q) (return q))
     :expected "[&](auto q) { return q; }"
     :call "(9)"
     :value 9)
    (:name multi-form-body
     :description "Bodies hold multiple forms; earlier forms emit as statements before `return`."
     :code (lambda () (declare (values int))
             (let ((y 2))
               (return (+ y 40))))
     :expected "[&]() -> int { auto y = 2; return (y)+(40); }"
     :call "()"
     :value 42)
    (:name immediate-call
     :description "A lambda in head position is parenthesised and can be invoked immediately."
     :code ((lambda (a) (declare (type int a) (values int)) (return a)) 41)
     :expected "([&](int a) -> int { return a; })(41)"
     :value 41
     :direct t))
)

;;; :code s-expression that must signal instead of emitting
;;; :error substring of the expected condition message
(defparameter *lambda-error-tests*
  '((:name multiple-values-error
     :description "More than one `values` type is rejected, mirroring `defun`."
     :code (lambda () (declare (values int float)) (return 1))
     :error "multiple return values unsupported"))
)

(defun run-string-tests ()
  (format t "~&== string tests ==~%")
  (dolist (e *lambda-tests*)
    (destructuring-bind (&key name code expected call value direct description) e
      (declare (ignore call value direct description))
      (let ((got (handler-case (normalize (emit-str code))
                   (condition (c) (format nil "<error ~a>" c)))))
        (report (string= got expected) name
          "got ~s expected ~s" got expected)))
  )
)

(defun cxx-compiler ()
  (loop for c in '("/usr/bin/g++" "/usr/bin/clang++" "/usr/bin/c++")
    when (probe-file c) return c)
)

(defun run-value-tests ()
  (format t "~&== value tests ==~%")
  (let ((compiler (cxx-compiler)))
    (if (not compiler)
      (format t "SKIP no C++ compiler found~%")
      (let* ((dir (asdf:system-relative-pathname
                    'cl-cpp-generator2 "t/03_lambda/build/"))
             (src (merge-pathnames "lambda_value_tests.cpp" dir))
             (exe (merge-pathnames "lambda_value_tests" dir)))
        (ensure-directories-exist dir)
        (with-open-file (s src :direction :output :if-exists :supersede
                          :if-does-not-exist :create)
          (format s "// generated by t/03_lambda/lambda-tests.lisp~%")
          (format s "#include <cstdio>~%~%")
          (format s "int main() {~%  int fails = 0;~%")
          (format s "  int x = 5; int y = 7;~%")
          (dolist (e *lambda-tests*)
            (destructuring-bind (&key name code expected call value direct description) e
              (declare (ignore expected description))
              (let ((lam (emit-str code)))
                (if direct
                  (format s "  {~%    long got = (long)(~a);~%" lam)
                  (progn
                    (format s "  {~%    auto f = ~a;~%" lam)
                    (format s "    long got = (long)(f~a);~%" call))))
                (format s "    if (got != ~a) { std::printf(\"FAIL ~a: %ld != ~a\\n\", got); fails++; }~%"
                  value name value)
                (format s "    else { std::printf(\"ok   ~a = %ld\\n\", got); }~%" name)
                (format s "  }~%")))
          (format s "  std::printf(\"%d value failures\\n\", fails);~%")
          (format s "  return fails == 0 ? 0 : 1;~%}~%"))
        (let ((compile-ok
                (zerop (sb-ext:process-exit-code
                         (sb-ext:run-program compiler
                           (list "-std=c++20" "-O0" "-w"
                             "-o" (namestring exe)
                             (namestring src))
                           :output *standard-output*
                           :error *standard-output*)))))
          (report compile-ok "value-tests-compile" "~a" (namestring src))
          (when compile-ok
            (let ((code (sb-ext:process-exit-code
                          (sb-ext:run-program (namestring exe) nil
                            :output *standard-output*
                            :error *standard-output*))))
              (report (zerop code) "value-tests-run"
                "~a lambda~:p compiled from ~a"
                (length *lambda-tests*) (namestring src)))))
      )
    )
  )
)

(defun run-error-tests ()
  (format t "~&== error tests ==~%")
  (dolist (e *lambda-error-tests*)
    (destructuring-bind (&key name code error description) e
      (declare (ignore description))
      ;; break enters the debugger via *invoke-debugger-hook* and bypasses
      ;; handler-case, so catch its message with a throwing hook instead
      (let ((msg (catch 'broke
                   (let ((sb-ext:*invoke-debugger-hook*
                           (lambda (c hook)
                             (declare (ignore hook))
                             (throw 'broke (format nil "~a" c)))))
                     (emit-str code)
                     "<no error>"))))
        (report (and (string/= msg "<no error>")
                  (search error msg)
                  t)
          name "got ~s expected substring ~s" msg error)))
  )
)

(defun run-lambda-tests ()
  (setf *failures* 0)
  (setf *checks* 0)
  (run-string-tests)
  (run-value-tests)
  (run-error-tests)
  (format t "~%~a checks, ~a failures~%" *checks* *failures*)
  (unless (zerop *failures*)
    (sb-ext:quit :unix-status 1))
)

(run-lambda-tests)
