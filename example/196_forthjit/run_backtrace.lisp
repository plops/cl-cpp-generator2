(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(load "gen01_config.lisp")
(load "gen01_operation.lisp")
(load "gen01_jit_compiler.lisp")

(handler-case
    (load "gen01_forth_vm.lisp")
  (error (c)
    (format t "CAUGHT ERROR: ~a~%" c)
    (sb-debug:print-backtrace)
    (uiop:quit 1)))

(load "gen01_main.lisp")
