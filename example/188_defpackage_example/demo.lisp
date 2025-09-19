(load "~/quicklisp/setup.lisp")

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(defpackage #:my-cpp-project
  (:use #:cl #:cl-cpp-generator2)) 

(in-package #:my-cpp-project)


(format t "~a:~%~a~%"
        *package*
        (cl-cpp-generator2:emit-c :code
          `(do0
             (include <stdio.h>) 
             (defstruct0 struct_a (a int)))))
