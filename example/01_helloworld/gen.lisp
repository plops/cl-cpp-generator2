(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)



(progn
  (defparameter *code-file* (asdf:system-relative-pathname 'cl-cpp-generator2 "example/01_helloworld/source/helloworld.c"))
  (let* ((code
	  `(do0
	    (include <stdio.h>)
	    (defun main (argc argv)
	      (declare (type int argc)
		       (type char** argv)
		       (values int))
	      (printf (string "hello world!"))
	      (return 0)))))
    (write-source *code-file* code)))
