(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

(format t "Test let with type: ~a~%" 
	(emit-c :code `(let ((a 1 :type int)) (return a))))
