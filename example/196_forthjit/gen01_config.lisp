(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(load "util.lisp")

(defparameter *project-dir* "/home/kiel/stage/cl-cpp-generator2/example/196_forthjit")
(defparameter *full-source-dir* *project-dir*)

(defparameter *l-prim*
  (let* ((l-prim0 `((:name Add :symbol +) (:name Sub :symbol -) (:name Mul :symbol *) (:name Dup)
		    (:name Drop) (:name Swap) (:name Dot :symbol ".") (:name LessThan :symbol < :short lt) (:name GreaterThan :symbol > :short gt)
		    (:name Equal :symbol = :short eq) (:name Fetch :symbol @) (:name Store :symbol !))))
    (loop for e in l-prim0
	  collect
	  (destructuring-bind (&key name symbol short) e
	    (let ((actual-short (string-downcase (format nil "~a" (or short name))))
		  (actual-symbol (or symbol (string-upcase (format nil "~a" name)))))
	      `(:name ,name
		:symbol ,actual-symbol
		:short ,actual-short))))))
