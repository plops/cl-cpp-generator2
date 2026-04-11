;; run this script like this: sbcl --load gen01.lisp --quit
(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more)))
  (setf *features* (set-exclusive-or *features* (list :more))))

(progn
  (defparameter *source-dir* #P"example/196_forthjit/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname 'cl-cpp-generator2 *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  
  (load "util.lisp")
  
  (let* ((l-prim0 `((:name Add :symbol +) (:name Sub :symbol -) (:name Mul :symbol *) (:name Dup)
		    (:name Drop) (:name Swap) (:name Dot :symbol ".") (:name LessThan :symbol < :short lt) (:name GreaterThan :symbol > :short gt)
		    (:name Equal :symbol = :short eq) (:name Fetch :symbol @) (:name Store :symbol !)))
	 (l-prim (loop for e in l-prim0
		       collect
		       (destructuring-bind (&key
					      name
					      (symbol (string-upcase (format nil "~a" name)))
					      (short (string-downcase (format nil "~a" name)))) e
			 `(:name ,name
			   :symbol ,symbol
			   :short ,short)))))
    (defparameter *l-prim* l-prim)
    (load "operation.lisp")
    (load "jit.lisp")
    (load "helpers.lisp")
    ;(load "vm.lisp")
    ;(load "main.lisp")
    ))
