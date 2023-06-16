(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
   (progn
    (defparameter *source-dir* #P"t/01_paren/source00/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*)))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (ensure-directories-exist *full-source-dir*)

  (loop for e in `((:name basic :code (* 3 (+ 1 2))))
	and e-i from 0
	do
	   (destructuring-bind (&key code name) e
	    (write-source
	     (asdf:system-relative-pathname
	      'cl-cpp-generator2
	      (merge-pathnames (format nil "c~2,'0d_~a.cpp" e-i name)
			       *source-dir*))
	     `(do0
	       (include<> cassert)
	       (defun main (argc argv)
		 (declare (values int)
			  (type int argc)
			  (type char** argv))
		 (assert (== ,code
			     ,(eval code)))
		 (return 0)))))))



