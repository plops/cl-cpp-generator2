(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more)))
  (setf *features* (set-exclusive-or *features* (list :more))))

(let ()
  (defparameter *source-dir* #P"example/138_lambdas/source01/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")
    
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     (include<>
      iostream
      memory
					;string
      vector
					;algorithm
      					;chrono
					;thread
      					;filesystem
					;unistd.h
					;cstdlib
      cmath
      complex
      unordered_map
      format
      thread
      
      )
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
       		(type char** argv))

       ,(lprint :msg "main entry point" :vars `(argc (aref argv 0)))

       (comments "45:00 overload sets")
     ))
   :omit-parens t
   :format t
   :tidy nil))

