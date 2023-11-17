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
      ;memory
					;string
      ;vector
					;algorithm
      					;chrono
					;thread
      					;filesystem
					;unistd.h
					;cstdlib
      cmath
      complex
      ;unordered_map
      format
      ;thread
      
      )
     (comments "The overload set behaves similar to typecase in Common Lisp. This code defines a struct called 'overload' that inherits from types provided by Ts. The call operators of the types can be invoked directly through this 'overload' struct, which it inherits. The 'overload' struct does not have a user-defined constructor or private members and is considered an aggregate.")

     (comments "During creation the object is provided with a set of lambdas as base classes for the overload set, it will inherit the call operator from these lambdas.")
     (space template "<typename... Ts>"
	    struct "overload : Ts..."
	    (progn
	      "using Ts::operator()...;"))
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
       		(type char** argv))

       ,(lprint :msg "main entry point" :vars `(argc (aref argv 0)))

       (do0
	(comments "45:00 overload sets")
	(let ((f (curly ,@(loop for e in `((int16_t int16 )
					   (int32_t int32
					    )
					   (int64_t int64 )
					   (float)
					   (double)
					   ("std::complex<float>" cfloat)
					   ("std::complex<double>" cdouble))
				collect
				(destructuring-bind (type &optional (short type) ) e
				  `(lambda (v)
				     (declare (type ,type v)
					      (capture ""))
				     ,(lprint :msg (format nil "~a thingy" type))))))))
	  (declare (type overload f))))

       (f 2)
       (f 2.0)
       (f 2d0)
       
       ))
   :omit-parens t
   :format t
   :tidy nil))

