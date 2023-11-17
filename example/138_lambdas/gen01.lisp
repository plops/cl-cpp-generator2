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
     (space template "<typename... Ts>"
	    struct "overload : Ts..."
	    (progn
	      (comments "operator of of all of these types so which means if you write using TS operator param paren dot dot it means that whatever types we give it.   you know if like the call operators of those types are going to be callable  directly through this overload object so it's kind of inheriting the call Operator so to say overload is an aggregate, no user defined constructor, no private members. elements of that aggregate are the base classes. initialize the overload object with aggregate initializaton using curly braces and give it a bunch of lambdas as base classes for the overload set. it will inherit the call operator from them. this behaves similar to typecase in common lisp")
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

