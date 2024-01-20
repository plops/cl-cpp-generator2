(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more)))
  (setf *features* (set-exclusive-or *features* (list :more))))

(let ()
  (defparameter *source-dir* #P"example/140_halide/source01/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames "lesson01.cpp"
		     *source-dir*))
   `(do0
     (include "Halide.h")
     (include<> format)
     "using namespace Halide;"
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       (let ((gradient (Func))
	     (x (Var))
	     (y (Var))
	     (e (Expr (+ x y))))
	 (setf (gradient x y)
	       e)
	 (let ((output (Buffer<int32_t> (gradient.realize (curly 800 600))))))
	 (dotimes (j (output.height))
	   (dotimes (i (output.width))
	     (when (!= (output i j)
			 (+ i j))
	       ,(lprint :msg "error"
			:vars `(i j))
	       (return -1))))
	 ,(lprint :msg "success")
	 (return 0))))
   :omit-parens t
   :format t
   :tidy nil))

