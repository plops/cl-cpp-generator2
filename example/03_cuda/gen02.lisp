(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

;; https://www.youtube.com/watch?v=uUEHuF5i_qI
;; https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/vectorAdd/vector_add/vector_add/vector_add.cu

(progn
  (defparameter *code-file* (asdf:system-relative-pathname 'cl-cpp-generator2 "example/03_cuda/source/vector_add.cu"))
  (let* ((code
	  `(do0
	    (include <cuda_runtime.h>
		     <device_launch_parameters.h>
		     <cstdlib>)
	    (defun vector_add ()
	      (declare (values "__global__ void")))
	    (defun init_array (a n)
	      (declare (values void)
		       (type int* a)
		       (type int n))
	      (dotimes (i n)
		(setf (aref a i)
		      (% (rand) 100))))
	    (defun main ()
	      (declare (values int))
	      (let ((n (<< 1 20))
		    (bytes (* n (sizeof bytes)))
		    a
		    b
		    c)
		(declare (type int* a b c)
			 (type int n)
			 (type size_t bytes))
		,@(loop for e in `(a b c) collect
		       `(cudaMallocManaged ,(format nil "&~a" e) bytes))
		,@(loop for e in `(a b) collect
		       `(init_array ,e n))))
	    
	    )))
    (write-source *code-file* code)))
