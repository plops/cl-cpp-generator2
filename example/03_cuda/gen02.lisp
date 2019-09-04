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
		     ;<device_launch_parameters.h>
		     <cstdlib>
		     <cassert>
		     ; <iostream>
		     )
	    "using namespace std;"
	    (defun vector_add (a b c n)
	      (declare (values "__global__ void")
		       (type int* a b c)
		       (type int n))
	      (let ((tid (+ (* blockDim.x blockIdx.x)
			    threadIdx.x)))
		(declare (type int tid))
		(when (< tid n)
		 (setf (aref c tid)
		       (+ (aref a tid)
			  (aref b tid))))))
	    (defun init_array (a n)
	      (declare (values void)
		       (type int* a)
		       (type int n))
	      (dotimes (i n)
		(setf (aref a i)
		      (% (rand) 100))))
	    (defun vector_add_cpu_assert (a b c n)
	      (declare (values void)
		       (type int* a b c)
		       (type int n))
	      (dotimes (i n)
		(assert (== (aref c i) (+ (aref a i)
				   (aref b i))))))
	    (defun main ()
	      (declare (values int))
	      (let ((n (<< 1 20))
		    (bytes (* n (sizeof int)))
		    a
		    b
		    c)
		(declare (type int* a b c)
			 ;(type int n)
			 ;x(type size_t bytes)
			 )
		,@(loop for e in `(a b c) collect
		       `(cudaMallocManaged ,(format nil "&~a" e) bytes))
		,@(loop for e in `(a b) collect
		       `(init_array ,e n))
		"// no communication between threads, so work splitting not critical"
		"// add padding"
		(let ((threads 256)
		      (blocks (/ (+ n (- threads 1)) threads)))
		  ,@(let ((n (expt 2 20)))
		      (loop for th in `(127 128 129 200 256 257 258) collect
			   (format nil "// n=~a threads=~a blocks=~a=~a"
				    n th (/ (+ n (- th 1))
					    th)
				    (floor (+ n (- th 1))
					   th))))
		  "// async kernel start"
		  ("vector_add<<<blocks, threads, 0, 0>>>" a b c n)
		  (cudaDeviceSynchronize)
		  (vector_add_cpu_assert a b c n)
		  (return 0)))))))
    (write-source *code-file* code)))
