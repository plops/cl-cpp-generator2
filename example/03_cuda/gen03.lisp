(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

;; https://www.youtube.com/watch?v=DpEgZe2bbU0
;; https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/matrixMul/matrix_mul/matrix_mul/matrix_mul.cu

(progn
  (defparameter *code-file* (asdf:system-relative-pathname 'cl-cpp-generator2 "example/03_cuda/source/03matmul.cu"))
  (let* ((code
	  `(do0
	    (include <cuda_runtime.h>
		     ;<device_launch_parameters.h>
		     <cstdlib> ;; randx
		     <cassert>
		     ; <iostream>
		     )
	    "using namespace std;"
	    (defun matrix_mul (a b c n)
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
	    (defun init_matrix (a n)
	      (declare (values void)
		       (type int* a)
		       (type int n))
	      (dotimes (i (* n n))
		(setf (aref a i) (% (rand) 100)))
	      )
	    (defun main ()
	      (declare (values int))
	      "// 1024x1024 square matrix"
	      (let ((n (<< 1 10))
		    (bytes (* n n (sizeof int)))
		    a
		    b
		    c)
		(declare (type int* a b c))
		,@(loop for e in `(a b c) collect
		       `(cudaMallocManaged (ref ,e) bytes))
		,@(loop for e in `(a b) collect
		       `(init_matrix ,e n))
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
		  "// managed memory need explicit sync"
		  (cudaDeviceSynchronize)
		  (vector_add_cpu_assert a b c n)
		  (return 0)))))))
    (write-source *code-file* code)))
 
