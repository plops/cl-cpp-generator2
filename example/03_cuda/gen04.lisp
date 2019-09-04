(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

;; https://www.youtube.com/watch?v=ga2ML1uGr5o
;; https://github.com/CoffeeBeforeArch/cuda_programming/tree/master/matrixMul/tiled_matrix_mul
;; https://github.com/CoffeeBeforeArch/from_scratch/blob/master/cacheTiling/matrix_mul.cu
(progn
  (defparameter *code-file* (asdf:system-relative-pathname 'cl-cpp-generator2 "example/03_cuda/source/04matmul.cu"))
  (let* ((code
	  `(do0
	    "// nvcc -o 04matmul 04matmul.cu"
	    "// nvprof 04matmul"
	    (include <cuda_runtime.h>
		     ;<device_launch_parameters.h>
		     <cstdlib> ;; randx
		     <cassert>
		     ; <iostream>
		     )
					; "using namespace std;"
	    "#define SHM_SIZE (16*16)"
	    " "
	    (defun matrix_mul (a b c n)
	       (declare (values "__global__ void")
			(type int* a b c)
			(type int n))
	       (let (((aref A SHM_SIZE) (curly 0))
		     ((aref B SHM_SIZE) (curly 0))
		     (col (+ (* blockDim.x blockIdx.x)
			     threadIdx.x))
		     (row (+ (* blockDim.y blockIdx.y)
			     threadIdx.y))
		     (tx threadIdx.x)
		     (ty threadIdx.y)
		     (dim blockDim.x)
		     (sum 0))
		 (declare (type int row col sum)
			  (type "__shared__ int" (aref A SHM_SIZE)
				(aref B SHM_SIZE)))
		 "// move tile across length of grid"
		 (dotimes (i (/ (+ n dim -1)
				dim))
		   (setf (aref A (+ tx (* dim ty))) (aref a (+ (* i dim) tx (* row n)))
			 (aref B (+ tx (* dim ty))) (aref b (+ (* i dim n) (* ty n) col)))
		   (__syncthreads)
		   "// accumulate partial results"
		   (dotimes (j dim)
		     (incf sum (* (aref A (+ (* ty  dim) j))
				  (aref B (+ (* j dim) tx)))))
		   (__syncthreads))

		 (setf (aref c (+ col (* row n))) sum)))
	    (defun matrix_mul_cpu_assert (a b c n)
	      (declare (values void)
		       (type int* a b c)
		       (type int n))
	      (let ((tmp 0))
		(declare (type int tmp))
		"// every row i"
		(dotimes (i n)
		  "// every column j"
		  (dotimes (j n)
		    "// every row-col pair"
		    (setf tmp 0)
		    (dotimes (k n)
		      (incf tmp (* (aref a (+ k (* i n)))
				   (aref b (+ j (* k n))))))
		    (assert (== tmp (aref c (+ j (* i n)))))))))
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
		"// one thread per output element"
		"// square thread blocks"
		(let ((threads 16)
		      (blocks (/ (+ n (- threads 1)) threads)))
		  ,@(let ((n (expt 2 10)))
		      (loop for th in `(14 16 32) collect
			   (format nil "// n=~a threads=~a blocks=~a=~a"
				    n th (/ (+ n (- th 1))
					    th)
				    (floor (+ n (- th 1))
					   th))))
		  "// kernel launch parameters"
		  (let ((threads2 (dim3 threads threads))
			(blocks2 (dim3 blocks blocks)))
		   "// async kernel start"
		   ("matrix_mul<<<blocks2, threads2, 0, 0>>>" a b c n))
		  "// managed memory need explicit sync"
		  (cudaDeviceSynchronize)
		  (matrix_mul_cpu_assert a b c n)
		  (return 0)))))))
    (write-source *code-file* code)))
 
