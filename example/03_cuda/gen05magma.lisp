(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *code-file* (asdf:system-relative-pathname 'cl-cpp-generator2 "example/03_cuda/source/05magma_isamax.cpp"))
  (let* ((code
	  `(do0
	    "// https://developer.nvidia.com/sites/default/files/akamai/cuda/files/Misc/mygpu.pdf magma by example"
	    "// g++ -O3 -fopenmp -march=native -std=c++11 -DHAVE_CUBLAS -I/opt/cuda/include -I/usr/local/magma/include -c -o 05magma_isamax.o 05magma_isamax.cpp; g++ -O3 -fopenmp -march=native -L/opt/cuda/lib64 -L/usr/local/magma/lib -lmagma -lopenblas -lcublas -lcudart -o 05magma_isamax 05magma_isamax.o; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/magma/lib/"
	    (include <cuda_runtime.h>
		     ;<device_launch_parameters.h>
		     <cstdlib> ;; randx
		     <cstdio>
		     <magma_v2.h>
					;<cassert>
					;; <iostream>
		     )
	    (defun main (argc argv)
	      (declare (values int)
		       (type int argc)
		       (type char** argv))
	      (magma_init)
	      (let ((queue (cast magma_queue_t NULL))
		    (dev (magma_int_t 0))
		    (m (magma_int_t 1024))
		    (a ))
		(declare (type float* a))
		(magma_queue_create dev &queue)
		(cudaMallocManaged &a (* m (sizeof float)))
		(dotimes (j m)
		  (setf (aref a j) (sinf (float j))))
		(let ((i (magma_isamax m a 1 queue)))
		  (cudaDeviceSynchronize))
		(do0
		 (magma_free a)
		 (magma_queue_destroy queue)
		 (magma_finalize))
		(return 0))))))
    (write-source *code-file* code)))
 
