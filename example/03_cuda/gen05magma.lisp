(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *code-file* (asdf:system-relative-pathname 'cl-cpp-generator2 "example/03_cuda/source/05magma_isamax.cpp"))
  (defun e (msg &key (void nil) data)
    (let ((res (gensym (string "res"))))
     (progn			       ; destructuring-bind (msg) cmds
       `(do0 
	 #+nil ,(format nil "constexpr(std::is_same<decltype(~a),void>::value)?~a:\"void\""
		  (emit-c :code msg)
		  (emit-c :code msg))
	 
	 (let ((,res ,(if void `(string "void") msg)))
	   ;; static_assert(!std::is_same<decltype(test_maxi()), void>::value, "void"); 
	   (<< "std::cout"
	       (- (dot ("std::chrono::high_resolution_clock::now")
		       (time_since_epoch)
		       (count))
		  g_start)
	       (string " ")
	       __FILE__
	       (string ":")
	       __LINE__
	       (string " ")
	       __PRETTY_FUNCTION__
	       (string ,(format nil " ~a: => " (emit-c :code msg)))
	       ,res
	       (string " ")
	       ,@(loop for e in data appending
		      `((string ,(format nil " ~a=" (emit-c :code e)))
			,e))
	       "std::endl"))))))
  (let* ((code
	  `(do0
	    "// https://developer.nvidia.com/sites/default/files/akamai/cuda/files/Misc/mygpu.pdf magma by example"
	    "// g++ -O3 -fopenmp -ggdb -march=native -std=c++11 -DHAVE_CUBLAS -I/opt/cuda/include -I/usr/local/magma/include -c -o 05magma_isamax.o 05magma_isamax.cpp; g++ -O3 -fopenmp -march=native -ggdb -L/opt/cuda/lib64 -L/usr/local/magma/lib -lm -lmagma -lopenblas -lcublas -lcudart -o 05magma_isamax 05magma_isamax.o; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/magma/lib/"
	    (include <cuda_runtime.h>
		     ;<device_launch_parameters.h>
		     <cstdlib> ;; randx
		     <cstdio>
		     <cmath>
		     <magma_v2.h>
					;<cassert>
		     <iostream>
		     <chrono>
		     )
	    
	    (defun main (argc argv)
	      (declare (values int)
		       (type int argc)
		       (type char** argv))
	      (let ((g_start (dot ("std::chrono::high_resolution_clock::now")
				  (time_since_epoch)
				  (count)))))
	      (magma_init)
	      (let ((queue (static_cast<magma_queue_t> NULL))
		    (dev (static_cast<magma_int_t> 0))
		    (m (static_cast<magma_int_t> 1024))
		    (a ))
		(declare (type float* a))
		,(e `(magma_queue_create dev &queue) :void t
		    :data `(dev))
		,(e `(cudaMallocManaged &a (* m (sizeof float)))
		    :data `(m a))
		(dotimes (j m)
		  ;,(e `(sinf (static_cast<float> j)))
		  (setf (aref a j) (sin (static_cast<float> j))))
		(let ((i (magma_isamax m a 1 queue)))
		  ,(e `(cudaDeviceSynchronize)))
		
		(do0
		 ,(e `(magma_free a))
		 ,(e `(magma_queue_destroy queue ) :void t)
		 ,(e `(magma_finalize)))
		(return 0))))))
    (write-source *code-file* code)))
 
