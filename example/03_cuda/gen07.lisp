(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

(progn
    (defun vkprint (call &optional rest)
      `(do0 ,call
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
	      __func__
	      (string ,(format nil " ~a " (emit-c :code call)))
	      ,@(loop for e in rest appending
		     `((string ,(format nil " ~a=" (emit-c :code e)))
		       ,e))
	      "std::endl")
	  )
      )
    (defun cuprint (call &optional rest)
      `(progn (let ((r ,call))
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
		  __func__
		  (string ,(format nil " ~a => " (emit-c :code call)))
		  r
		  (string " '")
		  (cudaGetErrorString r)
		  (string "' ")
		  ,@(loop for e in rest appending
			 `((string ,(format nil " ~a=" (emit-c :code e)))
			   ,e))
		  "std::endl"))
	      (assert (== cudaSuccess r))
	  )
      )
    (defun cuprint_ (call &optional rest)
      `(progn (let ((r ,call))
		(unless (== cudaSuccess r)
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
		     __func__
		     (string ,(format nil " ~a => " (emit-c :code call)))
		     r
		     (string " '")
		     (cudaGetErrorString r)
		     (string "' ")
		     ,@(loop for e in rest appending
			    `((string ,(format nil " ~a=" (emit-c :code e)))
			      ,e))
		     "std::endl"))
		(assert (== cudaSuccess r)))
	  )
    )

  (defparameter *code-file* (asdf:system-relative-pathname 'cl-cpp-generator2 "example/03_cuda/source/07prefixsum.cu"))
  (let* ((code
	  `(do0
	    	    
	    "// david kirk: programming massively parallel processors (third ed) p. 175 prefix sum" 
	    	    
	    (include  
	     <cassert>
	     <cstdio>
	     <iostream>
	     <cuda_runtime.h>
	     <chrono>)
	    
	    (let ((g_start (,(format nil "static_cast<~a>" (emit-c :code `(typeof (dot ("std::chrono::high_resolution_clock::now")
										       (time_since_epoch)
										       (count)))))
			     0))))


	    (defun sequential_scan (x y n)
	      (declare (type float* x y)
		       (type int n))
	      (let ((accum (aref x 0)))
		(setf (aref y 0) accum)
		(for ((= "int i" 1)
		      (< i n)
		      (incf i))
		     (incf accum (aref x i))
		     (setf (aref y i) accum))))

	    (defun kogge_stone_scan_kernel (x y n)
	      (declare (type float* x y)
		       (type in n))
	      )
	    
	    (defun main ()
	      (declare (values int))
	      (setf g_start (dot ("std::chrono::high_resolution_clock::now")
										       (time_since_epoch)
										       (count)))

	      (do0
	       (let ((n_cuda 0))
		 ,(cuprint `(cudaGetDeviceCount &n_cuda) `(n_cuda)))
	       ,(cuprint `(cudaSetDevice 0)))
	      (return 0)))))
    (write-source *code-file* code)))
