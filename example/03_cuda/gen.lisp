(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *code-file* (asdf:system-relative-pathname 'cl-cpp-generator2 "example/03_cuda/source/shader.cu"))
  (let* ((code
	  `(do0
	    (include <wmma.h>)
	    (defun tensor_op_16_16_16 (d a b c)
	      (declare (values "__device__ void")
		       (type float* d c)
		       (type half* a b))
	      (let ((Amat)
		    (Bmat)
		    (Cmat))
		(declare (type
			  "wmma::fragment<matrix_a, ...>" Amat)
			 (type
			  "wmma::fragment<matrix_b, ...>" Bmat))
		("wmma::load_matrix_sync" Amat a 16)
		("wmma::load_matrix_sync" Bmat b 16)
		("wmma::fill_fragment" Cmat .0s)
		("wmma::mma_sync" Cmat Amat Bmat Cmat)
		("wmma::store_matrix_sync" d Cmat 16 "wmma::row_major")))
	    )))
    (write-source *code-file* code)))
