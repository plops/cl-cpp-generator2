(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)
;; https://developer.nvidia.com/gtc/2019/video/S9593/video ;; low level at 21:20
;; https://developer.download.nvidia.com/video/gputechconf/gtc/2019/video/S9593/s9593-cutensor-high-performance-tensor-operations-in-cuda.mp4
;; https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9593-cutensor-high-performance-tensor-operations-in-cuda-v2.pdf
;; http://on-demand.gputechconf.com/gtc/2018/presentation/s8854-cutlass-software-primitives-for-dense-linear-algebra-at-all-levels-and-scales-within-cuda.pdf
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
