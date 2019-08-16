(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)
;; https://developer.nvidia.com/gtc/2018/video/S8854/video ;; intro to cutlass c++ templates for gemm
;; https://developer.nvidia.com/gtc/2019/video/S9593/video ;; low level at 21:20
;; https://developer.download.nvidia.com/video/gputechconf/gtc/2019/video/S9593/s9593-cutensor-high-performance-tensor-operations-in-cuda.mp4
;; https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9593-cutensor-high-performance-tensor-operations-in-cuda-v2.pdf
;; http://on-demand.gputechconf.com/gtc/2018/presentation/s8854-cutlass-software-primitives-for-dense-linear-algebra-at-all-levels-and-scales-within-cuda.pdf
;; https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-wmma-st

;; in order to improve efficiency (28:45):
;; - access memory with 128bit
;; - prevent bank conflicts during memory stores and loads

;; work level structure
;; block
;; thread block tile
;; warp tile
;; thread tile

;; 4 mma instructions to cover 32-by-32-by-4 matrix multiply (30:14)
;; spatially interleaved to allow 128bit loads
;; bank conflict (31:40)
;;  - 16 byte access (128bit) in 4 phases, 8 threads per phase
;;  - bank conflicts only possible by groups of 8 threads at a time
;;  - more details in gtc 2018 s81006 thomas-collignon

;; global memory striped pattern (32:50)
;; int lane = threadIdx.x % 32
;; int c = lane % 8
;; int s = lane / 8
;; int gmem_offset = c + s * lda

;; permuted layout in shared mem (33:17)
;; int lane = threadIdx.x % 32
;; int c = lane % 8
;; int s = lane / 8
;; int smem_row = (c & 1) | ((c>>1) & 2)
;; int bank = ((c<<1) & 4) | s ^ smem_row       <-- parens around xor, how?
;; int smem_offset = smem_row * ldm_smem + bank

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
