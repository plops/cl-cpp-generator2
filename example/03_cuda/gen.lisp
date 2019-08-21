(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

;; https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/

;; https://developer.nvidia.com/gtc/2018/video/S8854/video ;; intro to cutlass c++ templates for gemm
;; https://developer.nvidia.com/gtc/2019/video/S9593/video ;; low level at 21:20
;; https://developer.download.nvidia.com/video/gputechconf/gtc/2019/video/S9593/s9593-cutensor-high-performance-tensor-operations-in-cuda.mp4
;; https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9593-cutensor-high-performance-tensor-operations-in-cuda-v2.pdf
;; http://on-demand.gputechconf.com/gtc/2018/presentation/s8854-cutlass-software-primitives-for-dense-linear-algebra-at-all-levels-and-scales-within-cuda.pdf
;; https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9593-cutensor-high-performance-tensor-operations-in-cuda-v2.pdf
;; mma.sync.aligned.m8n8k4.alayout.blayout.dtype.f16.f16.ctyped, a, b, c;
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
;; // permutation function:
;; int smem_row = (c & 1) | ((c>>1) & 2)
;; int bank = ((c<<1) & 4) | s ^ smem_row       <-- parens around xor, how?
;; int smem_offset = smem_row * ldm_smem + bank

;; memory loads from 32 threads (36:17)

;; free fortran, c, c++ compiler:  https://www.pgroup.com/products/community.htm

;; https://arxiv.org/abs/1908.06649 A Computational Model for Tensor Core Units
(progn
  (defparameter *code-file* (asdf:system-relative-pathname 'cl-cpp-generator2 "example/03_cuda/source/shader.cu"))
  (let* ((code
	  `(do0
	    (include <wmma.h>)

	    "// independent dot products"
	    "// inefficient due to large working sets to hold parts of A and B"
	    (dotimes (i M)
	      (dotimes (j N)
		(dotimes (k K)
		  (incf (aref C i j)
			(* (aref A i k)
			   (aref B k j))))))

	    "// accumulate outer products"
	    "// load elements of A and B exactly once"
	    (dotimes (k K)
	      (dotimes (i M)
		(dotimes (j N)
		  (incf (aref C i j)
			(* (aref A i k)
			   (aref B k j))))))

	    "// partition into Mtile-by-Ntile independent matrix products"
	    (dotimes (mb M Mtile)
	      (dotimes (nb N Ntile)
		(dotimes (kb K Ktile)
		  (dotimes (k Ktile)
		    (dotimes (i Mtile)
		      (dotimes (j Ntile)
			(let ((row (+ mb i))
			      (col (+ nb j)))
			  (incf (aref C row col)
				(* (aref A row (+ kb k))
				   (aref B (+ kb k) col)))))))))
	      )
	    "// each warp computes independent matrix product"
	    (dotimes (mb M Mtile)
	      (dotimes (nb N Ntile)
		(dotimes (kb K Ktile)
		  "// load A and B tiles into shared memory"
		  (dotimes (m Mtile warp_m)
		    (dotimes (n Ntile warp_n)
		      (dotimes (k Ktile warp_k)
			"// compute warp_m by warp_n by warp_k GEMM"
			#+nil (let ((row (+ mb i))
			      (col (+ nb j)))
			  (incf (aref C row col)
				(* (aref A row (+ kb k))
				   (aref B (+ kb k) col)))))))))
	      )
	    "// accumulated matrix product in warps"
	    (dotimes (mb M Mtile)
	      (dotimes (nb N Ntile)
		(dotimes (kb K Ktile)
		  "// load A and B tiles into shared memory"
		  (dotimes (m Mtile warp_m)
		    (dotimes (n Ntile warp_n)
		      (dotimes (k Ktile warp_k)
			"// load A and B tile from SMEM into registers"
			(dotimes (tm warp_m thread_m)
			  (dotimes (tn warp_n thread_n)
			    (dotimes (tk warp_k thread_k)
			      "// compute thread_m by thread_n by thread_k GEMM")))))))))
	    "// threads compute accumulated matrix product"
	    "// A,B and C held in registers"
	    "// O(M*N) computations on O(M+N) elements"
	    "// opportunity for data reuse"
	    (dotimes (mb M Mtile)
	      (dotimes (nb N Ntile)
		(dotimes (kb K Ktile)
		  "// load A and B tiles into shared memory"
		  (dotimes (m Mtile warp_m)
		    (dotimes (n Ntile warp_n)
		      (dotimes (k Ktile warp_k)
			"// load A and B tile from SMEM into registers"
			(dotimes (tm warp_m thread_m)
			  (dotimes (tn warp_n thread_n)
			    (dotimes (tk warp_k thread_k)
			      (dotimes (m thread_m)
				(dotimes (n thread_n)
				  (dotimes (k thread_k)
				    "// FMA instructions"
				    (incf (aref C m n)
					  (* (aref A m k)
					     (aref B n k)))))))))))))))
	    "// warp output tiles have to be accessed once per math operation"
	    "// store in register file to enable fastest access from 32 threads in warp"
	    "// warps in the same row load the same data from A"
	    "// warps in the same column load the same data from B"
	    "// call __syncthreads() as appropriate to ensure warp-synchronous execution"
	    "// threads can't access each others registers."
	    "// organize data such that an individual thread can reuse data as much as possible"
	    "// each thread computes an outer product"
	    "// threads in the same row of a warp fetch the same data from A"
	    "// threads in the same column of a warp fetch the same data from B"
	    "// wmma api is an alternative to the thread tile structure"
	    "// each tensor core can process a 4x4x4 operation with matrices D=A.B+C (A,B fp16; D,C fp16 or fp32)"
	     (dotimes (mb M Mtile)
	      (dotimes (nb N Ntile)
		(dotimes (kb K Ktile)
		  "// load A and B tiles into shared memory"
		  (dotimes (m Mtile warp_m)
		    (dotimes (n Ntile warp_n)
		      (defun block_matrix_product ()
			(declare (values "__device__ void"))
			(let (frag_a
			      frag_b
			      accum)
			  (declare (type (array fp16 THREAD_ITEMS_X) frag_a)
				   (type (array fp16 THREAD_ITEMS_Y) frag_b)
				   (type (array fp32 THREAD_ITEMS_X THREAD_ITEMS_Y) accum))
			    "// load A and B tile from SMEM into registers"
			  (dotimes (kblock  Kdim  BlockItemsK)
			    (__syncthreads)
			    "#pragma unroll"
			    (dotimes (warp_k BlockItemsK WarpItemsK)
			      "// fetch frag_a and frag_b from SMEM corresponding to k index"
			      "// accumulate an outer product"
			      "#pragma unroll"
			      (dotimes (thread_x ThreadItemsX)
				"#pragma unroll"
				(dotimes (thread_y ThreadItemsY)
				  (incf (aref accum thread_x thread_y)
					(* (aref frag_a y)
					   (aref frag_b x))))))
			    (__syncthreads)))))))))
	    
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
