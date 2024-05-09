(eval-when (:compile-toplevel :execute :load-toplevel)
     (ql:quickload "cl-cpp-generator2")
     ;(ql:quickload "cl-ppcre")
     ;(ql:quickload "cl-change-case")
     ) 
(in-package :cl-cpp-generator2)
(progn
  (defparameter *source-dir* #P"example/61_cuda_code/source/")
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  ;; FIXME: this might need rework because i use int a{0} now instead of int a = 0
  (let ((type-definitions
	  `(do0
	    (defclass Points ()
	      "float *m_x,*m_y;"
	      "public:"
	      (defmethod Points ()
		(declare
		 (values "__host__ __device__")
		 (construct (m_x NULL)
			    (m_y NULL))))
	      (defmethod Points (x y)
		(declare
		 (type float* x y)
		 (values "__host__ __device__")
		 (construct (m_x x)
			    (m_y y))))
	      (defmethod get_point (idx)
		(declare
		 (type int idx)
		 (const)
		 (values "__host__ __device__ __forceinline__ float2"))
		(return (make_float2 (aref m_x idx)
				     (aref m_y idx))))
	      (defmethod set_point (idx p)
		(declare
		 (type int idx)
		 (type "const float2&" p)
		 (values "__host__ __device__ __forceinline__ void"))
		(setf  (aref m_x idx) p.x
		       (aref m_y idx) p.y))
	      (defmethod set (x y)
		(declare
		 (type float* x y)
			
		 (values "__host__ __device__ __forceinline__ void"))
		(setf  m_x x
		       m_y y)))

	    (defclass Bounding_box ()
	      "float2 m_p_min, m_p_max;"
	      "public:"
	      (defmethod Bounding_box ()
		(declare
		 (values "__host__ __device__")
		 #+nil (construct (m_p_min (make_float2 0s0 0s0))
				  (m_p_max (make_float2 1s0 1s0))
				  ))
		(setf m_p_min (make_float2 0s0 0s0)
		      m_p_max (make_float2 1s0 1s0)
		      ))
	      (defmethod compute_center (center)
		(declare
		 (type float2& center)
		 (const)
		 (values "__host__ __device__ void")
		 )
		,@(loop for e in `(x y)
			collect
			`(setf (dot center ,e)
			       (* .5 (+ (dot m_p_min ,e)
					(dot m_p_max ,e))))))
	      ,@(loop for e in `(min max)
		      collect
		      `(defmethod ,(format nil "get_~a" e) ()
			 (declare
			  (type float2& center)
			  (const)
			  (values "__host__ __device__ __forceinline__ const float2&"))
			 (return ,(format nil "m_p_~a" e))))
		     
	      (defmethod contains (p)
		(declare
		 (type "const float2&" p)
		 (const)
		 (values "__host__ __device__ bool"))
		(return
		  (and ,@(loop for e in `(x y)
			       appending
			       `((<= (dot m_p_min ,e)
				     (dot p ,e))
				 (<= (dot p ,e)
				     (dot m_p_max ,e)
				     )
				 )))))

	      (defmethod set (xmin ymin xmax ymax)
		(declare
		 (type float xmin ymin xmax ymax)
		 (values "__host__ __device__ void"))
		,@(loop for e in `(min max)
			appending
			(loop for f in `(x y)
			      collect
			      `(setf ,(format nil "m_p_~a.~a" e f)
				     ,(format nil "~a~a" f e))))))

	    (defclass Quadtree_node ()
	      "int m_id, m_begin, m_end;"
	      "Bounding_box m_bounding_box;"
	      "public:"
	      (defmethod Quadtree_node ()
		(declare
		 (values "__host__ __device__")
		 (construct (m_id 0)
			    (m_begin 0)
			    (m_end 0))))
	      
	      ,@(loop for e in `((:name id :params () :param-types () :const t :code  (return m_id) :values "__host__ __device__ int")
				 (:name set_id :params (new_id) :param-types (int) :const nil :code  (setf m_id new_id) :values "__host__ __device__ void")
				 (:name bounding_box :params () :param-types () :const t :code  (return m_bounding_box) :values "__host__ __device__ __forceinline__ const Bounding_box")
				 (:name set_bounding_box :params (xmin ymin xmax ymax) :param-types (float float float float) :const nil :code  (m_bounding_box.set   xmin ymin xmax ymax) :values "__host__ __device__ __forceinline__ void")
				 (:name num_points :params () :param-types () :const t :code  (return (- m_end m_begin)) :values "__host__ __device__ __forceinline__ int")
				 (:name points_begin :params () :param-types () :const t :code  (return m_begin) :values "__host__ __device__ __forceinline__ int")
				 (:name points_end :params () :param-types () :const t :code  (return m_end) :values "__host__ __device__ __forceinline__ int")
				 (:name set_range :params (begin end) :param-types (int int) :const nil :code  (setf m_begin begin
														     m_end end)
				  :values "__host__ __device__ __forceinline__ void"))
		      collect
		      (destructuring-bind (&key name params param-types const code values) e
			`(defmethod ,name ,params
			   (declare ,@(loop for param in params and ty in param-types
					    collect
					    `(type ,ty ,param))
				    (values ,values)
				    ,(if const
					 `(const)
					 `())
				    )
			   ,code
			   ))))
	    (defclass Parameters ()
	      "public:"
	      "int m_point_selector, m_num_nodes_at_this_level, m_depth;"
	      "const int m_max_depth, m_min_points_per_node;"
	       
	       
	      (defmethod Parameters (max_depth min_points_per_node)
		(declare
		 (type int max_depth min_points_per_node)
		 (values "__host__ __device__")
		 (construct (m_point_selector 0)
			    (m_num_nodes_at_this_level 1)
			    (m_depth 0)
			    (m_max_depth max_depth)
			    (m_min_points_per_node min_points_per_node))))
	      (defmethod Parameters (params flag)
		(declare
		 (type "const Parameters&" params)
		 (type bool flag)
		 (values "__host__ __device__")
		 (construct (m_point_selector (% (+ params.m_point_selector 1)
						 2))
			    (m_num_nodes_at_this_level (* 4 params.m_num_nodes_at_this_level))
			    (m_depth (+ 1 params.m_depth))
			    (m_max_depth params.m_max_depth)
			    (m_min_points_per_node params.m_min_points_per_node))))
	      )
	    (defclass Random_generator ()
	      "public:"
	      "int count;"
	      
	      (defmethod Random_generator ()
		(declare
		 (values "__host__ __device__")
		 (construct (count 0))))
	      (defmethod hash (a)
		(declare (type "unsigned int" a)
			 (values "__host__ __device__ unsigned int"))
		,@(loop for e in `((+ #x7ed55d16 + << 12)
				   (^ #xc761c23c ^ >> 19)
				   (+ #x165667b1 + << 5)
				   (+ #xd3a2646c ^  << 9)
				   (+ #xfd7046c5 + << 3)
				   (^ #xb55a4f09 ^ >> 16))
			collect
			(destructuring-bind (op val op2 sh len) e
			  `(setf a (,op2 (,op a ,(format nil "0b~32,'0b" val))
					 (,sh a ,len)))))
		(return a))
	      (defmethod "operator()" ()
		(declare (values "__host__ __device__ __forceinline__ thrust::tuple<float, float>"))
		"#ifdef __CUDA_ARCH__"
		(let (
		      (seed (hash (+ count
				     threadIdx.x
				     (* blockDim.x blockIdx.x)
				     ))))
		  (incf count (* blockDim.x gridDim.x)))
		"#else"
		(let ((seed (hash 0))))
		"#endif"
		"thrust::default_random_engine rng(seed);"
		"thrust::random::uniform_real_distribution<float> distrib;"
		(return (thrust--make_tuple (distrib rng)
					    (distrib rng))))
	      
	      ))))

    (let ((fn-h (asdf:system-relative-pathname
		 'cl-cpp-generator2
		 (merge-pathnames #P"quadtree.h"
				  *source-dir*))))
      (with-open-file (sh fn-h
			  :direction :output
			  :if-exists :supersede
			  :if-does-not-exist :create)
	(emit-c :code
		`(do0
		  "#pragma once"
		  (include <thrust/random.h>
			   <thrust/device_vector.h>
			   <thrust/host_vector.h>
			   <cooperative_groups.h>

			   )
		  ,type-definitions)
		:hook-defun #'(lambda (str)
				(format sh "~a~%" str))
		:hook-defclass #'(lambda (str)
                                   (format sh "~a;~%" str))
		:header-only t))
      (sb-ext:run-program "/usr/bin/clang-format"
                          (list "-i"  (namestring fn-h))))
    
    (write-source (asdf:system-relative-pathname
		   'cl-cpp-generator2
		   (merge-pathnames #P"quadtree.cu"
				    *source-dir*))
		  `(do0
		    (include "quadtree.h")
		   
		    ,type-definitions

		    "namespace cg = cooperative_groups;"
		    (include <helper_cuda.h>)
		    (defun build_quadtree_kernel (nodes
						  points
						  params)
		      (declare (type Quadtree_node* nodes)
			       (type Points* points)
			       (type Parameters params)
			       (values "template <int NUM_THREADS_PER_BLOCK> __global__ void"))
		      (let ((cta (cg--this_thread_block))
			    (NUM_WARPS_PER_BLOCK (/ NUM_THREADS_PER_BLOCK
						    warpSize))
			    (warp_id (/ threadIdx.x warpSize))
			    (lane_id (% threadIdx.x warpSize))
			    (lane_mask_lt (- (<< lane_id) 1))
			    (smem[])
			    (s_num_pts)
			    (&node (aref nodes blockIdx.x))
			    (num_points (node.num_points)))
			(declare (type "extern __shared__ int" smem[])
				 (type (array "volatile int*" 4) s_num_pts))
			"#pragma unroll"
			(dotimes (i 4)
			  (setf (aref s_num_pts i)
				(cast "volatile int*"
				      (ref (aref smem (* i NUM_WARPS_PER_BLOCK))))))
			(do0
			 (comments "stop recursion here, points[0] contains all points")
			 (let ((center)
			       (range_begin)
			       (range_end)
			       (warp_counts (curly 0 0 0 0)))
			   (declare (type float2 center)
				    (type int range_begin range_end)
				    (type (array int 4) warp_counts))
			   (when (or (<= params.m_max_depth
					 params.m_depth)
				     (<= num_points
					 params.m_min_points_per_node))
			     (when (== 1 params.m_point_selector)
			       (let ((it (node.points_begin))
				     (end (node.points_end)))
				 "#pragma unroll"
				 (for ((incf it threadIdx.x)
				       (< it end)
				       (incf it NUM_THREADS_PER_BLOCK))
				      (when (< it end)
					(dot (aref points 0)
					     (set_point it
							(dot (aref points 1)
							     (get_point it)))))
				      )))
			     (return))))
			(do0
			 (comments "find number of points in each warp, and points to move to each quadrant")
			 (let ((bbox (node.bounding_box)))
			   (declare (type "const Bounding_box&" bbox))
			   (bbox.compute_center center)
			   (let ((num_points_per_warp
				   (max warpSize
					(/ (+ num_points
					      NUM_WARPS_PER_BLOCK
					      -1)
					   NUM_WARPS_PER_BLOCK
					   )))
				 )
			     (setf range_begin (+ (node.points_begin)
						  (* warp_id num_points_per_warp))
				   range_end (min (+ range_begin
						     num_points_per_warp)
						  (node.points_end))))))
			(do0
			 (comments "count points in each child")
			 (let ((&in_points (aref points params.m_point_selector))
			       (tile32 (cg--tiled_partition<32> cta)))
			   "#pragma unroll"
			   (for ((= "int range_it"
				    (+ range_begin
				       (tile32.thread_rank)))
				 (tile32.any (< range_it
						range_end))
				 (incf range_it warpSize))
				(let ((is_active (< range_it
						    range_end))
				      (p (? is_active
					    (in_points.get_point range_it)
					    (make_float2 0s0 0s0)))
				      )
				  ,@(loop for flag in `((and (< p.x center.x) (<= center.y p.y))
							(and (<= center.x p.x) (<= center.y p.y))
							(and (< p.x center.x) (< p.y center.y))
							(and (<= center.x p.x) (< p.y center.y)))
					  and i from 0
					  collect
					  `(progn
					     (let ((num_pts (__popc (tile32.ballot
								     (and is_active ,flag)
								     ))))
					       (incf (aref warp_counts ,i)
						     (tile32.shfl num_pts 0)))))
				  ))
			   (when (== 0 (tile32.thread_rank))
			     ,@(loop for i below 4 collect
				     `(setf (aref (aref s_num_pts ,i)
						  warp_id)
					    (aref warp_counts ,i))))
			   (cg--sync cta)
			   (do0
			    (comments "scan warps' results")
			    (when (< warp_id 4)
			      (let ((num_pts (? (< (tile32.thread_rank)
						   NUM_WARPS_PER_BLOCK)
						(aref (aref s_num_pts warp_id)
						      (tile32.thread_rank))
						0)))
				"#pragma unroll"
				(for ((= "int offset" 1)
				      (< offset NUM_WARPS_PER_BLOCK)
				      (*= offset 2))
				     (let ((n (tile32.shfl_up num_pts offset)))
				       (when (<= offset (tile32.thread_rank))
					 (incf num_pts n))))
				(when (< (tile32.thread_rank)
					 NUM_WARPS_PER_BLOCK)
				  (setf (aref (aref s_num_pts
						    warp_id)
					      (tile32.thread_rank))
					num_pts)))))
			   (cg--sync cta)
			   (do0
			    (comments "global offset")
			    (when (== 0 warp_id)
			      (let ((sum (aref s_num_pts 0
					       (- NUM_WARPS_PER_BLOCK 1))))
				(for ((= "int row" 1)
				      (< row 4)
				      (incf row))
				     (let ((tmp (aref s_num_pts
						      row (- NUM_WARPS_PER_BLOCK 1)) ))
				       (cg--sync tile32)
				       (when (< (tile32.thread_rank)
						NUM_WARPS_PER_BLOCK)
					 (incf
					  (aref s_num_pts
						row
						(tile32.thread_rank))
					  sum))
				       (cg--sync tile32)
				       (incf sum tmp))))))

			   (cg--sync cta)
			   (do0
			    (comments "scan exclusive")
			    (let ((val 0))
			      (when (< threadIdx.x
				       (* 4 NUM_WARPS_PER_BLOCK))
				(setf val (? (== 0 threadIdx.x)
					     0
					     (aref smem (- threadIdx.x 1))))
				(incf val (node.points_begin)))))
			   (cg--sync cta)
			   (do0
			    (comments "move points")
			    (unless (or (<= params.m_max_depth
					    params.m_depth)
					(<= num_points
					    params.m_min_points_per_node))
			      (let ((&out_points
				      (aref points (% (+ params.m_point_selector 1)
						      2))))
				,@(loop for i below 4
					collect
					`(setf (aref warp_counts ,i)
					       (aref s_num_pts ,i warp_id)))
				(let ((&in_points (aref points params.m_point_selector)))
				  (do0
				   (comments "reorder points")
				   (for ((= "int range_it"
					    (+ range_begin
					       (tile32.thread_rank))
					    )
					 (tile32.any (< range_it
							range_end))
					 (incf range_it warpSize))
					(let ((is_active (< range_it range_end))
					      (p (? is_active
						    (in_points.get_point range_it)
						    (make_float2 0s0 0s0))))
					  ,@(loop for pred in `((and (< p.x center.x)
								     (<= center.y p.y))
								(and (<= center.x p.x)
								     (<= center.y p.y))
								(and (< p.x center.x)
								     (< p.y center.y))
								(and (<= center.x p.x)
								     (< p.y center.y)))
						  and i from 0
						  collect
						  `(progn
						     (let ((pred (and is_active
								      ,pred))
							   (vote (tile32.ballot pred))
							   (dest (+ (aref warp_counts ,i)
								    (__popc (logand vote
										    lane_mask_lt)))))
						       (when pred
							 (out_points.set_point dest p))
						       (incf (aref warp_counts ,i)
							     (tile32.shfl (__popc vote)
									  0)))))
					  )))))))
			   (cg--sync cta)
			   (when (== 0 (tile32.thread_rank))
			     ,@(loop for i below 4
				     collect
				     `(setf (aref s_num_pts ,i warp_id)
					    (aref warp_counts ,i)))
			     )
			   (cg--sync cta)
			   (do0
			    (comments "launch new blocks")
			    (unless (or
				     (<= params.m_max_depth
					 params.m_depth)
				     (<= num_points
					 params.m_min_points_per_node))
			      (when (== (- NUM_THREADS_PER_BLOCK 1)
					threadIdx.x)
				(let ((*children
					(ref
					 (aref nodes
					       (- params.m_num_nodes_at_this_level
						  (logand (node.id)
							  ~3))
					       )))
				      (child_offset (* 4 (node.id)))
				      )
				  ,@(loop for i below 4
					  collect
					  `(dot (aref children (+ child_offset ,i))
						(set_id (+ ,i (* 4 (node.id))))))
				  (let ((&bbox (node.bounding_box))
					(&p_min (bbox.get_min)
						)
					(&p_max (bbox.get_max)))
				    (do0
				     (comments "set bboxes of the children")
				     ,@(loop for e in `((p_min.x center.y center.x p_max.y)
							(center.x center.y p_max.x p_max.y)
							(p_min.x p_min.y center.x center.y)
							(center.x p_min.y p_max.x center.y))
					     and i from 0
					     collect
					     `(dot (aref children (+ child_offset ,i))
						   (set_bounding_box ,@e))))
				    (do0
				     (comments "set ranges of children")
				     ,@ (let ((old `(node.points_begin)))
					  (loop for i below 4
					       
						collect
						(let ((new `(aref s_num_pts ,i   warp_id)))
						  (prog1
						      `(dot
							(aref children (+ ,i child_offset))
							(set_range ,old ,new))
						    (setf old new))))))
				    (do0
				     (comments "launch children")
				     ("build_quadtree_kernel<NUM_THREADS_PER_BLOCK><<<4,NUM_THREADS_PER_BLOCK,4*NUM_WARPS_PER_BLOCK*sizeof(int)>>>"
				      (ref (aref children child_offset))
				      points
				      (Parameters params true)))
				    )))))
			   ))
			)
		      )
		    (defun cdpQuadTree (warp_size)
		      (declare (type int warp_size)
			       (values bool))
		      (let ((num_points 1024)
			    (max_depth 8)
			    (min_points_per_node 16))
			,@(loop for e in `(x y)
				collect
				`(do0
				  ,@(loop for f in `(0 1)
					  collect
					  `(,(format nil "thrust::device_vector<float> ~a_d~a" e f) num_points ))))
			"Random_generator rnd;"
			(thrust--generate
			 (thrust--make_zip_iterator
			  (thrust--make_tuple
			   (x_d0.begin)
			   (y_d0.begin))
			 
			  )
			 (thrust--make_zip_iterator
			  (thrust--make_tuple
			   (x_d0.end)
			   (y_d0.end)))
			 rnd)
			(do0
			 (comments "host structures to analyze device structures")
			 "Points points_init[2];"
			 ,@(loop for i below 2
				 collect
				 `(dot (aref points_init ,i)
				       (set (thrust--raw_pointer_cast
					     (ref (aref ,(format nil   "x_d~a" i) 0)))
					    (thrust--raw_pointer_cast
					     (ref (aref ,(format nil   "y_d~a" i) 0)))))))

		        (do0
			 (comments "allocate memory to store points")
			 "Points *points;"
			 ,(let ((target `points)
				(init `points_init)
				(len `(* 2 (sizeof Points))))
			    `(do0 (checkCudaErrors (cudaMalloc
						    (cast "void**"
							  (ref ,target))
						    ,len))
				  (checkCudaErrors (cudaMemcpy
						    ,target
						    ,init
						    ,len
						    cudaMemcpyHostToDevice)))))
			(do0
			 (let ((max_nodes 0)
			       (num_nodes_at_level 1))
			   (dotimes (i max_depth)
			     (incf max_nodes
				   num_nodes_at_level)
			     (*= num_nodes_at_level 4))
			   ))

			(do0
			 "Quadtree_node root,*nodes;"
			 (root.set_range 0 num_points)
			 ,(let ((target `nodes)
				(init `&root)
				(len `(* max_nodes (sizeof Quadtree_node)))
				(copy-len `(sizeof Quadtree_node)))
			    `(do0 (checkCudaErrors (cudaMalloc
						    (cast "void**"
							  (ref ,target))
						    ,len))
				  (checkCudaErrors (cudaMemcpy
						    ,target
						    ,init
						    ,copy-len
						    cudaMemcpyHostToDevice))))
			 )
			(do0
			 (comments "recursion limit to max_depth")
			 (cudaDeviceSetLimit cudaLimitDevRuntimeSyncDepth max_depth))
			(do0
			 (comments "build quadtree")
			 "Parameters params(max_depth,min_points_per_node);"
			 (let ((NUM_THREADS_PER_BLOCK 128)
			       (NUM_WARPS_PER_BLOCK (/ NUM_THREADS_PER_BLOCK warp_size))
			       (smem_size (* 4 NUM_WARPS_PER_BLOCK (sizeof int))))
			   (declare (type "const int" NUM_THREADS_PER_BLOCK
					  NUM_WARPS_PER_BLOCK)
				    (type "const size_t" smem_size))
			   ("build_quadtree_kernel<NUM_THREADS_PER_BLOCK><<<1,NUM_THREADS_PER_BLOCK,smem_size>>>"
			    nodes points params))
			 (checkCudaErrors (cudaGetLastError)))
			(return true)
			)

		      )
		    (defun main (argc argv)
		      (declare (type int argc)
			       (type char** argv)
			       (values int))
		      (let ((cuda_device (findCudaDevice argc
							 (cast "const char**" argv)))
			    (deviceProps)
			    )
			(declare (type cudaDeviceProp deviceProps))
			(checkCudaErrors
			 (cudaGetDeviceProperties &deviceProps
						  cuda_device))
			#+nil (let ((cdpCapable (or
						 (<= 4 deviceProps.major)
						 (and (== 3 deviceProps.major)
						      (<= 5 deviceProps.minor)
						      ))))
				)
			(let ((ok (cdpQuadTree deviceProps.warpSize))))
			)
		     
		      (return (? ok EXIT_SUCCESS
				 EXIT_FAILURE))
		      )
		    )))
  #+nil
  (with-open-file (s "source/CMakeLists.txt" :direction :output
					     :if-exists :supersede
					     :if-does-not-exist :create)
    (macrolet ((out (fmt &rest rest)
		 `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
      (out "cmake_minimum_required( VERSION 3.4 )")
      (out "project( mytest LANGUAGES CXX )")
      (out "set( CMAKE_CXX_COMPILER nvc++ )")
      (out "set( CMAKE_CXX_FLAGS \"-stdpar\"  )")
      (out "set( CMAKE_VERBOSE_MAKEFILE ON )")
      (out "set( CMAKE_CXX_STANDARD 17 )")
					;(out "set( CMAKE_CXX_COMPILER clang++ )")
      
					;(out "set( CMAKE_CXX_FLAGS )")
					;(out "find_package( Vulkan )")
      (out "set( SRCS ~{~a~^~%~} )"
	   (directory "source/*.cpp"))
      (out "add_executable( mytest ${SRCS} )")
      (out "target_include_directories( mytest PUBLIC /home/martin/stage/cl-cpp-generator2/example/58_stdpar/source/ )")
      
					;(out "target_link_libraries( mytest PRIVATE vulkan )")
					;(out "target_precompile_headers( mytest PRIVATE vis_00_base.hpp )")
      )
    ))



