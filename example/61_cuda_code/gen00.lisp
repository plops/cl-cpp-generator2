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
			(construct (m_p_min (make_float2 0s0 0s0))
				   (m_p_max (make_float2 1s0 1s0))
				   )))
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
			 (&& ,@(loop for e in `(x y)
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
				 (:name point_end :params () :param-types () :const t :code  (return m_end) :values "__host__ __device__ __forceinline__ int")
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
	       "int m_point_selector, m_num_nodes_at_this_level, m_depth;"
	       "const int m_max_depth, m_min_points_per_node;"
	       
	      "public:"
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
		 (construct (m_point_selector (% (+ params.point_selector 1)
						 2))
			    (m_num_nodes_at_this_level (* 4 params.num_nodes_at_this_level))
			    (m_depth (+ 1 params.depth))
			    (m_max_depth params.max_depth)
			    (m_min_points_per_node params.min_points_per_node))))
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
			    <cooperative_groups.h>)
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
		       (dotimes (i 4)
			 (setf (aref s_num_pts i)
			       (cast "volatile int*"
				     (ref (aref smem (* i NUM_WARS_PER_BLOCK))))))
		       (do0
			(comments "stop recursion here, points[0] contains all points")
			(let ((center)
			      (range_begin)
			      (range_end)
			      (warp_counts (curly 0 0 0 0)))
			  (declare (type float2 center)
				   (type int range_begin range_end)
				   (type (array int 4) warp_counts))
			  (when (or (<= params.max_depth
					    params.depth)
					(<= num_points
					    params.min_points_per_node))
			    (when (== 1 params.point_selector)
			      (let ((it (node.points_begin))
				    (end (node.points_end)))
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
				(range_begin (+ (node.points_begin)
						(* warp_id num_points_per_warp)))
				(range_end (min (+ range_begin
						   num_points_per_warp)
						(node.points_end)))))))
		       (do0
			(comments "count points in each child")
			(let ((&in_points (aref points params.point_selector))
			      (tile32 (cg--tiled_partition<32> cta)))
			  (for ((= range_it
				   (+ range_begin
				      (tile32.thread_rank)))
				(tile32.any (< range_it
					       range_end))
				(incf range_it warpSize))
			       (let ((is_active (< range_it
						   range_end))
				     (p (? is_active
					   (in_points.get_points range_it)
					   (make_float2 0s0 0s0)))
				     )
				 ,@(loop for flag in `((and (< p.x center.x) (<= center.y p.y))
						       (and (<= center.x p.x) (<= center.y p.y))
						       (and (< p.x center.x) (< p.y center.y))
						       (and (<= center.x p.x) (< p.y center.y)))
					 and i from 0
					 collect
					 `(let ((num_pts (__popc (tile32.ballot
								  (and is_active ,flag)
								  ))))
					    (incf (aref warp_counts ,i)
						  (tile32.shfl num_pts 0))))
				 ))
			  (when (== 0 (tile32.thread_rank))
			    ,@(loop for i below 4 collect
				    `(setf (aref (aref s_num_pts ,i)
						 warp_id)
					   (aref warp_cnts ,i))))
			  (cg--sync cta)
			  (let ((val 0))
			    (when (< threadIdx.x (* 4 NUM_WARPS_PER_BLOCK))
			      (setf val (? (== 0 threadIdx.x)
					   0 (aref smem (- threadIdx.x 1))))
			      (incf val (node.points_begin))))
			  ))
		       )
		     )
		   
		   (defun main (argc argv)
		     (declare (type int argc)
			      (type char** argv)
			      (values int))
		     (return EXIT_SUCCESS)
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



