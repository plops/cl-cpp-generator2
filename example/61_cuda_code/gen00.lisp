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
			(construct (m_p_min (make_float2 .0 .0))
				   (m_p_max (make_float2 1.0 1.0))
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



