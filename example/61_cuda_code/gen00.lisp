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
					    ,(format nil "~a~a" f e)))))))))

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



