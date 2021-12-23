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
					    (aref m_y idx))))))))

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



