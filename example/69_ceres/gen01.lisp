(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

;; ceres-solver.org/nnls_tutorial.html#introduction
;; https://github.com/ceres-solver/ceres-solver/blob/master/examples/helloworld.cc
;; sudo dnf install creres-solver-devel

(progn
  (defparameter *source-dir* #P"example/69_ceres/source_powell/")
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun lprint (&key (msg "") (vars nil))
    #+nil `(comments ,msg)
    #-nil`(progn				;do0
	    " "
	    (do0				;let
	     #+nil ((lock (std--unique_lock<std--mutex> ,(g `_stdout_mutex)))
		    )

	     (do0
					;("std::setprecision" 3)
	      (<< "std::cout"
		  ;;"std::endl"
		  ("std::setw" 10)
		  (dot ("std::chrono::high_resolution_clock::now")
		       (time_since_epoch)
		       (count))
					;,(g `_start_time)

		  (string " ")
		  ("std::this_thread::get_id")
		  (string " ")
		  __FILE__
		  (string ":")
		  __LINE__
		  (string " ")
		  __func__
		  (string " ")
		  (string ,msg)
		  (string " ")
		  ,@(loop for e in vars appending
			  `(("std::setw" 8)
					;("std::width" 8)
			    (string ,(format nil " ~a='" (emit-c :code e)))
			    ,e
			    (string "'")))
		  "std::endl"
		  "std::flush")))))
  (let ((powell-def
	 `((:name F1 :params (x1 x2) :code (+ (aref x1 0)
					      (* 10d0 (aref x2 0))))
	   (:name F2 :params (x3 x4) :code (* (sqrt 5d0)
					      (- (aref x3 0)
						 (aref x4 0))))
	   (:name F3 :params (x2 x3) :code (* (- (aref x2 0)
						 (* 2d0 (aref x3 0)))
					      (- (aref x2 0)
						 (* 2d0 (aref x3 0)))))
	   (:name F4 :params (x1 x4) :code (* (sqrt 10d0)
					      (- (aref x1 0)
						 (aref x4 0))
					      (- (aref x1 0)
						 (aref x4 0)))))
	  ))
    (write-source (asdf:system-relative-pathname
		   'cl-cpp-generator2
		   (merge-pathnames #P"hello_template.h"
				    *source-dir*))
		  `(do0
		    (pragma once)
		    (include <math.h>)
		    ,@(loop for e in powell-def
			    collect
			    (destructuring-bind (&key name params code) e
			      `(defclass+ ,name ()
				 "public:"
				 (defmethod "operator()" (,@params residual)
				   (declare
				    (type "const T* const" ,@params)
				    (type "T*" residual)
				    (template "typename T")
				    (const)
				    (values bool))
				   (setf (aref residual 0)
					 ,code)
				   (return true)))))
		    ))

    (write-source (asdf:system-relative-pathname
		   'cl-cpp-generator2
		   (merge-pathnames #P"hello.cpp"
				    *source-dir*))
		  `(do0
		    (include "hello_template.h"
			     <ceres/ceres.h>
			     <glog/logging.h>)
		    (include ;<tuple>
					;  <mutex>
		     <thread>
		     <iostream>
		     <iomanip>
		     <chrono>
		     <math.h>
					;  <memory>
		     )
		    ,@(loop for e in `(AutoDiffCostFunction
				       CostFunction
				       Problem
				       Solve
				       Solver)
			    collect
			    (format nil "using ceres::~a;" e))
					;,type-definitions
		    (defun main (argc argv)
		      (declare (type int argc)
			       (type char** argv)
			       (values int))
		      (google--InitGoogleLogging (aref argv 0))
		      ,@(loop for e in `(3 -1 0 1)
			      and i from 1
			      collect
			      `(let ((,(format nil "x~a" i) ,(* 1d0 e))))
			      )
		      (let ((problem (Problem)))

			,@(loop for e in powell-def
				collect
				(destructuring-bind (&key name params code) e
				  `(problem.AddResidualBlock (new (,(format nil "AutoDiffCostFunction<~a, 1, 1, 1>" name)
								    (new ,name)))
							     nullptr ,@(loop for e in params collect
									     `(ref ,e)))))

			(let ((options (Solver--Options))
			      (summary (Solver--Summary)))
			  (setf options.minimizer_progress_to_stdout true
				options.max_num_iterations 100
				options.linear_solver_type ceres--DENSE_QR)
			  (Solve options &problem &summary)
			  ,(lprint :vars `(x1 x2 x3 x4 (summary.BriefReport))))
			)
		      (return 0)
		      )
		    )))

  (with-open-file (s "source_powell/CMakeLists.txt" :direction :output
		     :if-exists :supersede
		     :if-does-not-exist :create)
    ;;https://clang.llvm.org/docs/AddressSanitizer.html
    ;; cmake -DCMAKE_BUILD_TYPE=Debug -GNinja ..
    (let ((dbg "-ggdb -O1 -fno-omit-frame-pointer -fsanitize=address -fsanitize-address-use-after-return=always -fsanitize-address-use-after-scope "))
      (macrolet ((out (fmt &rest rest)
		   `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
	(out "cmake_minimum_required( VERSION 3.4 )")
	(out "project( mytest LANGUAGES CXX )")
	(out "set( CMAKE_CXX_COMPILER clang++ )")
					;(out "set( CMAKE_CXX_FLAGS \"\"  )")
	(out "set( CMAKE_VERBOSE_MAKEFILE ON )")
	(out "set (CMAKE_CXX_FLAGS_DEBUG \"${CMAKE_CXX_FLAGS_DEBUG} ~a \")" dbg)
	(out "set (CMAKE_LINKER_FLAGS_DEBUG \"${CMAKE_LINKER_FLAGS_DEBUG} ~a \")" dbg )
					;(out "set( CMAKE_CXX_STANDARD 23 )")
					;(out "set( CMAKE_CXX_COMPILER clang++ )")

		 			;(out "set( CMAKE_CXX_FLAGS )")
	(out "find_package( Qt5 5.9 REQUIRED Core Gui Widgets PrintSupport )")
	(out "set( SRCS ~{~a~^~%~} )"
	     (directory "source_powell/hello.cpp"))

	(out "add_executable( mytest ${SRCS} )")
	(out "target_compile_features( mytest PUBLIC cxx_std_17 )")
					;(out "target_include_directories( mytest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} /usr/local/include  )")
					;(out "target_link_directories( mytest PUBLIC /usr/local/lib )")
	(out "find_package ( Ceres REQUIRED ) ")
	(out "target_include_directories( mytest PRIVATE ${CERES_INCLUDE_DIRS} )")
	(out "target_link_libraries( mytest PRIVATE ${CERES_LIBRARIES} )")


	;; Core Gui Widgets PrintSupport Svg Xml OpenGL
					;(out "target_link_libraries( mytest PRIVATE Qt5::Core Qt5::Gui Qt5::PrintSupport Qt5::Widgets Threads::Threads JKQTPlotterSharedLib_ )")

					; (out "target_link_libraries ( mytest Threads::Threads )")
					;(out "target_precompile_headers( mytest PRIVATE vis_00_base.hpp )")
	))
    ))



