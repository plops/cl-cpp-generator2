(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

;; ceres-solver.org/nnls_tutorial.html#introduction
;; https://github.com/ceres-solver/ceres-solver/blob/master/examples/helloworld.cc
;; sudo dnf install creres-solver-devel

(progn
  (defparameter *source-dir* #P"example/69_ceres/source/")
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
  (let ((type-definitions
	 `(do0
	   )))

    #+nil    (let ((fn-h (asdf:system-relative-pathname
			  'cl-cpp-generator2
			  (merge-pathnames #P"hello.h"
					   *source-dir*))))
	       (with-open-file (sh fn-h
				   :direction :output
				   :if-exists :supersede
				   :if-does-not-exist :create)
		 (emit-c :code
			 `(do0
			   (pragma once)
			   (include <tuple>
				    <mutex>
				    <thread>
		     		    <iostream>
				    <iomanip>
				    <chrono>
				    <memory>
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
		   (merge-pathnames #P"hello_template.h"
				    *source-dir*))
		  `(do0
		    (pragma once)
		    (defclass+ CostFunctor ()
		      "public:"
		      (defmethod "operator()" (x residual)
			(declare
			 (type "const T* const" x)
			 (type "T*" residual)
			 (template "typename T")
			 (const)
			 (values bool))
			(setf (aref residual 0)
			      (- 10.0d0 (aref x 0)))
			(return true)))
		    #+nil (defclass+ F4 ()
			    "public:"
			    (defmethod "operator()" (x1 x4 residual)
			      (declare
			       (type "const T* const" x1 x4)
			       (type "T*" residual)
			       (template "typename T")
			       (const)
			       (values bool))
			      (setf (aref residual 0)
				    (* (sqrt 10d0)
				       (- (aref x1 0)
					  (aref x4 0))
				       (- (aref x1 0)
					  (aref x4 0))))
			      (return true)))
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
		      (let ((initial_x 5.0d0)
			    (x initial_x)
			    (problem (Problem))
			    (*cost_function (new ("AutoDiffCostFunction<CostFunctor, 1, 1>"
						  (new CostFunctor)))))
			(problem.AddResidualBlock cost_function
						  nullptr &x)
			(let ((options (Solver--Options))
			      (summary (Solver--Summary)))
			  (setf options.minimizer_progress_to_stdout true)
			  (Solve options &problem &summary)
			  ,(lprint :vars `(initial_x x (summary.BriefReport))))
			)
		      (return 0)
		      )
		    )))

  (with-open-file (s "source/CMakeLists.txt" :direction :output
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
	     (directory "source/hello.cpp"))

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



