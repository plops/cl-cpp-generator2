(eval-when (:compile-toplevel :execute :load-toplevel)
	   (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

;; ceres-solver.org/nnls_tutorial.html#introduction
;; sudo dnf install creres-solver-devel

;; lerp http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p0811r3.html
;; https://stackoverflow.com/questions/67577625/c-efficient-interpolation-of-a-stdvector

;; how to plot?
;; jkqtplotter  (not in fedora)
;; sudo dnf install qcustomplot-qt5-devel qcustomplot-qt5

;; https://github.com/hazelnusse/cpp-plotter/tree/master/includeb
;; https://doc.qt.io/qt-5/cmake-manual.html

;; https://github.com/filipecalasans/realTimePlot  based on qcustomplot
;; implot

(progn
  (defparameter *source-dir* #P"example/69_ceres/source_03spline_curve/")
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
  (let ((params (loop for e below 5 collect
		      (make-symbol (string-upcase (format nil "x~a" e))))))
    (write-source (asdf:system-relative-pathname
		   'cl-cpp-generator2
		   (merge-pathnames #P"hello_template.h"
				    *source-dir*))
		  `(do0
		    (pragma once)
		    (include <cmath>
			     <ceres/ceres.h>
					;<ceres/cubic_interpolation.h>
			     )
		    ,@(loop for e in `(AutoDiffCostFunction
				       CostFunction
				       Problem
				       Solve
				       Solver
					; Grid1D
					; CubicInterpolator
				       )
			    collect
			    (format nil "using ceres::~a;" e))

		    (defclass+ ExponentialResidual ()
		      "public:"
		      "const double x_;"
		      "const double y_;"
		      (defmethod ExponentialResidual (x y)
			(declare (type double x y)
				 (construct (x_ x)
					    (y_ y))
				 (values :constructor)))
		      (defmethod "operator()" (x0 residual)
			(declare
			 (type "const T* const" x0)
			 (type "T*" residual)
			 (template "typename T")
			 (const)
			 (values bool))

			(do0
			 ;; https://github.com/ceres-solver/ceres-solver/blob/master/include/ceres/autodiff_cost_function.h
			 (comments "piecewise linear interpolation of equidistant points")
			 (let (;(data (curly ,@params))
			       (mi 0d0)
			       (ma 5d0)
			       (N ,(length params))
			       (xrel (/ (- x_ mi)
					(- ma mi)))
			       (xpos (* xrel (- N 2)))
			       (lo_idx (int xpos))
			       (tau (- xpos lo_idx)) ;; is zero when interpolation asks for point at lo_idx
			       (hi_idx (+ 1 lo_idx))
			       )
			   #+nil (do0 ,(lprint :vars `(xrel xpos hi_idx))
				      (assert (<= hi_idx ,(- (length params) 1)))
				      (assert (<= lo_idx ,(- (length params) 2)))
				      (assert (<= 0 hi_idx))
				      (assert (<= 0 lo_idx)))
			   (let ((lo_val (aref x0 lo_idx))
				 (hi_val (aref x0 hi_idx))
				 (lerp (+ (* tau lo_val)
					  (* (- 1d0 tau)
					     (- hi_val lo_val)))))
			     (setf (aref residual 0)
				   (- y_ lerp)))))
			(return true)))

		    ))

    (let ((qt-code `(do0
		     (defclass MainWindow "public QMainWindow"
		       "Q_OBJECT"
		       "public:"

		       "QCustomPlot* plot_;"
		       "int graph_count_;"
		       (defmethod MainWindow (&key (parent 0))
			 (declare (type QWidget* parent)
				  (explicit)
				  (construct
				   (graph_count_ 0)
				   (QMainWindow parent)
					;(plot_ (curly (new (QCustomPlot this))))
				   )
				  (values :constructor))
			 (setf plot_ (new (QCustomPlot this)))
			 (setCentralWidget plot_)
			 (setGeometry 400 250 542 390))
		       (defmethod plot_scatter (x y)
			 (declare (type std--vector<double> x y))
			 (assert (== (x.size)
				     (y.size)))
			 "QVector<double> qx(x.size()),qy(y.size());"
			 (dotimes (i (x.size))
			   (setf (aref qx i) (aref x i)))
			 (dotimes (i (y.size))
			   (setf (aref qy i) (aref y i)))
			 (plot_->addGraph)
			 (-> plot_
			     (graph graph_count_)
			     (setData qx qy))
			 (-> plot_
			     (graph graph_count_)
			     (setLineStyle QCPGraph--lsNone)
			     )
			 (-> plot_
			     (graph graph_count_)
			     (setScatterStyle (QCPScatterStyle QCPScatterStyle--ssCircle 4))
			     )
			 (incf graph_count_))
		       (defmethod plot_line (x y)
			 (declare (type std--vector<double> x y))
			 (assert (== (x.size)
				     (y.size)))
			 "QVector<double> qx(x.size()),qy(y.size());"
			 (dotimes (i (x.size))
			   (setf (aref qx i) (aref x i)))
			 (dotimes (i (y.size))
			   (setf (aref qy i) (aref y i)))
			 (plot_->addGraph)
			 (-> plot_
			     (graph graph_count_)
			     (setData qx qy))
	       		 #+nil(-> plot_
				  (graph graph_count_)
				  (setLineStyle QCPGraph--lsNone)
				  )
			 #+nil (-> plot_
				   (graph graph_count_)
				   (setScatterStyle (QCPScatterStyle QCPScatterStyle--ssCircle 4))
				   )
			 (incf graph_count_))
		       (defmethod ~MainWindow ()
			 (declare
			  (values :constructor)))
		       ))))
      (let ((fn-h (asdf:system-relative-pathname
		   'cl-cpp-generator2
		   (merge-pathnames #P"gui.h"
				    *source-dir*))))
	(with-open-file (sh fn-h
			    :direction :output
			    :if-exists :supersede
			    :if-does-not-exist :create)
			(emit-c :code
				`(do0
				  (pragma once)
				  (include
					;<QEvent>

				   <iostream>
				   <iomanip>
				   <chrono>

				   )
				  (include <QApplication>
					   <QMainWindow>
					   <qcustomplot.h>)
				  ,qt-code)
				:hook-defun #'(lambda (str)
						(format sh "~a~%" str))
				:hook-defclass #'(lambda (str)
						   (format sh "~a;~%" str))
				:header-only t))
	(sb-ext:run-program "/usr/bin/clang-format"
                            (list "-i"  (namestring fn-h))))
      (write-source (asdf:system-relative-pathname
		     'cl-cpp-generator2
		     (merge-pathnames #P"gui.cpp"
				      *source-dir*))
		    `(do0
		      (include ;"gui.h"
		       "moc_gui.h")
		      ,qt-code)))

    (write-source (asdf:system-relative-pathname
		   'cl-cpp-generator2
		   (merge-pathnames #P"hello.cpp"
				    *source-dir*))
		  `(do0
		    (include "hello_template.h"
			     <ceres/ceres.h>
					;<ceres/cubic_interpolation.h>
			     <glog/logging.h>)
		    (include ;<tuple>
					;  <mutex>
		     <thread>
		     <iostream>
		     <iomanip>
		     <chrono>
		     <cmath>
		     <cassert>
					;  <memory>
		     )

		    (include <QApplication>
			     <QMainWindow>
			     <qcustomplot.h>)
		    (include "gui.h")
		    ,@(loop for e in `(AutoDiffCostFunction
				       CostFunction
				       Problem
				       Solve
				       Solver
					; Grid1D
					;CubicInterpolator
				       )
			    collect
			    (format nil "using ceres::~a;" e))
					;,type-definitions



		    (defun main (argc argv)
		      (declare (type int argc)
			       (type char** argv)
			       (values int))
		      (google--InitGoogleLogging (aref argv 0))
		      (do0
		       "QApplication app(argc,argv);"
		       "MainWindow w;")


		      ,@(loop for e in params
			      and i from 1
			      collect
			      `(let ((,(format nil "~a" e) ,(* 1d0 1))))
			      )
		      (let ((problem (Problem)))
			,(let* ((num-data 67)
				(x (loop for i below num-data collect
					 (* 5d0 (/ i (- num-data 1d0)))))
				(y (loop for e in x
					 collect
					 (+ (* .5 .2 (- (random 2d0) 1d0)) ;; note: this is uniform random, not gaussian
					    (exp (+ (* .3 e) .1))))))
			   `(do0
			     (let ((n ,num-data)
				   (data_x (std--vector<double> (curly ,@x)))
				   (data_y (std--vector<double> (curly ,@y)) ;(curly ,@y)
					   )
				   (params (curly ,@(loop for e in params collect 1d0))))

			       (declare ;(type (array double ,num-data) data_x data_y)
				(type (array double ,(length params)) params))
			       (w.plot_scatter data_x data_y)
			       (dotimes (i n)
				 (problem.AddResidualBlock
				  (new (,(format nil "AutoDiffCostFunction<ExponentialResidual,1,~a>"
						 (length params))
					(new (ExponentialResidual (aref data_x i)
								  (aref data_y i))
					     )))
				  nullptr
				  ;; (new (ceres--CauchyLoss .5d0))
				  params)))))

			(let ((options (Solver--Options))
			      (summary (Solver--Summary)))
			  (setf options.minimizer_progress_to_stdout true
				options.max_num_iterations 100
				options.linear_solver_type ceres--DENSE_QR)
			  (Solve options &problem &summary)
			  ,(lprint :vars `( (summary.BriefReport)))
			  (dotimes (i ,(length params))
			    ,(lprint :vars `((aref params i))))
			  (do0
			   (let ((fit_n 120)
				 (fit_x (std--vector<double>))
				 (fit_y (std--vector<double>)))
			     (dotimes (i fit_n)
			       (let (
				     (mi 0d0)
				     (ma 5d0)
				     (x_ (/ (* ma i) (- fit_n 1)))
				     (N ,(length params))
				     (xrel (/ (- x_ mi)
					      (- ma mi)))
				     (xpos (* xrel (- N 2)))
				     (lo_idx (int xpos))
				     (tau (- xpos lo_idx)) ;; is zero when interpolation asks for point at lo_idx
				     (hi_idx (+ 1 lo_idx))
				     )
				 #-nil (do0 ;,(lprint :vars `(xrel xpos hi_idx))
					(assert (<= hi_idx ,(- (length params) 1)))
					(assert (<= lo_idx ,(- (length params) 2)))
					(assert (<= 0 hi_idx))
					(assert (<= 0 lo_idx)))
				 (let ((lo_val (aref params lo_idx))
				       (hi_val (aref params hi_idx))
				       (lerp (+ (* tau lo_val)
						(* (- 1d0 tau)
						   (- hi_val lo_val)))))
				   (fit_x.push_back x_)
				   (fit_y.push_back lerp)))
			       )
			     (w.plot_line fit_x fit_y))))
			)
		      (do0
		       (do0

			(w.show))
		       #+nil
		       (do0 "QPushButton button(\"hello world\");"
			    (button.show)))

		      (return (app.exec))
		      )
		    )))
  (sb-ext:run-program "/usr/bin/sh"
                      (list (format nil "~a" (elt (directory "source_03spline_curve/generate_moc.sh" ) 0))) )
  (with-open-file (s "source_03spline_curve/CMakeLists.txt" :direction :output
		     :if-exists :supersede
		     :if-does-not-exist :create)
		  ;;https://clang.llvm.org/docs/AddressSanitizer.html
		  ;; cmake -DCMAKE_BUILD_TYPE=Debug -GNinja ..
		  (let ((dbg "-ggdb -O0 -fno-omit-frame-pointer -fsanitize=address -fsanitize-address-use-after-return=always -fsanitize-address-use-after-scope "))
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
					;(out "find_package( Qt5 5.9 REQUIRED Core Gui Widgets PrintSupport )")
			      (out "find_package( Qt5 COMPONENTS Core Gui Widgets PrintSupport REQUIRED )")
			      (out "set( SRCS ~{~a~^~%~} )"
				   (directory "source_03spline_curve/*.cpp"))

			      (out "add_executable( mytest ${SRCS} )")
			      (out "target_compile_features( mytest PUBLIC cxx_std_17 )")
					;(out "target_include_directories( mytest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} /usr/local/include  )")
					;(out "target_link_directories( mytest PUBLIC /usr/local/lib )")
			      (out "find_package ( Ceres REQUIRED ) ")
			      (out "find_package ( QCustomPlot REQUIRED ) ")
					;(out "find_package ( PkgConfig REQUIRED )")
					;(out "pkg_check_modules( QCP REQUIRED qcustomplot-qt5 )")
					;(out "qt5_generate_moc( ~{~a~^ ~} gui.moc TARGET mytest )" (directory "source_03spline_curve/gui.h"))
			      (out "target_include_directories( mytest PRIVATE ${CERES_INCLUDE_DIRS} )")
					; (out "target_link_libraries( mytest PRIVATE ${CERES_LIBRARIES} ${QCP_LIBRARIES} )")
					;(out "set_target_properties( Qt5::Core PROPERTIES MAP_IMPORTED_CONFIG_DEBUG \"RELEASE\" )")
					;(out "set( CMAKE_AUTOMOC ON )")
					;(out "set( CMAKE_AUTORCC ON )")
					;(out "set( CMAKE_AUTOUIC ON )")
			      ;; Core Gui Widgets PrintSupport Svg Xml OpenGL
			      (out "target_link_libraries( mytest PRIVATE Qt5::Core Qt5::Gui Qt5::PrintSupport Qt5::Widgets Threads::Threads ${CERES_LIBRARIES} qcustomplot )") ; ${QCP_LIBRARIES}

					; (out "target_link_libraries ( mytest Threads::Threads )")
					;(out "target_precompile_headers( mytest PRIVATE vis_00_base.hpp )")
			      ))
		  ))



