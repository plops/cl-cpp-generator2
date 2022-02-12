(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

;; for classes with templates use write-source and defclass+

;; for cpp files without header use write-source

;; for class definitions and implementation in separate h and cpp file

(defun write-class (&key name dir code headers header-preamble implementation-preamble moc)
  "split class definition in .h file and implementation in .cpp file. use defclass in code. headers will only be included into the .cpp file. the .h file will get forward class declarations. additional headers can be added to the .h file with header-preamble and to the .cpp file with implementation preamble. if moc is true create moc_<name>.h file from <name>.h"
  (let ((fn-h (format nil "~a/~a.h" dir name))
	(fn-h-nodir (format nil "~a.h" name))
	(fn-moc-h (format nil "~a/moc_~a.h" dir name))
	(fn-moc-h-nodir (format nil "moc_~a.h" name))
	(fn-cpp (format nil "~a/~a.cpp" dir name)))
    (with-open-file (sh fn-h
			:direction :output
			:if-exists :supersede
			:if-does-not-exist :create)
      (loop for e in `((pragma once)
		       ,header-preamble
		       ,@(loop for h in headers
			       collect
			       ;; write forward declaration for classes
			       (format nil "class ~a;" h)))
	    do
	    (when e
	      (format sh "~a~%"
		      (emit-c :code e))))

      (when code
	(emit-c :code
		`(do0
		  ,code)
		:hook-defun #'(lambda (str)
				(format sh "~a~%" str))
		:hook-defclass #'(lambda (str)
                                   (format sh "~a;~%" str))
		:header-only t)))
    (sb-ext:run-program "/usr/bin/clang-format"
                        (list "-i"  (namestring fn-h)
			      "-o"))
    (when moc
      (sb-ext:run-program "/usr/bin/moc-qt5"
                          (list (namestring fn-h)
				"-o" (namestring fn-moc-h))))
    (write-source fn-cpp
		  `(do0
		    ,(if implementation-preamble
			 implementation-preamble
			 `(comments "no implementation preamble"))
		    ,@(loop for h in headers
			    collect
			    `(include ,(format nil "<~a>" h)))
		    ,(if moc
			 `(include ,(format nil "~a" fn-moc-h-nodir))
			 `(include ,(format nil "~a" fn-h-nodir)))
		    ,(if code
			 code
			 `(comments "no code"))))))

(progn
  (defparameter *source-dir* #P"example/70_qttest/source/")
  (load "util.lisp")
  (write-class
   :dir (asdf:system-relative-pathname
	 'cl-cpp-generator2
	 *source-dir*)
   :name `MainWindow
   :moc t
   :headers `(QCustomPlot QMainWindow QWidget)
   :header-preamble `(do0
		      (include <vector>))
   :code `(do0
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
			 (plot_ (new (QCustomPlot this))))
			(values :constructor))
	       (setCentralWidget plot_)
	       (setGeometry 400 250 542 390))
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
	       (incf graph_count_))
	     (defmethod ~MainWindow ()
	       (declare
		(values :constructor))))))


  (write-source (asdf:system-relative-pathname
		 'cl-cpp-generator2
		 (merge-pathnames #P"main.cpp"
				  *source-dir*))
		`(do0
		  (include "MainWindow.h")
		  (include
					;<tuple>
					;<mutex>
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

		  (defun main (argc argv)
		    (declare (type int argc)
			     (type char** argv)
			     (values int))
		    (do0
		     "QApplication app(argc,argv);"
		     "MainWindow w;")

		    (w.show)

		    (return (app.exec))
		    )
		  ))

  (with-open-file (s "source/CMakeLists.txt" :direction :output
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
					;(out "find_package ( Ceres REQUIRED ) ")
	(out "find_package ( PkgConfig REQUIRED )")
	(out "pkg_check_modules( QCP REQUIRED qcustomplot-qt5 )")
					;(out "qt5_generate_moc( ~{~a~^ ~} gui.moc TARGET mytest )" (directory "source_03spline_curve/gui.h"))
	(out "target_include_directories( mytest PRIVATE ${CERES_INCLUDE_DIRS} )")
					; (out "target_link_libraries( mytest PRIVATE ${CERES_LIBRARIES} ${QCP_LIBRARIES} )")
					;(out "set_target_properties( Qt5::Core PROPERTIES MAP_IMPORTED_CONFIG_DEBUG \"RELEASE\" )")
					;(out "set( CMAKE_AUTOMOC ON )")
					;(out "set( CMAKE_AUTORCC ON )")
					;(out "set( CMAKE_AUTOUIC ON )")
	;; Core Gui Widgets PrintSupport Svg Xml OpenGL ${CERES_LIBRARIES}
	(out "target_link_libraries( mytest PRIVATE Qt5::Core Qt5::Gui Qt5::PrintSupport Qt5::Widgets Threads::Threads  ${QCP_LIBRARIES} )")

					; (out "target_link_libraries ( mytest Threads::Threads )")
					;(out "target_precompile_headers( mytest PRIVATE vis_00_base.hpp )")
	))
    ))



