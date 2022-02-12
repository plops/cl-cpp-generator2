(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

;; for classes with templates use write-source and defclass+

;; for cpp files without header use write-source

;; for class definitions and implementation in separate h and cpp file

(progn
  (defparameter *source-dir* #P"example/70_qttest/source/")
  (load "util.lisp")
  (write-class
   :dir (asdf:system-relative-pathname
	 'cl-cpp-generator2
	 *source-dir*)
   :name `MainWindow
   :moc t
   :headers `(
	      QWidget)
   :header-preamble `(do0
		      (include <vector>
			       <QMainWindow>)
		      "class QCustomPlot;")
   :implementation-preamble `(include <qcustomplot.h>)
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

  (write-class
   :dir (asdf:system-relative-pathname
	 'cl-cpp-generator2
	 *source-dir*)
   :name `SysInfo
   :code `(do0
	   ;; Mastering Qt5 p. 56
	   (defclass SysInfo ()
	     "public:"

	     (defmethod SysInfo ()
	       (declare (values :constructor))
	       )
	     (defmethod ~SysInfo ()
	       (declare (values :constructor))
	       )
	     ,@(loop for (e f) in `((init void) (cpuLoadAverage double) (memoryUsed double))
		     collect
		     `(defmethod ,e ()
			(declare (virtual)
				 (pure)
				 (values ,f)))))))
  (write-class
   :dir (asdf:system-relative-pathname
	 'cl-cpp-generator2
	 *source-dir*)
   :name `SysInfoLinuxImpl
   :header-preamble `(include <QtGlobal>
			      <QVector>
			      "SysInfo.h")
   :implementation-preamble `(include <sys/types.h>
				      <sys/sysinfo.h>
				      <QFile>)
   :code `(do0
	   ;; Mastering Qt5 p. 62
	   (defclass SysInfoLinuxImpl "public SysInfo"
	     "public:"
	     "QVector<qulonglong> cpu_load_last_values_;"
	     (defmethod cpuRawData ()
	       (declare (values "QVector<qulonglong>")
			)
	       (let ((file (QFile (string "/proc/stat"))))
		 (file.open QIODevice--ReadOnly)
		 (let ((lines (file.readLine)))
		   (file.close)
		   ,(let ((def-totals `(totalUser totalUserNice totalSystem totalIdle)))
		      `(do0
			,@(loop for e in def-totals
				collect
				`(let ((,e (qulonglong 0)))))
			(std--sscanf (line.data)
				     (string ,(format nil "cpu ~{~a~^ ~}" (loop for e in def-totals
										collect
										"%llu")))
				     ,@(loop for e in def-totals
					     collect
					     `(ref ,e)))
			"QVector<qulonglong> rawData;"
			,@ (loop for e in def-totals
				 and i from 0
				 collect
				 `(dot rawData (append ,e)))
			(return rawData))))))
	     (defmethod SysInfoLinuxImpl ()
	       (declare
		(construct (SysInfo)
			   (cpu_load_last_values_))
		(values :constructor))
	       )


	     ,@(loop for (e f code) in `((init void (do0 (setf cpu_load_last_values_ (cpuRawData))))
					 (cpuLoadAverage double
							 (do0
							  (let ((firstSample cpu_load_last_values_)
								(secondSample (cpuRawData)))
							    (setf cpu_load_last_values_ secondSample)
							    (let ((overall (+ ,@(loop for i below 2 collect
										      `(- (aref secondSample ,i)
											  (aref firstSample ,i)))))
								  (total (+ overall (- (aref secondSample 3)
										       (aref firstSample 3))))
								  (percent (/ (* 100d0 overall)
									      total)))
							      (return (qBound 0d0 percent 100d0))))))
					 (memoryUsed double
						     (do0 "struct sysinfo memInfo;"
							  (sysinfo &memInfo)
							  (let ((totalMemory (* (+ (qulonglong memInfo.totalram)
										   memInfo.totalswap)
										memInfo.mem_unit))
								(totalMemoryUsed (* (+ (qulonglong (- memInfo.totalram
												      memInfo.freeram))
										       (- memInfo.totalswap
											  memInfo.freeswap))
										    memInfo.mem_unit))
								(percent (/ (* 100d0 totalMemoryUsed)
									    totalMemory)))
							    (return (qBound 0d0 percent 100d0))
							    ))))
		     collect
		     `(defmethod ,e ()
			(declare ;(override)
			 (values ,f))
			,code)))))


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
	     (directory "source/*.cpp"))

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
	(out "target_link_libraries( mytest PRIVATE ~{~a~^ ~} )"
	     `("Qt::Core" "Qt::Gui" "Qt::PrintSupport" "Qt::Widgets" "qcustomplot-qt5"
					;${QCP_LIBRARIES}
			  ))

					; (out "target_link_libraries ( mytest Threads::Threads )")
					;(out "target_precompile_headers( mytest PRIVATE vis_00_base.hpp )")
	))
    ))



