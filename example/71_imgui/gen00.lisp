(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

(let ((log-preamble `(do0 (include <iostream>
				   <iomanip>
				   <chrono>
				   <thread>
				   )
			  "extern std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;")))
  (progn
    ;; for classes with templates use write-source and defclass+
    ;; for cpp files without header use write-source
    ;; for class definitions and implementation in separate h and cpp file

    (defparameter *source-dir* #P"example/71_imgui/source/")
    (load "util.lisp")
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name `MainWindow
     :headers `()
     :header-preamble `(do0
			(include <vector>
				 )

			)
     :implementation-preamble `(do0 (include <qcustomplot.h>
					     )
				    ,log-preamble
				    )
     :code `(do0
	     (defclass MainWindow ()
	       "public:"

	       (defmethod MainWindow (&key (parent 0))
		 (declare (type QWidget* parent)
			  (explicit)
			  (construct
			   )
			  (values :constructor))

		 ,(lprint)
		 )
	       (defmethod ~MainWindow ()
		 (declare
		  (values :constructor))))))

    (write-source (asdf:system-relative-pathname
		   'cl-cpp-generator2
		   (merge-pathnames #P"main.cpp"
				    *source-dir*))
		  `(do0
		    (include "MainWindow.h"

			     )
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

		    "std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;"
		    (defun main (argc argv)
		      (declare (type int argc)
			       (type char** argv)
			       (values int))
		      (setf g_start_time ("std::chrono::high_resolution_clock::now"))
		      ,(lprint)

		      (return 0))))

    (with-open-file (s "source/CMakeLists.txt" :direction :output
		       :if-exists :supersede
		       :if-does-not-exist :create)
      ;;https://clang.llvm.org/docs/AddressSanitizer.html
      ;; cmake -DCMAKE_BUILD_TYPE=Debug -GNinja ..
      ;; -fno-omit-frame-pointer -fsanitize=address -fsanitize-address-use-after-return=always -fsanitize-address-use-after-scope
      (let ((dbg "-ggdb -O0 ")
	    (show-err   " -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel -Wstrict-overflow=5 -Wswitch-default -Wundef -Werror -Wno-unused"
	      )
	    (qt-components `(Core Gui PrintSupport Widgets Charts)))
	(macrolet ((out (fmt &rest rest)
		     `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
	  (out "cmake_minimum_required( VERSION 3.4 )")
	  (out "project( mytest LANGUAGES CXX )")
					;(out "set( CMAKE_CXX_COMPILER clang++ )")
	  (out "set( CMAKE_VERBOSE_MAKEFILE ON )")
	  (out "set (CMAKE_CXX_FLAGS_DEBUG \"${CMAKE_CXX_FLAGS_DEBUG} ~a ~a \")" dbg show-err)
	  (out "set (CMAKE_LINKER_FLAGS_DEBUG \"${CMAKE_LINKER_FLAGS_DEBUG} ~a ~a \")" dbg show-err )
					;(out "set( CMAKE_CXX_STANDARD 20 )")
					;(out "set( CMAKE_CXX_COMPILER clang++ )")

	  (out "set( SRCS ~{~a~^~%~} )"
	       (directory "source/*.cpp"))

	  (out "add_executable( mytest ${SRCS} )")
	  (out "target_compile_features( mytest PUBLIC cxx_std_17 )")

					;(out "qt5_generate_moc( ~{~a~^ ~} gui.moc TARGET mytest )" (directory "source_03spline_curve/gui.h"))
					;(out "target_include_directories( mytest PRIVATE ${CERES_INCLUDE_DIRS} )")
					; (out "target_link_libraries( mytest PRIVATE ${CERES_LIBRARIES} ${QCP_LIBRARIES} )")
					;(out "set_target_properties( Qt5::Core PROPERTIES MAP_IMPORTED_CONFIG_DEBUG \"RELEASE\" )")
					;(out "set( CMAKE_AUTOMOC ON )")
					;(out "set( CMAKE_AUTORCC ON )")
					;(out "set( CMAKE_AUTOUIC ON )")
	  ;; Core Gui Widgets PrintSupport Svg Xml OpenGL ${CERES_LIBRARIES}

	  (out "target_link_libraries( mytest PRIVATE ~{~a~^ ~} )"
	       `("imgui::imgui"))

					; (out "target_link_libraries ( mytest Threads::Threads )")
					;(out "target_precompile_headers( mytest PRIVATE vis_00_base.hpp )")
	  ))
      )))



