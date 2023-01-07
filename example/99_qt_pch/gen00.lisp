(eval-when (:compile-toplevel :execute :load-toplevel)
	   (ql:quickload "cl-cpp-generator2")
	   (ql:quickload "cl-ppcre")
	   (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  ;; for classes with templates use write-source and defclass+
  ;; for cpp files without header use write-source
  ;; for class definitions and implementation in separate h and cpp file
  (defparameter *source-dir* #P"example/99_qt_pch/source00/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")
  #+nil
  (let ((name `AGameCharacter))
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include "bla.h"))
     :implementation-preamble `(do0
				(include "bah.h"))
     :code `(do0
	     (defclass ,name ()
	       "public:"
	       (defmethod ,name ()
		 (declare
		  (construct
		   (Camera 3))
		  (values :constructor)))))))

  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"fatheader.hpp"
		     *source-dir*))
   `(do0
     (include
      ,@(loop for e in `(file-dialog
			 push-button
			 label
			 check-box
			 combo-box
			 v-box-layout
			 h-box-layout
			 drag
			 mime-data
			 tool-button
			 frame
			 validator
			 action
			 button-group
			 header-view
			 line-edit
			 spacer-item
			 stacked-widget
			 radio-button
			 tab-widget
			 tool-tip
			 mouse-event
			 style
			 timer
			 )
	      collect
	      (format nil "<~a>"
		      (cl-change-case:pascal-case
		       (format nil "q-~a" e))))
      )

     (do0
      (include <spdlog/spdlog.h>)
					;(include <popl.hpp>)

      )))


  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"main.cpp"
		     *source-dir*))
   `(do0
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       "(void)argv;"
       ,(lprint :msg "start" :vars `(argc))

       )))

  (with-open-file (s (format nil "~a/CMakeLists.txt" *full-source-dir*)
		     :direction :output
		     :if-exists :supersede
		     :if-does-not-exist :create)
		  (let ((l-dep `(spdlog Qt5Gui Qt5Core Qt5Widgets)))
		    (macrolet ((out (fmt &rest rest)
				    `(format s ,(format nil "~&~a~%" fmt) ,@rest))
			       )
			      (out "cmake_minimum_required( VERSION 3.0 FATAL_ERROR )")
			      (out "project( mytest )")

			      ;;(out "set( CMAKE_CXX_COMPILER clang++ )")
			      ;;(out "set( CMAKE_CXX_COMPILER g++ )")
			      (out "set( CMAKE_VERBOSE_MAKEFILE ON )")

			      (out "set( SRCS ~{~a~^~%~} )"
				   (append
				    (directory (format nil "~a/*.cpp" *full-source-dir*))
				    ))

			      (out "add_executable( mytest ${SRCS} )")
			      (out "set_property( TARGET mytest PROPERTY CXX_STANDARD 20 )")

			      (out "target_link_options( mytest PRIVATE -static-libgcc -static-libstdc++   )")

			      (out "find_package( PkgConfig REQUIRED )")
			      (loop for e in l-dep
				    do
				    (out "pkg_check_modules( ~a REQUIRED ~a )" e e))

			      (out "target_include_directories( mytest PUBLIC ~{${~a_INCLUDE_DIRS}~^ ~} )" l-dep)
			      (out "target_compile_options( mytest PUBLIC ~{${~a_CFLAGS_OTHER}~^ ~} )" l-dep)
			      #+nil
			      (progn
				(out "add_library( libnc SHARED IMPORTED )")
				(out "set_target_properties( libnc PROPERTIES IMPORTED_LOCATION /home/martin/stage/cl-cpp-generator2/example/88_libnc/dep/libnc-2021-04-24/libnc.so )"))
			      (out "target_link_libraries( mytest PRIVATE ~{${~a_LIBRARIES}~^ ~} )"
				   l-dep)
			      (out "target_precompile_headers( mytest PRIVATE fatheader.hpp )")))))

