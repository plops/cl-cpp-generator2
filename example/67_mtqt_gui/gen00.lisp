(eval-when (:compile-toplevel :execute :load-toplevel)
     (ql:quickload "cl-cpp-generator2")
     ;(ql:quickload "cl-ppcre")
     ;(ql:quickload "cl-change-case")
     ) 
(in-package :cl-cpp-generator2)
(progn
  (defparameter *source-dir* #P"example/67_mtqt_gui/source/")
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  #+nil (defun logprint (msg &optional rest)
      `(progn				;do0
	 " "
	 #-nolog
	 (do0 ;let
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
	       ,@(loop for e in rest appending
		       `(("std::setw" 8)
					;("std::width" 8)
			 (string ,(format nil " ~a='" (emit-c :code e)))
			 ,e
			 (string "'")))
	       "std::endl"
	       "std::flush")))))
  (let ((type-definitions
	  `(do0

	    (defclass AnyQAppLambda ()
	      "public:"
	      (defmethod run ()
		(declare (virtual)
			 (pure)))
	      (defmethod ~AnyQAppLambda ()
		(declare (virtual)
			 (values :constructor))))
	    
	    (defclass QAppLambda (AnyQAppLambda)
	      "public:"
	      ,@(loop for (e f) in `((Lambda lambda)
				     (std--tuple<Args...> args)
				     )
		      collect
		      (format nil "~a ~a;" (emit-c :code  e) f))
	      )
	    )))

    (let ((fn-h (asdf:system-relative-pathname
			  'cl-cpp-generator2
			  (merge-pathnames #P"mtgui.h"
					   *source-dir*))))
     (with-open-file (sh fn-h
			 :direction :output
			 :if-exists :supersede
			 :if-does-not-exist :create)
       (emit-c :code
	       `(do0
		 "#pragma once"
		 (include <tuple>)
		 "struct AnyQAppLambda;"
		 "struct QCoreApplication;"
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
		  (merge-pathnames #P"mtgui.cpp"
				   *source-dir*))
		  
		  `(do0
		    (include
		     <mtgui.h>
		     <mutex>
		     <thread>
		     <QEvent>
		     <QApplication>
		     <iostream>
					;<chrono>
		     )
		    
		   
		   ,(let ((n ;4995
			    118218
			   ))
		   `(defun main (argc argv)
		      (declare (type int argc)
			       (type char** argv)
			       (values int))
		      ))
		   )))
  #+nil
  (with-open-file (s "source/CMakeLists.txt" :direction :output
					     :if-exists :supersede
					     :if-does-not-exist :create)
    (macrolet ((out (fmt &rest rest)
		 `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
      (out "cmake_minimum_required( VERSION 3.4 )")
      (out "project( mytest LANGUAGES CXX )")
      (out "set( CMAKE_CXX_COMPILER clang++ )")
      (out "set( CMAKE_CXX_FLAGS \"\"  )")
      (out "set( CMAKE_VERBOSE_MAKEFILE ON )")
      (out "set( CMAKE_CXX_STANDARD 17 )")
					;(out "set( CMAKE_CXX_COMPILER clang++ )")
      
					;(out "set( CMAKE_CXX_FLAGS )")
      (out "find_package( OpenCV REQUIRED )")
      (out "set( SRCS ~{~a~^~%~} )"
	   (directory "source/*.cpp"))
      (out "add_executable( mytest ${SRCS} )")
      ;(out "target_include_directories( mytest PUBLIC /home/martin/stage/cl-cpp-generator2/example/58_stdpar/source/ )")
      
      (out "target_link_libraries( mytest ${OpenCV_LIBS} )")
					;(out "target_precompile_headers( mytest PRIVATE vis_00_base.hpp )")
      )
    ))

 

