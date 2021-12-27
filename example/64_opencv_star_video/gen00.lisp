(eval-when (:compile-toplevel :execute :load-toplevel)
     (ql:quickload "cl-cpp-generator2")
     ;(ql:quickload "cl-ppcre")
     ;(ql:quickload "cl-change-case")
     ) 
(in-package :cl-cpp-generator2)
(progn
  (defparameter *source-dir* #P"example/64_opencv_star_video/source/")
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (defun logprint (msg &optional rest)
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
	    (defclass Points ()
		     "float *x,*y;"
		     "public:"
		     (defmethod Points ()
		       (declare
			(values :constructor)
			(construct (x NULL)
				   (y NULL))))
		     (defmethod Points (x y)
		       (declare
			(type float* x y)
			(values :constructor)
			(construct (x x)
				   (y y))))))))

    (let ((fn-h (asdf:system-relative-pathname
			  'cl-cpp-generator2
			  (merge-pathnames #P"star_tracker.h"
					   *source-dir*))))
     (with-open-file (sh fn-h
			 :direction :output
			 :if-exists :supersede
			 :if-does-not-exist :create)
       (emit-c :code
	       `(do0
		 "#pragma once"
		 (include <opencv2/opencv.hpp>
			  ;<opencv2/videoio.hpp>
			  <iostream>
			  <chrono>
			  <thread>)
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
		  (merge-pathnames #P"star_tracker.cpp"
				   *source-dir*))
		 `(do0
		   (include "star_tracker.h")
		   
		   ,type-definitions

		   "using namespace std;"
		   "using namespace cv;"
		   (defun main (argc argv)
		     (declare (type int argc)
			      (type char** argv)
			      (values int))
		     (let ((fn (string "~/stars_XnRy3sJqfu4.webm"))
			   (cap (VideoCapture fn))
			   )
		       ;("VideoCapture cap" fn)
		       (unless (cap.isOpened)
			 ,(logprint "error opening file" `(fn) )
			 (return -1))
		       
		       (while 1
			      (let ((frame))
				(declare (type Mat frame))
				(>> cap frame)
				(when (frame.empty)
				  break)
				(imshow (string "frame")
					frame)
				(let ((c (static_cast<char> (waitKey 25))))
				  (when (== 27 c)
				    break))))
		       (cap.release)
		       (destroyAllWindows))
		     (return 0)
		     )
		   )))
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

 

