(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

(let ((log-preamble `(do0 (include <iostream>
				   <iomanip>
				   <chrono>
				   <thread>
				   <mutex>)
			  "extern std::mutex g_stdout_mutex;"
			  "extern std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;")))
  (progn
    ;; for classes with templates use write-source and defclass+
    ;; for cpp files without header use write-source
    ;; for class definitions and implementation in separate h and cpp file
    (defparameter *source* "00source")
    (defparameter *source-dir* (format nil "example/72_emsdk/~a/" *source*))
    (load "../71_imgui/util.lisp")
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name `index
     :headers `()
     :header-preamble `(do0
			(include<> iostream))
     :implementation-preamble `()
     :code `(do0

	     (defun main (argc argv)
	       (declare (type int argc)
			(type char** argv)
			(values int))
					;(setf g_start_time ("std::chrono::high_resolution_clock::now"))
	       (progn
		 ;;,(lprint :msg "start" :vars `(argc (aref argv 0)))
		 (<< std--cout (string "hello world") std--endl)
		 (return 0)))
	     ))

    (with-open-file (s (format nil "~a/CMakeLists.txt" *source*)
		       :direction :output
		       :if-exists :supersede
		       :if-does-not-exist :create)
      (let ((dbg "-ggdb -O0 ")
	    (asan "" ; "-fno-omit-frame-pointer -fsanitize=address -fsanitize-address-use-after-return=always -fsanitize-address-use-after-scope"
	      )
	    (show-err " -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self  -Wmissing-declarations -Wmissing-include-dirs  -Woverloaded-virtual -Wredundant-decls -Wshadow  -Wswitch-default -Wundef -Werror  -Wno-unused -Wno-unused-parameter"
	      ;;
	      ;; -Wold-style-cast -Wsign-conversion
	      ;; "-Wlogical-op -Wnoexcept  -Wstrict-null-sentinel  -Wsign-promo-Wstrict-overflow=5  "
	      ))
	(macrolet ((out (fmt &rest rest)
		     `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
	  (out "cmake_minimum_required( VERSION 3.5.1 )")

	  (out "project( example LANGUAGES CXX )")

	  (out "set( CMAKE_CXX_STANDARD 17 )")
	  (out "set( CMAKE_CXX_STANDARD_REQUIRED True )")

	  (progn
	    (out "set( CMAKE_VERBOSE_MAKEFILE ON )")
	    (out "set (CMAKE_CXX_FLAGS_DEBUG \"${CMAKE_CXX_FLAGS_DEBUG} ~a ~a ~a \")" dbg asan show-err))

	  (out "set( CMAKE_EXECUTABLE_SUFFIX \".html\" )")

	  (out "set( SRCS ~{~a~^~%~} )"
	       (append
		(directory "00source/*.cpp")))

	  (out "add_executable( index ${SRCS} )"))))))

