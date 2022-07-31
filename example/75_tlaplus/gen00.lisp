(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

;; convert this code
;; https://github.com/lemmy/BlockingQueue/blob/5917d5621d7ba5f2b8d66cd6180a058e5e929a87/impl/producer_consumer.c
;; into s-expressions (and maybe use c++ so that i can use my logging functions)

(in-package :cl-cpp-generator2)

(let ((log-preamble `(do0 (include <iostream>
				   <iomanip>
				   <chrono>
				   <thread>
				   )
			  "extern std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;")))
  (defun lprint (&key (msg "") (vars nil))
  #+nil `(comments ,msg)
  #-nil`(progn				;do0
	  " "
	  (do0				;let
	   #+nil ((lock (std--unique_lock<std--mutex> ,(g `_stdout_mutex)))
		  )

	   (do0
					;("std::setprecision" 3)
	    ;; https://stackoverflow.com/questions/7889136/stdchrono-and-cout

	    "std::chrono::duration<double>  timestamp = std::chrono::high_resolution_clock::now() - g_start_time;"
	    (<< "std::cout"
		;;"std::endl"
		("std::setw" 10)
		#+nil (dot ("std::chrono::high_resolution_clock::now")
			   (time_since_epoch)
			   (count)
			   )
		(dot timestamp
		     (count))
					;,(g `_start_time)

					;(string " ")
					;(dot now (period))
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
  (progn
    ;; for classes with templates use write-source and defclass+
    ;; for cpp files without header use write-source
    ;; for class definitions and implementation in separate h and cpp file

    (defparameter *source-dir* #P"example/75_tlaplus/source/")
    (ensure-directories-exist (asdf:system-relative-pathname
			       'cl-cpp-generator2
			       *source-dir*))
    (write-source (asdf:system-relative-pathname
		   'cl-cpp-generator2
		   (merge-pathnames #P"main.cpp"
				    *source-dir*))
		  `(do0
		    
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

		    (include <stdio.h>
			     <stdlib.h>
			     <stdint.h>
			     <pthread.h>)
		    "uint32_t buf_size, numProducers, numConsumers, *buffer, fillIndex, useIndex, count=0;"
		    "pthread_cond_t modify;"
		    "pthread_mutex_t mutex;"


		    "std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;"


		    (defun append (value)
		      (declare (type uint32_t value))
		      (setf (aref buffer fillIndex)
			    value
			    fillIndex (% (+ fillIndex 1)
					 buf_size))
		      (incf count))
		    (defun head ()
		      (declare (values uint32_t))
		      (let ((tmp (aref buffer useIndex)))
			(setf 
			 useIndex (% (+ useIndex 1)
				     buf_size))
			(decf count)
			(return tmp)))

		    
		    (defun producer (arg)
		      (declare (type void* arg)
			       (values void*))
		      ,@(loop for (name code )
			      in `((red 31)
				   (green 32)
				   (yellow 33)
				   (blue 34)
				   (magenta 35)
				   (cyan 36)
				   (white 37))
			    collect
			    (format nil "#define ~a \"~a\""
				    (string-upcase (format nil "P~a" name))
				    (format nil "\\033[1;~amp\\033[0m" code)))
		      (while 1
			     (pthread_mutex_lock &mutex)
			     (while (== count buf_size)
				    (printf PRED)
				    (pthread_cond_wait &modify &mutex))
			     (append (% (rand) 10))
			     (printf PYELLOW)
			     (fflush )
			     (pthread_cond_signal &modify)
			     (pthread_mutex_unlock &mutex)))

		    (defun consumer (arg)
		      (declare (type void* arg)
			       (values void*))
		      ,@(loop for (name code )
				in `((red 31)
				     (green 32)
				     (yellow 33)
				     (blue 34)
				     (magenta 35)
				     (cyan 36)
				     (white 37))
			      collect
			      (format nil "#define ~a \"~a\" "
				       (string-upcase (format nil "~a" name))
				       (format nil "\\033[1;~amc%01d\\033[0m" code)))
		      (let ((id (deref (static_cast<uint32_t*> arg)))
					;(report (long 0))
			    )
			(while 1
			       (pthread_mutex_lock &mutex)
			       (while (== 0 count)
				      (printf RED id)
				      (pthread_cond_wait &modify &mutex)))
			(head)
			(printf YELLOW id)
			(pthread_cond_signal &modify)
			(pthread_mutex_unlock &mutex)))

		    (defun main (argc argv)
		      (declare (type int argc)
			       (type char** argv)
			       (values int))
		      (when (< argc 4)
			(printf (string "Usage: ./main <buffer_size> <#producers> <#consumers>\\n"))
			(exit 1))
		      (setf g_start_time ("std::chrono::high_resolution_clock::now"))
		      
		      ,(lprint :msg "start" :vars `(argc (aref argv 0)))

		      (do0
		       (srand 999)
			  (do0
			   ,@(loop for e in `(buf_size numProducers numConsumers)
				   and e-i from 1
				   collect
				   `(setf ,e (atoi (aref argv ,e-i))))))
		      (do0
		       ,(lprint :msg "initiate mutex and condition variable")
		       (pthread_mutex_init &mutex nullptr)
		       (pthread_cond_init &modify nullptr))
		      (do0
		       ,(lprint :msg "allocate buffer" :vars `(buf_size))
		       (setf buffer (static_cast<uint32_t*> (malloc (* buf_size
								       (sizeof uint32_t))))
			     ))
		      "pthread_t prods[numProducers], cons[numConsumers];"
		      "uint32_t threadIds[numConsumers], i;"
		      (do0
		       (do0
			,(lprint :msg "start consumers" :vars `(numConsumers))
			(dotimes (i numConsumers)
			  (setf (aref threadIds i) i)
			  (pthread_create (+ cons i)
					  nullptr
					  consumer
					  (+ threadIds i))))
		       
		       (do0
			,(lprint :msg "start producers" :vars `(numProducers))
			(dotimes (i numProducers)
			  (pthread_create (+ prods i)
					  nullptr
					  producer
					  nullptr))))

		      		      
		      (do0
		       ,(lprint :msg "wait for threads to finish")
		       (dotimes (i numProducers)
			 (pthread_join (aref prods i) nullptr))
		       (dotimes (i numConsumers)
			 (pthread_join (aref cons i) nullptr)))
		      ,(lprint :msg "leave program")
		      (return 0))))

    (with-open-file (s "source/CMakeLists.txt" :direction :output
		       :if-exists :supersede
		       :if-does-not-exist :create)
      ;;https://clang.llvm.org/docs/AddressSanitizer.html
      ;; cmake -DCMAKE_BUILD_TYPE=Debug -GNinja ..
      ;;
      (let ((dbg "-ggdb -O0 ")
	    (asan "-fno-omit-frame-pointer -fsanitize=address -fsanitize-address-use-after-return=always -fsanitize-address-use-after-scope")
	    (show-err ""; " -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self  -Wmissing-declarations -Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wswitch-default -Wundef -Werror -Wno-unused"
					;"-Wlogical-op -Wnoexcept  -Wstrict-null-sentinel  -Wsign-promo-Wstrict-overflow=5  "

	      ))
	(macrolet ((out (fmt &rest rest)
		     `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
	  (out "cmake_minimum_required( VERSION 3.4 )")
	  (out "project( mytest LANGUAGES CXX )")
	  (out "set( CMAKE_CXX_COMPILER clang++ )")
	  (out "set( CMAKE_VERBOSE_MAKEFILE ON )")
	  (out "set (CMAKE_CXX_FLAGS_DEBUG \"${CMAKE_CXX_FLAGS_DEBUG} ~a ~a ~a \")" dbg asan show-err)
	  (out "set (CMAKE_LINKER_FLAGS_DEBUG \"${CMAKE_LINKER_FLAGS_DEBUG} ~a ~a \")" dbg show-err )
					;(out "set( CMAKE_CXX_STANDARD 20 )")
					;(out "set( CMAKE_CXX_COMPILER clang++ )")

	  (out "set( SRCS ~{~a~^~%~} )"
	       (append
		(directory "source/*.cpp")))

	  (out "add_executable( mytest ${SRCS} )")
	  (out "target_compile_features( mytest PUBLIC cxx_std_17 )")

	  #+nil (loop for e in `(imgui implot)
		do
		(out "find_package( ~a CONFIG REQUIRED )" e))

	  #+nil (out "target_link_libraries( mytest PRIVATE ~{~a~^ ~} )"
	       `("imgui::imgui"
		 "implot::implot"))

					; (out "target_link_libraries ( mytest Threads::Threads )")
					;(out "target_precompile_headers( mytest PRIVATE vis_00_base.hpp )")
	  ))
      )))



