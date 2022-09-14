(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

(let (#+nil (log-preamble `(do0 (include <iostream>
				   <iomanip>
				   <chrono>
				   <thread>
				   )
			  "extern std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;")))
  (defun lprint (&key (msg "") (vars nil))
    `(lprint (curly  __FILE__
		     (string ":")
		     ("std::to_string" __LINE__)
		     (string " ")
		     __func__
		     (string " ")
		     (string ,msg)
		     (string " ")
		     ,@(loop for e in vars appending
			     `(;("std::setw" 8)
					;("std::width" 8)
			       (string ,(format nil " ~a='" (emit-c :code e)))
			       ("std::to_string" ,e)
			       (string "'"))))))
  (defun init-lprint ()
    `(defun lprint (il)
	  (declare (type "std::initializer_list<std::string>" il))

	  "std::chrono::duration<double>  timestamp = std::chrono::high_resolution_clock::now() - g_start_time;"
	  (<< "std::cout"
	      ("std::setw" 10)
	      (dot timestamp
		   (count))
	      (string " ")
	      ("std::this_thread::get_id")
	      (string " ")
	      )
	  (for-range ((elem :type "const auto&")
		      il)
		     (<< "std::cout"

			 elem) )
	  (<< "std::cout"
	      "std::endl"
	      "std::flush")))
  (progn
    ;; for classes with templates use write-source and defclass+
    ;; for cpp files without header use write-source
    ;; for class definitions and implementation in separate h and cpp file

    (defparameter *source-dir* #P"example/82_static_cpp/source/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*))
    (ensure-directories-exist *full-source-dir*)
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
					;  <memory>
		     )
		   

					;"namespace stdex = std::experimental;"


		    "std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;"

		    ,(init-lprint)

		    (defun main (argc argv)
		      (declare (type int argc)
			       (type char** argv)
			       (values int))


		      (setf g_start_time ("std::chrono::high_resolution_clock::now"))

		      ,(lprint :msg "start" :vars `(argc))
		   

		      (return 0))))

    (with-open-file (s (format nil "~a/CMakeLists.txt" *full-source-dir*)
		       :direction :output
		       :if-exists :supersede
		       :if-does-not-exist :create)
      ;;https://clang.llvm.org/docs/AddressSanitizer.html
      ;; cmake -DCMAKE_BUILD_TYPE=Debug -GNinja ..
      ;;
      (let ((dbg "-ggdb -O0")
	    (asan "" ; "-fno-omit-frame-pointer -fsanitize=address -fsanitize-address-use-after-return=always -fsanitize-address-use-after-scope"
	      )
	    (show-err ;"";
	     " -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self  -Wmissing-declarations -Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wswitch-default -Wundef -Werror -Wno-unused"
					;"-Wlogical-op -Wnoexcept  -Wstrict-null-sentinel  -Wsign-promo-Wstrict-overflow=5  "

	      ))
	(macrolet ((out (fmt &rest rest)
		     `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
	  (out "cmake_minimum_required( VERSION 3.13 )")
	  (out "project( mytest LANGUAGES CXX )")
	  #+nil(loop for e in `(xtl xsimd xtensor)
		do
		(format s "find_package( ~a REQUIRED )~%" e))
	  ;(out "set( CMAKE_CXX_COMPILER clang++ )")
	  (out "set( CMAKE_CXX_COMPILER g++ )")
	  (out "set( CMAKE_VERBOSE_MAKEFILE ON )")
	  (out "set (CMAKE_CXX_FLAGS_DEBUG \"${CMAKE_CXX_FLAGS_DEBUG} ~a ~a ~a \")" dbg asan show-err)
	  (out "set (CMAKE_LINKER_FLAGS_DEBUG \"${CMAKE_LINKER_FLAGS_DEBUG} ~a ~a \")" dbg show-err )
					;(out "set( CMAKE_CXX_STANDARD 23 )")

	  

	  (out "set( SRCS ~{~a~^~%~} )"
	       (append
		(directory (format nil "~a/*.cpp" *full-source-dir*))))

	  (out "add_executable( mytest ${SRCS} )")
					;(out "include_directories( /usr/local/include/  )")

	  (out "target_compile_features( mytest PUBLIC cxx_std_23 )")

	  (out "target_link_options( mytest PRIVATE -static -static-libgcc -static-libstdc++  )")
	  #+nil (loop for e in `(imgui implot)
		      do
		      (out "find_package( ~a CONFIG REQUIRED )" e))
	  #+nil
	  (out "target_link_libraries( mytest PRIVATE ~{~a~^ ~} )"
	       `(			;"imgui::imgui"
					; "implot::implot"
					;"std::mdspan"
		 "xtensor"
		 "xtensor::optimize"
		 "xtensor::use_xsimd"

		 ))

					; (out "target_link_libraries ( mytest Threads::Threads )")
					;(out "target_precompile_headers( mytest PRIVATE vis_00_base.hpp )")
	  ))
      )))



