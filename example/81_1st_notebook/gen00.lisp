(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *source-dir* #P"example/81_1st_notebook/source/")
  (ensure-directories-exist (asdf:system-relative-pathname
			     'cl-cpp-generator2
			     *source-dir*))
  (defun lprint2 (&key (msg "") (vars nil))
    `(progn				;do0
       " "
       (do0				;let
	#+nil ((lockxxxxxxx+ (std--unique_lock<std--mutex> ,(g `_stdout_mutex)))
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

  (defun lprint (&key (msg "") (vars nil))
    `(lprint __FILE__
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
		       (string "'")))))
  (let ((notebook-name "main")
	(idx "00")
	)
    (write-notebook
     :nb-file (asdf:system-relative-pathname
	       'cl-cpp-generator2
	       (merge-pathnames (format nil "c~a_~a.ipynb" idx notebook-name)
				*source-dir*))
     :nb-code
     `((cpp
	(do0 ,(format nil "//|default_exp c~a_~a" idx notebook-name)))
       (markdown "show all known namespaces")
       (cpp
	".namespace")
       (markdown "show all global variables")
       (cpp
	".g")
       (markdown "show help for cling commands")
       (cpp ".help")
       (markdown "show all typedefs")
       (cpp ".typedef")
       (markdown "show all parsed files")
       (cpp ".files")
       (markdown "show all parsed files with statistics")
       (cpp ".fileEx")
       (markdown "show optimization level")
       (cpp
	".O")
       (cpp
	(do0
	 (comments "looks like i have to execute this cell before i can execute the includes")
	 (do
	  (let ((_a (int 12))
		(_b (int 32)))
	    (+ _a _b)))))
       (cpp
	(do0
	 (comments "Clang 9.0.1 (http://root.cern.ch/git/clang.git ddd3a61c4ec7cb9661e8dc9781dc797f70537519) (http://root.cern.ch/git/llvm.git c41338c59334340ee4d85a7c9bbdf49a4f59f76b)")
	 __amd64__ ;; 1
	 __linux__ ;; 1
	 __cplusplus__ ;; 201703L
	 __SSE__ ;; 1
	 __SSE2__ ;; 1
	 __MMX__ ;; 1
	 __GLIBCXX_FAST_MATH ;; 0
	 _GLIBCXX_USE_FLOAT128 ;; 1
	 NDEBUG ;; 1
	 __VERSION__))
       ;; .undo [n] tries to undo the last n lines
       (cpp
	".g _a")
       (cpp
	(do0
	 "//|export"
	 (include
					;<tuple>
					;<mutex>
	  <thread>
	  <iostream>
	  <iomanip>
	  <chrono>
					;  <memory>
	  )
	 ))
       (cpp
	(do0
	 "std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;"))
       (cpp
	(do0
	 (setf g_start_time ("std::chrono::high_resolution_clock::now"))))
       (cpp
	(defun lprint (il)
	  (declare (type "std::initializer_list<std::string>" il))

	  "std::chrono::duration<double>  timestamp = std::chrono::high_resolution_clock::now() - g_start_time;"
	  (<< "std::cout"
	      ("std::setw" 10)
	      (dot timestamp
		   (count))
	      (string " ")
	      ("std::this_thread::get_id")
	      (string " ")
	      __FILE__
	      (string ":")
	      __LINE__
	      (string " ")
	      __func__
	      (string " "))
	  (for-range ((elem :type "const auto&")
		      il)
		     (<< "std::cout"
			 ("std::setw" 8)
			 elem) )
	  (<< "std::cout"
		  "std::endl"
		  "std::flush")))
       (cpp
	(progn
	  (let ((a (int 3))
		(b (int 44)))
	    ,(lprint :vars `(a b (+ a b))))))
       (cpp
	"//|export"
	(defun main (argc argv)
	  (declare (type int argc)
		   (type char** argv)
		   (values int))

	  (return 0))))
     ))
  (with-open-file (s "source/CMakeLists.txt" :direction :output
		     :if-exists :supersede
		     :if-does-not-exist :create)
    (let ((dbg "-ggdb -O0 -std=gnu++2b")
	  (asan "" ; "-fno-omit-frame-pointer -fsanitize=address -fsanitize-address-use-after-return=always -fsanitize-address-use-after-scope"
	    )
	  (show-err ;"";
					;" -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self  -Wmissing-declarations -Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wswitch-default -Wundef -Werror -Wno-unused"
	   "-Wlogical-op -Wnoexcept  -Wstrict-null-sentinel  -Wsign-promo-Wstrict-overflow=5  "

	    ))
      (macrolet ((out (fmt &rest rest)
		   `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
	(out "cmake_minimum_required( VERSION 3.19 )")
	(out "project( mytest LANGUAGES CXX )")
	(out "find_package( mdspan REQUIRED )")
					; (out "set( CMAKE_CXX_COMPILER clang++ )")
	(out "set( CMAKE_CXX_COMPILER clang++ )")
	(out "set( CMAKE_VERBOSE_MAKEFILE ON )")
	(out "set (CMAKE_CXX_FLAGS_DEBUG \"${CMAKE_CXX_FLAGS_DEBUG} ~a ~a ~a \")" dbg asan show-err)
	(out "set (CMAKE_LINKER_FLAGS_DEBUG \"${CMAKE_LINKER_FLAGS_DEBUG} ~a ~a \")" dbg show-err )
					;(out "set( CMAKE_CXX_STANDARD 23 )")


	(out "set( SRCS ~{~a~^~%~} )"
	     (append
	      (directory "source/*.cpp")))

	(out "add_executable( mytest ${SRCS} )")
	#+nil (out "include_directories( /usr/local/include/  )")

					;(out "target_compile_features( mytest PUBLIC cxx_std_23 )")

	#+nil (loop for e in `(imgui implot)
		    do
		    (out "find_package( ~a CONFIG REQUIRED )" e))

	#+nil (out "target_link_libraries( mytest PRIVATE ~{~a~^ ~} )"
		   `(			;"imgui::imgui"
					; "implot::implot"
		     "std::mdspan"
		     ))

					; (out "target_link_libraries ( mytest Threads::Threads )")
					;(out "target_precompile_headers( mytest PRIVATE vis_00_base.hpp )")
	))
    ))



