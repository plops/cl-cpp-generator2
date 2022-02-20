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
    (defparameter *source* "03source")
    (defparameter *source-dir* (format nil "example/72_emsdk/~a/" *source*))
    (load "util.lisp")

    (write-impl-class
     :name `SokolApp
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :private-header-preamble
     `(include<>

       sokol_app.h
       sokol_gfx.h
       sokol_glue.h)
     :private-implementation-preamble
     `(do0
       ,log-preamble
       (do0
	"#define SOKOL_GLES2"
	(include<>

	 sokol_app.h
	 sokol_gfx.h
	 sokol_glue.h))
       (do0
	"#define SOKOL_IMGUI_IMPL"
	(include<> imgui.h
		   util/sokol_imgui.h)
	))
     :private-members `(#+nil (:name desc :type "sg_desc"
				     :init-form (sg_desc (curly))
				     ))
     :private-constructor-code `(do0
				 ,(lprint :msg (format nil "constructor "))
					;(setf desc->context (sapp_sgcontext))
					;(sg_setup (desc.get))
				 )
     :private-destructor-code `(do0

				)
     )

    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name `index
     :headers `()
     :header-preamble `(do0
			(include<>
			 emscripten.h

			 ))
     :implementation-preamble `(do0
				,log-preamble
				(include "SokolApp.h"))
     :code `(do0
	     (do0 "std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;"
		  "std::mutex g_stdout_mutex;")
	     (defun main (argc argv)
	       (declare (type int argc)
			(type char** argv)
			(values "extern \"C\" int"))
	       (setf g_start_time ("std::chrono::high_resolution_clock::now"))
	       (progn
		 ,(lprint :msg "enter program" :vars `(argc (aref argv)))
		 (let ((sa (SokolApp)))
		   )
		 ,(lprint :msg "exit program")
		 (return 0)))
	     ))



    (write-cmake
     (format nil "~a/CMakeLists.txt" *source*)
     :code
     (let ((dbg "-ggdb -O0 ")
	   (asan "" ; "-fno-omit-frame-pointer -fsanitize=address -fsanitize-address-use-after-return=always -fsanitize-address-use-after-scope"
	     )
	   (show-err ""; " -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self  -Wmissing-declarations -Wmissing-include-dirs  -Woverloaded-virtual -Wredundant-decls -Wshadow  -Wswitch-default -Wundef   -Wunused -Wunused-parameter  -Wold-style-cast -Wsign-conversion "
	     ;;
	     ;; -Werror ;; i rather see the warnings
	     ;; "-Wlogical-op -Wnoexcept  -Wstrict-null-sentinel  -Wsign-promo-Wstrict-overflow=5  " ;; not supported by emcc
	     ))
       (macrolet ((out (fmt &rest rest)
		    `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
	 (out "cmake_minimum_required( VERSION 3.0 )")

	 (out "project( example LANGUAGES CXX )")

	 (out "set( CMAKE_CXX_STANDARD 17 )")
	 (out "set( CMAKE_CXX_STANDARD_REQUIRED True )")
					;(out "set( OpenCV_DIR /home/martin/src/opencv/build_wasm/ )")
					;(out "set( OpenCV_STATIC ON )")
					;(out "find_package( OpenCV REQUIRED )")
					;(out "include_directories( ${OpenCV_INCLUDE_DIRS} /home/martin/src/opencv_contrib/modules/aruco/include )")
	 (out "include_directories( ~{~a~^ ~} )" `(/home/martin/src/sokol
						   /home/martin/src/imgui))
	 (out "option( BUILD_WASM \"Build Webassembly\" ON )")
	 (progn
	   (out "set( CMAKE_VERBOSE_MAKEFILE ON )")
					;(out "set( USE_FLAGS \"-s USE_SDL=2\" )")
					;(out "set( CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} ${USE_FLAGS}\" )")
	   (out "set( CMAKE_CXX_FLAGS_DEBUG \"${CMAKE_CXX_FLAGS_DEBUG}  ~a ~a ~a \")"
		dbg asan show-err)
	   )

	 (out "set( CMAKE_EXECUTABLE_SUFFIX \".html\" )")

	 (out "set( SRCS ~{~a~^~%~} )"
	      (append
	       (directory (format nil "~a/*.cpp" *source*))
					;(directory "/home/martin/src/opencv_contrib/modules/aruco/src/*.cpp")
	       ;; git clone -b docking --single-branch https://github.com/ocornut/imgui
					;(directory "/home/martin/src/opencv_contrib/modules/aruco/src/*.cpp")
	       ))

	 (out "add_executable( index ${SRCS} )")

	 #+nil (loop for e in `(imgui ;implot GLEW
				)
		     do
		     (out "find_package( ~a CONFIG REQUIRED )" e))

	 #+nil (out "target_link_libraries( index PRIVATE ~{~a~^ ~} )"
		    `("imgui::imgui"
					;"implot::implot"
					;"GLEW::GLEW"
					;"${OpenCV_LIBS}"
		      ))
	 )))
    ))

