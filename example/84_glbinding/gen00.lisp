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

    (defparameter *source-dir* #P"example/84_glbinding/source00/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*))

    (ensure-directories-exist *full-source-dir*)
    (load "../84_glbinding/util.lisp")

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


		    #+nil (do0

			   (include <fstream>
				    <array>))

		    (do0
		     (include <glbinding/gl/gl.h>
			      <glbinding/glbinding.h>)
		     "using namespace gl;")
		    (do0
		     "#define GLFW_INCLUDE_NONE"
		     (include <GLFW/glfw3.h>
			      )
		     #+nil (do0 "#define GLFW_EXPOSE_NATIVE_X11"
				(include
				 <GLFW/glfw3native.h>))
					;(include <imgui/imgui.h>)

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

		      (let ((*window ((lambda ()
					(declare (values GLFWwindow*))
					(unless (glfwInit)
					  ,(lprint :msg "glfwInit failed"))
					(let ((startWidth 800)
					      (startHeight 600)
					      (window (glfwCreateWindow startWidth startHeight
									(string "hello bgfx")
									nullptr
									nullptr))
					      )
					  (declare (type "const auto" startWidth startHeight))
					  (unless window
					    ,(lprint :msg "can't create glfw window"))
					  (return window))
					)))))
		      (let ((width (int 0))
			    (height (int 0)))

			(do0
			 (comments "lazy function pointer loading")
			 (glbinding--initialize glfwGetProcAddress
						false))

			(while (not (glfwWindowShouldClose window))
			  (glfwPollEvents)



			  ((lambda ()
			     (declare (capture &width &height window))
			     (comments "react to changing window size")
			     (let ((oldwidth width)
				   (oldheight height))
			       (glfwGetWindowSize window &width &height)
			       (when (or (!= width oldwidth)
					 (!= height oldheight))
				 (comments "set view")
				 (glViewport 0 0 width height)))))
			  (do0
			   (comments "draw frame")
			   (glClear GL_COLOR_BUFFER_BIT)

			   )))

		      (glfwTerminate)




		      (return 0))))

    (with-open-file (s (format nil "~a/CMakeLists.txt" *full-source-dir*)
		       :direction :output
		       :if-exists :supersede
		       :if-does-not-exist :create)
      ;;https://clang.llvm.org/docs/AddressSanitizer.html
      ;; cmake -DCMAKE_BUILD_TYPE=Debug -GNinja ..
      ;;
      (let ((dbg "-ggdb -O0")
	    (asan ;""
	     "-fno-omit-frame-pointer -fsanitize=address -fsanitize-address-use-after-return=always -fsanitize-address-use-after-scope"
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
	  (out "set( CMAKE_CXX_COMPILER clang++ )")
					;(out "set( CMAKE_CXX_COMPILER g++ )")
	  (out "set( CMAKE_VERBOSE_MAKEFILE ON )")
	  (out "set (CMAKE_CXX_FLAGS_DEBUG \"${CMAKE_CXX_FLAGS_DEBUG} ~a ~a ~a \")" dbg asan show-err)
	  (out "set (CMAKE_LINKER_FLAGS_DEBUG \"${CMAKE_LINKER_FLAGS_DEBUG} ~a ~a \")" dbg show-err )
					;(out "set( CMAKE_CXX_STANDARD 23 )")



	  (out "set( SRCS ~{~a~^~%~} )"
	       (append
		(directory (format nil "~a/*.cpp" *full-source-dir*))
					;(directory (format nil "/home/martin/src/bgfx/examples/common/imgui/imgui.cpp"))
					;(directory (format nil "/home/martin/src/bgfx/3rdparty/dear-imgui/imgui*.cpp"))

		))

	  (out "add_executable( mytest ${SRCS} )")
					;(out "include_directories( /usr/local/include/  )")
	  #+nil (out "target_include_directories( mytest PRIVATE
/home/martin/src/entt/src/ )")

	  (out "target_compile_features( mytest PUBLIC cxx_std_20 )")

					;(out "target_link_options( mytest PRIVATE -static -static-libgcc -static-libstdc++  )")
	  (loop for e in `(glbinding)
		do
		(out "find_package( ~a REQUIRED )" e))






	  (out "target_link_libraries( mytest PRIVATE ~{~a~^ ~} )"
	       `(
		 "glbinding::glbinding"
		 GL X11 glfw
		 dl pthread
					;rt
					;imgui

					;"imgui::imgui"
					; "implot::implot"
					;"std::mdspan"
					;"xtensor"
					;"xtensor::optimize"
					;"xtensor::use_xsimd"

		 ))

					; (out "target_link_libraries ( mytest Threads::Threads )")
					;(out "target_precompile_headers( mytest PRIVATE vis_00_base.hpp )")
	  ))
      )))



