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
     :implementation-preamble `(do0
				,log-preamble
				)
     :code `(do0
	     (defclass MainWindow ()
	       "public:"

	       (defmethod MainWindow ()
		 (declare
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
		    (do0
		     (include "imgui_impl_opengl3_loader.h"
			      "imgui.h"
			      "imgui_impl_glfw.h"
			      "imgui_impl_opengl3.h"

			      <GLFW/glfw3.h>)
		     (comments "https://gist.github.com/TheOpenDevProject/1662fa2bfd8ef087d94ad4ed27746120")
		     (defclass+ DestroyGLFWwindow ()
		       "public:"
		       (defmethod "operator()" (ptr)
			 (declare (type GLFWwindow* ptr))
			 ,(lprint :msg "Destroy GLFW window context.")
			 (glfwDestroyWindow ptr))))

		    (defun main (argc argv)
		      (declare (type int argc)
			       (type char** argv)
			       (values int))
		      (setf g_start_time ("std::chrono::high_resolution_clock::now"))
		      (do0
		       (comments "glfw initialization")
		       (comments "https://github.com/ocornut/imgui/blob/docking/examples/example_glfw_opengl3/main.cpp")
		       (glfwSetErrorCallback (lambda (err description)
					       (declare (type int err)
							(type "const char*" description))
					       ,(lprint :msg "glfw error" :vars `(err description))))
		       (unless (glfwInit)
			 ,(lprint :msg "glfwInit failed."))
		       "const char* glsl_version = \"#version 130\";"
		       (glfwWindowHint GLFW_CONTEXT_VERSION_MAJOR 3)
		       (glfwWindowHint GLFW_CONTEXT_VERSION_MINOR 0)
		       "std::unique_ptr<GLFWwindow,DestroyGLFWwindow> window;"
		       (let ((w (glfwCreateWindow 1280 720
						  (string "dear imgui example")
						  nullptr nullptr)))
			 (when (== nullptr w)
			   ,(lprint :msg "glfwCreatWindow failed."))
			 (window.reset w)
			 (glfwMakeContextCurrent (window.get))
			 (glfwSwapInterval 1))
		       (comments "imgui brings its own opengl loader"
				 "https://github.com/ocornut/imgui/issues/4445")

		       (let ((screen_width (int 0))
			     (screen_height (int 0)))
			 (glfwGetFramebufferSize (window.get)
						 &screen_width
						 &screen_height)
			 ,(lprint :msg "framebuffer" :vars `(screen_width screen_height))
			 (glViewport 0 0 screen_width screen_height)))


		      ,(lprint :msg "leave program")

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
	       (append
		(directory "source/*.cpp")
		(directory "/home/martin/src/vcpkg/buildtrees/imgui/src/*/backends/imgui_impl_opengl3.cpp")))

	  (out "add_executable( mytest ${SRCS} )")
	  (out "target_compile_features( mytest PUBLIC cxx_std_17 )")

	  (loop for e in `(imgui)
		do
		(out "find_package( ~a CONFIG REQUIRED )" e))

	  (out "target_link_libraries( mytest PRIVATE ~{~a~^ ~} )"
	       `("imgui::imgui"))

					; (out "target_link_libraries ( mytest Threads::Threads )")
					;(out "target_precompile_headers( mytest PRIVATE vis_00_base.hpp )")
	  ))
      )))



