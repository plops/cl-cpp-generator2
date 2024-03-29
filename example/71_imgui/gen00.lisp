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
;;;<GLFW/glfw3.h>
				 )
			"class GLFWwindow;"
			"class ImVec4;"
			"class ImGuiIO;"

			)
     :implementation-preamble `(do0
				,log-preamble
				(include "imgui_impl_opengl3_loader.h"

					 "imgui.h"
					 "imgui_impl_glfw.h"
					 "imgui_impl_opengl3.h"

					 <GLFW/glfw3.h>
					 )
				(include "implot.h")
				)
     :code `(do0
	     (defclass MainWindow ()
	       "public:"
	       "bool show_demo_window_;"
					;"ImVec4* clear_color_;"
	       "ImGuiIO& io;"
	       (defmethod MainWindow ()
		 (declare
		  (explicit)
		  (construct
		   (show_demo_window_ true)
		   (io ((lambda ()
			  (declare (values ImGuiIO&))
			  (do0 (IMGUI_CHECKVERSION)
			       (ImGui--CreateContext)
			       (ImPlot--CreateContext)
			       (return (ImGui--GetIO))))))
		   #+nil (clear_color_ (new (ImVec4 .45s0
						    .55s0
						    .6s0
						    1s0)))
		   )
		  (values :constructor))

		 ,(lprint)
		 )
	       (defmethod ~MainWindow ()
		 (declare
		  (values :constructor))
		 ,(lprint)
		 #+nil (do0
			(ImPlot--DestroyContext)
			(ImGui--DestroyContext)))
	       (defmethod Init (window glsl_version )
		 (declare (type GLFWwindow* window)
			  (type "const char*" glsl_version))
		 ,(lprint)
		 (do0
		  #+nil(do0 (IMGUI_CHECKVERSION)
			    (ImGui--CreateContext)
			    (ImGui--GetIO))
		  (do0			;let ((&io (ImGui--GetIO)))
		   (comments "enable keyboard controls, docking multi-viewport")
		   ,@(loop for e in `(NavEnableKeyboard
				      DockingEnable
					;ViewportsEnable
				      )
			   collect
			   `(do0
			     (setf io.ConfigFlags
				   (logior io.ConfigFlags
					   ,(format nil "ImGuiConfigFlags_~a" e)))
			     ,(lprint :msg (format nil "~a" e)
				      :vars `(,(format nil "ImGuiConfigFlags_~a" e)
					       io.ConfigFlags
					       (dot (ImGui--GetIO)
						    ConfigFlags)))))
		   )
		  (do0
		   ,(lprint :msg "setup ImGUI style")
		   (ImGui--StyleColorsDark)
					;(ImGui--StyleColorsClassic)
		   (let ((style (ImGui--GetStyle)))
		     (when
			 (logand
			  io.ConfigFlags
			  ImGuiConfigFlags_ViewportsEnable)
		       (setf style.WindowRounding 0s0
			     (dot style (aref Colors ImGuiCol_WindowBg)
				  w)
			     1s0)))
		   (ImGui_ImplGlfw_InitForOpenGL window true)
		   (ImGui_ImplOpenGL3_Init glsl_version)
		   (let ((font_fn (string
				   ,(format nil "~a"
					    (elt
					     (directory "/home/martin/src/vcpkg/buildtrees/imgui/src/*/misc/fonts/DroidSans.ttf")
					     0))))
			 (font_size 16s0))
		     (let ((*font (-> io.Fonts (AddFontFromFileTTF font_fn font_size))))
		       (if (== font nullptr)
			   ,(lprint :msg "loading font failed" :vars `(font_fn font_size))
			   ,(lprint :msg "loaded font" :vars `(font_fn font_size)))))))
		 )
	       (defmethod NewFrame ()
		 (do0 (ImGui_ImplOpenGL3_NewFrame)
		      (ImGui_ImplGlfw_NewFrame)
		      (ImGui--NewFrame)
		      (ImGui--DockSpaceOverViewport)))
	       (defmethod Update ()
		 (do0
		  (when show_demo_window_
		    (ImGui--ShowDemoWindow &show_demo_window_)
		    (ImPlot--ShowDemoWindow))
		  (progn
		    (ImGui--Begin (string "hello"))
		    (ImGui--Checkbox (string "demo window")
				     &show_demo_window_)
		    (ImGui--Text (string "Application average %.3f ms/frame (%.1f FPS)")
				 (/ 1000s0 (dot (ImGui--GetIO) Framerate))
				 (dot (ImGui--GetIO) Framerate))
		    (ImGui--End))
		  )
		 (ImGui--EndFrame)
		 )
	       (defmethod Render (window)
		 (declare (type GLFWwindow* window))
		 (do0

		  (let ((screen_width (int 0))
			(screen_height (int 0)))
		    (glfwGetFramebufferSize window
					    &screen_width
					    &screen_height)
					;,(lprint :msg "framebuffer" :vars `(screen_width screen_height))
		    (glViewport 0 0 screen_width screen_height)
		    #+nil(glClearColor (* clear_color_->x clear_color_->w)
				       (* clear_color_->y clear_color_->w)
				       (* clear_color_->z clear_color_->w)
				       clear_color_->w)
		    (glClear GL_COLOR_BUFFER_BIT)
		    (ImGui--Render)
		    (ImGui_ImplOpenGL3_RenderDrawData
		     (ImGui--GetDrawData)))
		  (do0
		   (comments "update and render additional platform windows")
		   (when
		       (logand
			io.ConfigFlags
			ImGuiConfigFlags_ViewportsEnable)
		     (let ((*backup_current_context
			    (glfwGetCurrentContext)))
		       (ImGui--UpdatePlatformWindows)
		       (ImGui--RenderPlatformWindowsDefault)
		       (glfwMakeContextCurrent backup_current_context)))))
		 (glfwSwapBuffers window)
		 )
	       (defmethod Shutdown ()
		 (do0
		  ,(lprint)
		  (ImGui_ImplOpenGL3_Shutdown)
		  (ImGui_ImplGlfw_Shutdown)
		  (ImGui--DestroyContext))
		 ))))


    (write-source (asdf:system-relative-pathname
		   'cl-cpp-generator2
		   (merge-pathnames #P"main.cpp"
				    *source-dir*))
		  `(do0
		    (include "MainWindow.h")
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
		     (include "implot.h"
			      )
		     (comments "https://gist.github.com/TheOpenDevProject/1662fa2bfd8ef087d94ad4ed27746120")
		     (defclass+ DestroyGLFWwindow ()
		       "public:"
		       (defmethod "operator()" (ptr)
			 (declare (type GLFWwindow* ptr))
			 ,(lprint :msg "Destroy GLFW window context.")
			 (glfwDestroyWindow ptr)
			 (glfwTerminate))))

		    (defun main (argc argv)
		      (declare (type int argc)
			       (type char** argv)
			       (values int))
		      (setf g_start_time ("std::chrono::high_resolution_clock::now"))
		      ,(lprint :msg "start" :vars `(argc (aref argv 0)))
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
			 ,(lprint :msg "enable vsync")
			 (glfwSwapInterval 1))
		       (comments "imgui brings its own opengl loader"
				 "https://github.com/ocornut/imgui/issues/4445")

		       "MainWindow M;"
		       (M.Init (window.get) glsl_version)
		       (while (!glfwWindowShouldClose (window.get))
			 (do0
			  (glfwPollEvents)
			  (M.NewFrame)
			  (M.Update)
			  (M.Render  (window.get))
			  )))
		      (do0
		       ,(lprint :msg "cleanup")
		       (M.Shutdown)
		       #+nil (do0 (glfwDestroyWindow window)
				  (glfwTerminate)))
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
		(directory "source/*.cpp")
		(directory "/home/martin/src/vcpkg/buildtrees/implot/src/*/implot_demo.cpp")
					;(directory "/home/martin/src/vcpkg/buildtrees/imgui/src/*/backends/imgui_impl_opengl3.cpp")
		))

	  (out "add_executable( mytest ${SRCS} )")
	  (out "target_compile_features( mytest PUBLIC cxx_std_17 )")

	  (loop for e in `(imgui implot)
		do
		(out "find_package( ~a CONFIG REQUIRED )" e))

	  (out "target_link_libraries( mytest PRIVATE ~{~a~^ ~} )"
	       `("imgui::imgui"
		 "implot::implot"))

					; (out "target_link_libraries ( mytest Threads::Threads )")
					;(out "target_precompile_headers( mytest PRIVATE vis_00_base.hpp )")
	  ))
      )))



