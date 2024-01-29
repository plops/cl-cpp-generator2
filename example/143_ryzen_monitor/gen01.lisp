(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more)))
  (setf *features* (set-exclusive-or *features* (list :more))))

(let ()
  (defparameter *source-dir* #P"example/143_ryzen_monitor/source01/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     (include "imgui.h"
	      "imgui_impl_glfw.h"
	      "imgui_impl_opengl3.h")
     (include<> GLFW/glfw3.h
		format
		iostream)
     (defun glfw_error_callback (err description)
       (declare (type int err)
		(type "const char*" description))
       ,(lprint :vars `(err description)))
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))

       (do0
	(glfwSetErrorCallback glfw_error_callback)
	(unless (glfwInit)
	  ,(lprint :msg "glfwInit failed")
	  (return 1))
	(let ((glsl_version (string "#version 130")))
	  (glfwWindowHint GLFW_CONTEXT_VERSION_MAJOR 4)
	  (glfwWindowHint GLFW_CONTEXT_VERSION_MINOR 6))
	(let ((window (glfwCreateWindow 1280 720
					(string "ryzen_mon_glgui")
					nullptr
					nullptr)))
	  (when (== nullptr window)
	    ,(lprint :msg "Can't open glfw window")
	    (return 1))
	  (glfwMakeContextCurrent window)
	  (glfwSwapInterval 1)
	  (IMGUI_CHECKVERSION)
	  (ImGui--CreateContext)
	  (let ((io (ImGui--GetIO)))
	    (setf io.ConfigFlags
		  (or io.ConfigFlags
		      ImGuiConfigFlags_NavEnableKeyboard)))
	  (ImGui--StyleColorsDark)
	  (ImGui_ImplGlfw_InitForOpenGL window true)
	  (ImGui_ImplOpenGL3_Init glsl_version))
	)

       (do0
	(let ((show_demo_window true)
	      (clear_color (ImVec4 .4s0 .5s0 .6s0 1s0)))
	  (while (!glfwWindowShouldClose window)
		 (glfwPollEvents)
		 (ImGui_ImplOpenGL3_NewFrame)
		 (ImGui_ImplGlfw_NewFrame)
		 (ImGui--NewFrame)
		 (when show_demo_window
		   (ImGui--ShowDemoWindow &show_demo_window))
		 (ImGui--Render)
		 (let ((w 0)
		       (h 0))
		   (glfwGetFramebufferSize window &w &h)
		   (glViewport 0 0 w h)
		   (glClearColor (* clear_color.x clear_color.w)
				 (* clear_color.y clear_color.w)
				 (* clear_color.z clear_color.w)
				 clear_color.w)
		   (glClear GL_COLOR_BUFFER_BIT)
		   (ImGui_ImplOpenGL3_RenderDrawData
		    (ImGui--GetDrawData))
		   (glfwSwapBuffers window)))))
       (do0
	(ImGui_ImplOpenGL3_Shutdown)
	(ImGui_ImplGlfw_Shutdown)
	(ImGui--DestroyContext)
	(glfwDestroyWindow window)
	(glfwTerminate)
	(return 0))
       ))
   :omit-parens t
   :format t
   :tidy nil))

