(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *code-file* (asdf:system-relative-pathname 'cl-cpp-generator2 "example/04_vulkan/source/run_01_base.cpp"))
  (let* ((code
	  `(do0
	    "// https://vulkan-tutorial.com/en/Drawing_a_triangle/Setup/Base_code"
	    "// g++ -std=c++17 run_01_base.cpp  `pkg-config --static --libs glfw3` -lvulkan -o run_00_base"
	    " "
	    (do0 "#define GLFW_INCLUDE_VULKAN"
		 (include <GLFW/glfw3.h>)
		 " ")
	    
	    (do0
	     "#define GLM_FORCE_RADIANS"
	     "#define GLM_FORCE_DEPTH_ZERO_TO_ONE"
	     " "
	     (include <glm/vec4.hpp>
		      <glm/mat4x4.hpp>)
	     " "
	     )
	    
	    " "
	    (include <iostream>
		     <stdexcept>
		     <functional>
		     <cstdlib>
		     <vulkan/vulkan.h>)
	    (defclass HelloTriangleApplication ()
	      "public:"
	      (defun run ()
		(declare (values void))
		(initVulkan)
		(mainLoop)
		(cleanup))
	      "private:"
	      (defun initVulkan ()
		(declare (values void)))
	      (defun mainLoop ()
		(declare (values void)))
	      (defun cleanup ()
		(declare (values void)))
	      )
	    (defun main ()
	      (declare (values int))

	      (let ((app))
		(declare (type HelloTriangleApplication app))
		(handler-case
		    (app.run)
		  ("const std::exception&" (e)
		    (return EXIT_FAILURE)))
		(return EXIT_SUCCESS))
	      
	      (glfwInit)
	      (glfwWindowHint GLFW_CLIENT_API GLFW_NO_API)
	      (let ((window (glfwCreateWindow 800 600
					      (string "vulkan window")
					      nullptr
					      nullptr)))
		(let ((extensionCount 0))
		  (declare (type uint32_t extensionCount))
		  (vkEnumerateInstanceExtensionProperties
		   nullptr
		   &extensionCount
		   nullptr))
		(let ((matrix)
		      (vec)
		      (test (* matrix vec)))
		  (declare (type "glm::mat4" matrix)
			   (type "glm::vec4" vec))
		  (while (not (glfwWindowShouldClose window))
		    (glfwPollEvents))
		  (glfwDestroyWindow window)
		  (glfwTerminate)
		  (return 0)))))))
    (write-source *code-file* code)))
 
