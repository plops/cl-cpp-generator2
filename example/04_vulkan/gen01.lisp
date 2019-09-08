(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

(progn
  (defun vk (params)
    (destructuring-bind (type var &rest args) params
     `(let ((,var (curly)))
	(declare (type ,type ,var))
	,@(loop for i from 0 below (length args) by 2 collect
	       (let ((keyword (elt args i))
		     (value (elt args (+ i 1))))
		 `(setf (dot ,var ,keyword) ,value))))))
  (defparameter *code-file* (asdf:system-relative-pathname 'cl-cpp-generator2 "example/04_vulkan/source/run_01_base.cpp"))
  (let* ((code
	  `(do0
	    "// https://vulkan-tutorial.com/en/Drawing_a_triangle/Setup/Base_code"
	    "// g++ -std=c++17 run_01_base.cpp  `pkg-config --static --libs glfw3` -lvulkan -o run_01_base"
	    " "
	    (do0 "#define GLFW_INCLUDE_VULKAN"
		 (include <GLFW/glfw3.h>)
		 " ")
	    #+nil
	    (include <vulkan/vulkan.h>)
	    
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
		     )
	    (defclass HelloTriangleApplication ()
	      "public:"
	      (defun run ()
		(declare (values void))
		(initWindow)
		(initVulkan)
		(mainLoop)
		(cleanup))
	      "private:"
	      (let ((_window)
		    (_instance))
		(declare (type GLFWwindow* _window)
			 (type VkInstance _instance))
	       (defun initWindow ()
		 (declare (values void))
		 (glfwInit)
		 (glfwWindowHint GLFW_CLIENT_API GLFW_NO_API)
		 (glfwWindowHint GLFW_RESIZABLE GLFW_FALSE)
		 (setf _window (glfwCreateWindow 800 600
						(string "vulkan window")
						nullptr
						nullptr)))
	       (defun createInstance ()
		 (declare (values void))
		 ,(vk `(VkApplicationInfo
		       appInfo
		       :sType VK_STRUCTURE_TYPE_APPLICATION_INFO
		       :pApplicationName (string "Hello Triangle")
		       :applicationVersion (VK_MAKE_VERSION 1 0 0)
		       :pEngineName (string "No Engine")
		       :engineVersion (VK_MAKE_VERSION 1 0 0)
		       :apiVersion VK_API_VERSION_1_0))
		 
		 (let ((glfwExtensionCount  0)
		       (glfwExtensions))
		   (declare (type uint32_t glfwExtensionCount)
			    (type "const char**" glfwExtensions))
		   (setf glfwExtensions (glfwGetRequiredInstanceExtensions
					 &glfwExtensionCount))

		   ,(vk `(VkInstanceCreateInfo
			  createInfo
			  :sType VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
			  :pApplicationInfo &appInfo
			  :enabledExtensionCount glfwExtensionCount
			  :ppEnabledExtensionNames glfwExtensions
			  :enabledLayerCount 0))
		   (unless (== VK_SUCCESS
			       (vkCreateInstance &createInfo
						 nullptr
						 &_instance))
		     (throw ("std::runtime_error"
			     (string "failed to create instance"))))))
	       (defun initVulkan ()
		 (declare (values void))
		 (createInstance)
		 )
	       (defun mainLoop ()
		 (declare (values void))
		 (while (not (glfwWindowShouldClose _window))
		   (glfwPollEvents)))
	       (defun cleanup ()
		 (declare (values void))
		 (glfwDestroyWindow _window)
		 (glfwTerminate)
		 ))
	      )
	    (defun main ()
	      (declare (values int))

	      (let ((app))
		(declare (type HelloTriangleApplication app))
		(handler-case
		    (app.run)
		  ("const std::exception&" (e)
		    (<< "std::cerr"
			(e.what)
			"std::endl")
		    (return EXIT_FAILURE)))
		(return EXIT_SUCCESS))
	      
	      
	     #+nnil
	      (let ()
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
		  
		  (return 0)))))))
    (write-source *code-file* code)))
 
