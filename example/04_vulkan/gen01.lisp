(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

;(setf *features* (union *features* '(:nolog)))
(setf *features* (set-difference *features* '(:nolog)))

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
	    "// https://vulkan-tutorial.com/en/Drawing_a_triangle/Setup/Validation_layers"
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
		     <cstring>
		     <optional>
		     )
	    (defstruct0 QueueFamilyIndices 
	      (graphicsFamily "std::optional<uint32_t>")
	      #+nil(defun isComplete ()
		(declare (values bool))
		(return (graphicsFamily.has_value))))
	    (defun findQueueFamilies (device)
	      (declare (type VkPhysicalDevice device)
		       (values QueueFamilyIndices))
	      (let ((indices)
		    (queueFamilyCount 0))
		(declare (type QueueFamilyIndices indices)
			 (type uint32_t queueFamilyCount))
		(vkGetPhysicalDeviceQueueFamilyProperties
		 device &queueFamilyCount nullptr)
		(let (((queueFamilies queueFamilyCount)))
		  (declare (type "std::vector<VkQueueFamilyProperties>"
				 (queueFamilies queueFamilyCount)))
		  (vkGetPhysicalDeviceQueueFamilyProperties
		   device
		   &queueFamilyCount
		   (queueFamilies.data))
		  (let ((i 0))
		    (foreach
		     (family queueFamilies)
		     (when (and (< 0 family.queueCount)
				(logand family.queueFlags
					VK_QUEUE_GRAPHICS_BIT))
		       (setf indices.graphicsFamily i))
		     (incf i))))
		(return indices)))
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
		    (_instance)
		    ;#-nolog (_enableValidationLayers true)
		    #-nolog (_validationLayers (curly (string "VK_LAYER_KHRONOS_validation")))
		    (_physicalDevice VK_NULL_HANDLE))
		(declare (type GLFWwindow* _window)
			 (type VkInstance _instance)
			 #-nolog (type "const bool" _enableValidationLayers)
			 #-nolog (type "const std::vector<const char*>" _validationLayers)
			 (type VkPhysicalDevice _physicalDevice))
	       #-nolog (defun checkValidationLayerSupport ()
		 (declare (values bool))
		 (let ((layerCount 0))
		   (declare (type uint32_t layerCount))
		   (vkEnumerateInstanceLayerProperties &layerCount nullptr)
		   (let (((availableLayers layerCount)))
		     (declare (type "std::vector<VkLayerProperties>"
				    (availableLayers layerCount)))
		     (vkEnumerateInstanceLayerProperties
		      &layerCount
		      (availableLayers.data))

		     (foreach
		      (layerName _validationLayers)
		      (let ((layerFound false))
			(foreach
			 (layerProperties availableLayers)
			 (when (== 0 (strcmp layerName layerProperties.layerName))
			   (setf layerFound true)
			   break))
			(unless layerFound
			  (return false))))
		     (return true))))
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
		 #-nolog (;when (and _enableValidationLayers  (not (checkValidationLayerSupport)))
			  unless (checkValidationLayerSupport)
			   (throw ("std::runtime_error"
				   (string "validation layers requested, but unavailable."))))
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
			  :enabledLayerCount
			  #+nolog 0
			  #-nolog ("static_cast<uint32_t>"
				   (_validationLayers.size))
			  #-nolog :ppEnabledLayerNames
			  #-nolog (_validationLayers.data)))
		   (unless (== VK_SUCCESS
			       (vkCreateInstance &createInfo
						 nullptr
						 &_instance))
		     (throw ("std::runtime_error"
			     (string "failed to create instance"))))))
	       (defun initVulkan ()
		 (declare (values void))
		 (createInstance)
		 (pickPhysicalDevice))
	       (defun isDeviceSuitable ( device)
		 (declare (values bool)
			  (type VkPhysicalDevice device))
		 (let ((indices (findQueueFamilies device)))
		   (declare (type QueueFamilyIndices indices))
		   (return (indices.graphicsFamily.has_value))
		   #+nil (return (indices.isComplete)))
		 )
	       (defun pickPhysicalDevice ()
		 (declare (values void))
		 (let ((deviceCount 0))
		   (declare (type uint32_t deviceCount))
		   (vkEnumeratePhysicalDevices _instance &deviceCount nullptr)
		   (when (== 0 deviceCount)
		     (throw ("std::runtime_error"
			     (string "failed to find gpu with vulkan support."))))
		   (let (((devices deviceCount)))
		     (declare (type "std::vector<VkPhysicalDevice>"
				    (devices deviceCount)))
		     (vkEnumeratePhysicalDevices _instance &deviceCount
						 (devices.data))
		     (foreach (device devices)
			      (when (isDeviceSuitable device)
				(setf _physicalDevice device)
				break))
		     (when (== VK_NULL_HANDLE
			       _physicalDevice)
		       (throw ("std::runtime_error"
			       (string "failed to find a suitable gpu."))))))
		 )
	       (defun mainLoop ()
		 (declare (values void))
		 (while (not (glfwWindowShouldClose _window))
		   (glfwPollEvents)))
	       (defun cleanup ()
		 (declare (values void))
		 (vkDestroyInstance _instance nullptr)
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
 
