(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

;; if nolog is off, then validation layers will be used to check for mistakes
;; if surface is on, then a window surface is created; otherwise only off-screen render
(setf *features* (union *features* '(:surface)))
(setf *features* (set-difference *features* '(:nolog)))

(progn
  (defun vk (params)
    "many vulkan functions get their arguments in the form of one or
more structs. this function helps to initialize those structs."
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
		     #+surface <set>
		     )
	    (defstruct0 QueueFamilyIndices 
		(graphicsFamily "std::optional<uint32_t>")
	      #+surface (presentFamily "std::optional<uint32_t>")
	      #+nil(defun isComplete ()
		(declare (values bool))
		(return (graphicsFamily.has_value))))
	    
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
		    (_physicalDevice VK_NULL_HANDLE)
		    (_device)
		    (_graphicsQueue)
		    #+surface (_presentQueue)
		    #+surface (_surface))
		(declare (type GLFWwindow* _window)
			 (type VkInstance _instance)
			 #-nolog (type "const bool" _enableValidationLayers)
			 #-nolog (type "const std::vector<const char*>" _validationLayers)
			 (type VkPhysicalDevice _physicalDevice)
			 (type VkDevice _device)
			 (type VkQueue _graphicsQueue
			       #+surface _presentQueue)
			 
			 #+surface (type VkSurfaceKHR _surface))
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
		     #+surface
		     (let ((presentSupport false))
		       (declare (type VkBool32 presentSupport))
		       (vkGetPhysicalDeviceSurfaceSupportKHR
			device i _surface &presentSupport)
		       (when (and (< 0 family.queueCount)
				  presentSupport)
			 (setf indices.presentFamily i)))
		     (incf i))))
		(return indices)))
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
		 "// initialize member _instance"
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
		 #+surface
		 (do0 "// create window surface because it can influence physical device selection"
		      (createSurface))
		 (pickPhysicalDevice)
		 (createLogicalDevice))
	       #+surface
	       (defun createSurface ()
		 (declare (values void))
		 "// initialize _surface member"
		 "// must be destroyed before the instance is destroyed"
		 (unless (== VK_SUCCESS
			     (glfwCreateWindowSurface
			      _instance _window
			      nullptr &_surface))
		   (throw ("std::runtime_error"
			     (string "failed to create window surface")))
		   ))
	       (defun createLogicalDevice ()
		 (declare (values void))
		 "// initialize members _device and _graphicsQueue"
		 (let ((indices (findQueueFamilies _physicalDevice))
		       (queuePriority 1s0))
		   (declare (type float queuePriority))
		   (let ((queueCreateInfos)
			 (uniqueQueueFamilies
			  (curly
			   (indices.graphicsFamily.value)
			   #+surface (indices.presentFamily.value))))
		     (declare (type "std::vector<VkDeviceQueueCreateInfo>"
				    queueCreateInfos)
			      (type "std::set<uint32_t>" uniqueQueueFamilies))
		     
		     (foreach (queueFamily uniqueQueueFamilies)
			      ,(vk `(VkDeviceQueueCreateInfo
				       queueCreateInfo
				       :sType VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO
				       :queueFamilyIndex queueFamily
				       :queueCount 1
				       :pQueuePriorities &queuePriority))
			      (queueCreateInfos.push_back queueCreateInfo)))
		   (let ((deviceFeatures (curly))
			 )
		     (declare (type VkPhysicalDeviceFeatures deviceFeatures))
		     ,(vk `(VkDeviceCreateInfo
			 createInfo
			 :sType VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO
			 :pQueueCreateInfos (queueCreateInfos.data)
			 :queueCreateInfoCount (static_cast<uint32_t>
						(queueCreateInfos.size))
			 :pEnabledFeatures &deviceFeatures
			 :enabledExtensionCount 0
			 :enabledLayerCount
			 #-nolog (static_cast<uint32_t> (_validationLayers.size))
			 #+nolog 0
			 #-nolog :ppEnabledLayerNames #-nolog (_validationLayers.data)))
		     (unless (== VK_SUCCESS
				 (vkCreateDevice _physicalDevice &createInfo
						 nullptr &_device))
		       (throw ("std::runtime_error" (string "failed to create logical device"))))
		     (vkGetDeviceQueue _device (indices.graphicsFamily.value)
				       0 &_graphicsQueue)
		     #+surface
		     (vkGetDeviceQueue _device (indices.presentFamily.value)
				       0 &_presentQueue))))
	       (defun isDeviceSuitable ( device)
		 (declare (values bool)
			  (type VkPhysicalDevice device))
		 (let ((indices (findQueueFamilies device)))
		   (declare (type QueueFamilyIndices indices))
		   (return (and (indices.graphicsFamily.has_value)
				#+surface (indices.presentFamily.has_value)))
		   #+nil (return (indices.isComplete))))
	       (defun pickPhysicalDevice ()
		 
		 (declare (values void))
		 "// initialize member _physicalDevice"
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
		 (vkDestroyDevice _device nullptr)
		 #+surface
		 (vkDestroySurfaceKHR _instance _surface nullptr)
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
 
