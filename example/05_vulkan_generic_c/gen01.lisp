(setf *features* (union *features* '(:generic-c)))

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre"))

(in-package :cl-cpp-generator2)

;; if nolog is off, then validation layers will be used to check for mistakes
;; if surface is on, then a window surface is created; otherwise only off-screen render
;; if nolog-frame is off then draw frame prints lots of stuff
(setf *features* (union *features* '(:surface  :nolog-frame
				     ;:nolog
				     )))
(setf *features* (set-difference *features* '(:nolog ;:nolog-frame
					      )))

;; gcc -std=c18 -c vulkan_00_main.c -Wmissing-declarations
;; gcc -std=c18 -c vulkan_01_instance.c -Wmissing-declarations

;; git clone https://github.com/recp/cglm
;; 

(progn
  ;; make sure to run this code twice during the first time, so that
  ;; the functions are defined
  (defparameter *code-file* (asdf:system-relative-pathname 'cl-cpp-generator2 "example/05_vulkan_generic_c/source/run_01_base.c"))
  (defparameter *vertex-file* (asdf:system-relative-pathname 'cl-cpp-generator2 "example/05_vulkan_generic_c/source/run_01_base.vert"))
  (defparameter *frag-file* (asdf:system-relative-pathname 'cl-cpp-generator2 "example/05_vulkan_generic_c/source/run_01_base.frag"))


  (progn
    (defun with-single-time-commands (args)
      (destructuring-bind ((buffer) &rest body) args
	`(let ((,buffer
		(beginSingleTimeCommands)))
	   ,@body
	   (endSingleTimeCommands ,buffer))))


    (defun vkprint (msg
		    &optional rest)
      ;;"{sec}.{nsec} {__FILE__}:{__LINE__} {__func__}"
      (let* ((m `(string ,(format nil " ~a: " msg)))
	     (l `(((string "%6.6f") (- current_time ,(g `_start_time)))
					;((printf_dec_format tp.tv_sec) tp.tv_sec)
					;((string "."))
					;((printf_dec_format tp.tv_nsec) tp.tv_nsec)
		  ((string " "))
		  ((printf_dec_format __FILE__) __FILE__)
		  ((string ":"))
		  ((printf_dec_format __LINE__) __LINE__)
		  ((string " "))
		  ((printf_dec_format __func__) __func__)
		  (,m)
		  ,@(loop for e in rest appending
			 `(((string ,(format nil " ~a=" (emit-c :code e))))
			   ((printf_dec_format ,e) ,e)
			   ((string " (%s)") (type_string ,e))
			   ))
		  ((string "\\n")))))
	`(progn
	   (let (;(tp)
		 (current_time (now)))
	     ;(declare (type "struct timespec" tp))
	     ;; https://stackoverflow.com/questions/6749621/how-to-create-a-high-resolution-timer-in-linux-to-measure-program-performance
	     ;(clock_gettime CLOCK_REALTIME &tp)
	     ,@(loop for e in l collect
		    (destructuring-bind (fmt &optional value) e
		      (if value
			  `(printf ,fmt ,value)
			  `(printf ,fmt))))))))
    
    (defun vkthrow (cmd)
      `(unless (== VK_SUCCESS
		   ,cmd)
	 #+nil "// throw"
	 ,(vkprint
	   (substitute #\Space #\Newline (format nil "failed to ~a" cmd)))
	 #+nil(
	       throw ("std::runtime_error"
		      (string ,(substitute #\Space #\Newline (format nil "failed to ~a" cmd)))))))
    

    

    (progn
      
      (defun vk-info-type (verb subject &key (prefix "Vk")
					  (suffix "Info"))
	"convert two lisp symbols like allocate command-buffer  to vkCommandBufferAllocate"
	(format nil "~a~{~a~}~{~a~}~a"
		prefix
		(mapcar #'string-capitalize
			(cl-ppcre:split "-" (format nil "~a" subject)))
		(mapcar #'string-capitalize
			(cl-ppcre:split "-" (format nil "~a" verb)))
		suffix))
      (defun vk-info-stype (verb subject &key (prefix "VK_STRUCTURE_TYPE_")
					   (suffix "_INFO"))
	"convert a lisp symbol like allocate command-buffer to VK_COMMAND_BUFFER_ALLOCATE_INFO"
	(format nil "~a~{~a~^_~}_~{~a~^_~}~a"
		prefix
		(mapcar #'string-upcase
			(cl-ppcre:split "-" (format nil "~a" subject)))
		(mapcar #'string-upcase
			(cl-ppcre:split "-" (format nil "~a" verb)))
		suffix))
      (defun vk-info-function (verb subject &key  (prefix "vk")
					      (suffix ""))
	"convert a lisp symbol like allocate command-buffer to vkAllocateCommandBuffers. use suffix to create plural"
	(format nil "~a~{~a~}~{~a~}~a"
		prefix
		(mapcar #'string-capitalize
			(cl-ppcre:split "-" (format nil "~a" verb)))
		(mapcar #'string-capitalize
			(cl-ppcre:split "-" (format nil "~a" subject)))
		suffix))
      (defun vkcall (params &key (plural nil)  (throw nil) (khr nil))
	"this macro helps to initialize an info object and call the corresponding vulkan function. splitting the command into verb subject and the plural argument seem to be enough automatically generate structure type names function names and the sType for a large subset of vulkan. subject is command-buffer, verb is create, info-params is a property list with member settings for the info struct and args a list that will be used in the call to the function. use khr to indicate that KHR should be appended to function and _KHR to the sType constant. the optional instance is used to print a relevant instance address in the debug message."
	(destructuring-bind (verb subject info-params args &optional instance)
	    params
	  `(progn
	     ,(vk `(,(vk-info-type
		      verb subject :suffix (if khr
					       (format nil "Info~a" khr)
					       "Info"))
		     info
		     :sType ,(vk-info-stype
			      verb subject :suffix (if khr
						       (format nil "_INFO_~a" khr)
						       "_INFO"))
		     ,@info-params)
		  )
	     ,(let ((suffix (if plural
				(if (stringp plural)
				    plural
				    "s")
				"")))
		(if khr
		    (setf suffix (format nil "~aKHR" suffix)))
		`(do0
		  
		  ,(if throw
		       (vkthrow
			`(,(vk-info-function verb subject
					     :suffix suffix)
			   ,@args))
		       `(,(vk-info-function verb subject
					    :suffix suffix)
			  ,@args))
					
		  ,(if instance
		       (vkprint (format nil " ~a ~a" verb subject)
				`(,(emit-c :code instance)))
		       (vkprint (format nil " ~a ~a" verb subject))))))))
      
      
      (defun set-members (params)
	"setf on multiple member variables of an instance"
	(destructuring-bind (instance &rest args) params
	  `(setf ,@(loop for i from 0 below (length args) by 2 appending
			(let ((keyword (elt args i))
			      (value (elt args (+ i 1))))
			  `((dot ,instance ,keyword) ,value))))))
      (defun vk (params)
	"many vulkan functions get their arguments in the form of one or
more structs. this function helps to initialize those structs."
	(destructuring-bind (type var &rest args) params
	  `(let ((,var (curly)))
	     (declare (type ,type ,var))
	     ,@(loop for i from 0 below (length args) by 2 collect
		    (let ((keyword (elt args i))
			  (value (elt args (+ i 1))))
		      `(setf (dot ,var ,keyword) ,value))))))))
  (progn
    ;; collect code that will be emitted in utils.h
    (defparameter *utils-code* nil)
    (defun emit-utils (&key code)
      (push code *utils-code*)
      " "))
  (progn
    (defparameter *module-global-parameters* nil)
    (defparameter *module* nil)

    (defun emit-globals (&key init)
      (let ((l `(
		 (_start_time double)
		 ;; 
		 (_window GLFWwindow* NULL)   
		 (_instance VkInstance)
					;#-nolog (_enableValidationLayers "const _Bool" )
		 #-nolog (_validationLayers[1] "const char* const" (curly (string "VK_LAYER_KHRONOS_validation")))
		 (_physicalDevice VkPhysicalDevice VK_NULL_HANDLE)
		 (_device VkDevice)
		 (_graphicsQueue VkQueue)
		 ;; 
		 (_msaaSamples VkSampleCountFlagBits)
		 (_colorImage VkImage)
		 (_colorImageMemory VkDeviceMemory)
		 (_colorImageView VkImageView)
		 ;;
		 (_mipLevels uint32_t)
		 (_textureImage VkImage)
		 (_textureImageMemory  VkDeviceMemory)
		 (_textureImageView VkImageView)
		 (_textureSampler VkSampler)
		 ;;
		 (_presentQueue  VkQueue)
		 (_surface VkSurfaceKHR)
		 (_deviceExtensions[1] "const char* const" (curly "VK_KHR_SWAPCHAIN_EXTENSION_NAME"
								  ))
		 (_swapChain VkSwapchainKHR)
					;(_N_IMAGES "const int" 4) ;; swapChainSupport.capabilities.maxImageCount
		 (_swapChainImages[_N_IMAGES] VkImage)
		 (_swapChainImageFormat VkFormat)
		 (_swapChainExtent VkExtent2D)
		 (_swapChainImageViews[_N_IMAGES] VkImageView)
		 (_descriptorSetLayout VkDescriptorSetLayout)
		 (_pipelineLayout VkPipelineLayout)
		 (_renderPass VkRenderPass)
		 (_graphicsPipeline VkPipeline)
		 (_swapChainFramebuffers[_N_IMAGES] VkFramebuffer)
		 (_commandPool VkCommandPool)
		 (_commandBuffers[_N_IMAGES] VkCommandBuffer)
					;(_MAX_FRAMES_IN_FLIGHT "const int" 2)
		 (_imageAvailableSemaphores[_MAX_FRAMES_IN_FLIGHT] VkSemaphore)
		 (_renderFinishedSemaphores[_MAX_FRAMES_IN_FLIGHT] VkSemaphore)
	       
		 (_currentFrame size_t)
		 (_inFlightFences[_MAX_FRAMES_IN_FLIGHT] VkFence)
		 (_framebufferResized _Bool)
		 (_vertexBuffer VkBuffer)
		 (_indexBuffer VkBuffer)
		 (_vertexBufferMemory VkDeviceMemory)
		 (_indexBufferMemory VkDeviceMemory)
		 (_uniformBuffers[_N_IMAGES] VkBuffer)
		 (_uniformBuffersMemory[_N_IMAGES] VkDeviceMemory)
		 (_descriptorPool VkDescriptorPool)
		 (_descriptorSets[_N_IMAGES] VkDescriptorSet)

		 (_depthImage VkImage)
		 (_depthImageMemory VkDeviceMemory)
		 (_depthImageView VkImageView)

		 (_vertices Vertex*)
		 (_num_vertices int)
		 (_indices uint32_t*)
		 (_num_indices int)
		 )))
	(if init
	    `(curly
	      ,@(remove-if
		 #'null
		 (loop for e in l collect
		      (destructuring-bind (name type &optional value) e
			(when value
			  `(= ,(format nil ".~a" (elt (cl-ppcre:split "\\[" (format nil "~a" name)) 0)) ,value))))))
	    `(do0
	      "enum {_N_IMAGES=4,_MAX_FRAMES_IN_FLIGHT=2};"
	      (defstruct0 State
		  ,@(loop for e in l collect
			 (destructuring-bind (name type &optional value) e
			   `(,name ,type))))))))
  
    (defun define-module (args)
      "each module will be written into a c file with module-name. the global-parameters the module will write to will be specified with their type in global-parameters. a file global.h will be written that contains the parameters that were defined in all modules. global parameters that are accessed read-only or have already been specified in another module need not occur in this list (but can). the prototypes of functions that are specified in a module are collected in functions.h. i think i can (ab)use gcc's warnings -Wmissing-declarations to generate this header. i split the code this way to reduce the amount of code that needs to be recompiled during iterative/interactive development. if the module-name contains vulkan, include vulkan headers. if it contains glfw, include glfw headers."
      (destructuring-bind (module-name global-parameters module-code) args
	(let ((header ()))
	
	  (push `(do0
		
		  " "
		  "#define GLFW_INCLUDE_VULKAN"
		  (include <GLFW/glfw3.h>)
		  " "
		  (include "utils.h")
		  " "
		  (include "globals.h")
		  " "
		  (include "proto2.h")
			
		  " "
		  )
		header)
	  (unless (cl-ppcre:scan "main" (string-downcase (format nil "~a" module-name)))
	    (push `(do0 "extern State state;")
		  header)
	    )
	
	  (push `(:name ,module-name :code (do0 ,@(reverse header) ,module-code))
		*module*))
	(loop for par in global-parameters do
	     (destructuring-bind (parameter-name
				  &key (direction 'in)
				  (type 'int)
				  (default nil)) par
	       (push `(:name ,parameter-name :type ,type :default ,default)
		     *module-global-parameters*))))))
  (defun g (arg)
    `(dot state ,arg))
  (define-module
      `(main ()
	     (do0
	      (let ((state ,(emit-globals :init t)))
		(declare (type State state)))
	      (defun mainLoop ()
		,(vkprint "mainLoop")
		
		(while (not (glfwWindowShouldClose ,(g `_window)))
		  (glfwPollEvents)
		  (drawFrame)
		  )
		,(vkprint "wait for gpu before cleanup")
		(vkDeviceWaitIdle ,(g `_device)) ;; wait for gpu before cleanup
		)
	      (defun run ()
		(initWindow)
		(initVulkan)
		(mainLoop)
		(cleanup)
		)
	      
	      (defun main ()
		(declare (values int))
		(setf ,(g `_start_time) (now))
		(run)))))
  
  (define-module
      `(instance
	((_window :direction 'out :type VkInstance) )
	(do0
	 (include <string.h>)
	 (defun cleanupInstance ()
	   (vkDestroyInstance ,(g `_instance) NULL))
	 #-nolog (defun checkValidationLayerSupport ()
		   (declare (values _Bool))
		   
		   (let ((layerCount 0))
		     (declare (type uint32_t layerCount))
		     (vkEnumerateInstanceLayerProperties &layerCount NULL)
		     (let ((availableLayers[layerCount]))
		       (declare (type "VkLayerProperties"
				      availableLayers[layerCount]
				      ))
		       (vkEnumerateInstanceLayerProperties
			&layerCount
			availableLayers)

		       (foreach
			(layerName ,(g `_validationLayers))
			(let ((layerFound false))
			  (foreach
			   (layerProperties availableLayers)
			   (when (== 0 (strcmp layerName layerProperties.layerName))
			     ,(vkprint "look for layer"
				       `(layerName))
			     (setf layerFound true)
			     break))
			  (unless layerFound
			    (return false))))
		       (return true))))
	 (defun createInstance ()
	   (declare (values void))
	   "// initialize member _instance"
	   #-nolog ( ;when (and _enableValidationLayers  (not (checkValidationLayerSupport)))
		    unless (checkValidationLayerSupport)
		    "// throw"
		    ,(vkprint "validation layers requested, but unavailable." `())
		    #+nil(throw ("std::runtime_error"

				 )))
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
	     ,(vkcall
	       `(create
		 instance
		 (:pApplicationInfo &appInfo
				    :enabledExtensionCount glfwExtensionCount
				    :ppEnabledExtensionNames glfwExtensions
				    :enabledLayerCount
				    #+nolog 0
				    #-nolog (length ,(g `_validationLayers))
				    :ppEnabledLayerNames
				    #+nolog NULL
				    #-nolog ,(g `_validationLayers))
		 (&info
		  NULL
		  (ref ,(g `_instance)))
		 ,(g `_instance)
		 )
	       :throw t))))))
  
  (define-module
      `(init
	()
	(do0
	 (defun initVulkan ()
	   (declare (values void))
	   (createInstance)
	   (do0 "// create window surface because it can influence physical device selection"
		(createSurface))
	   (pickPhysicalDevice)
	   (createLogicalDevice)
	   
	   ,(let ((l`(
		      (createSwapChain)
		      (createImageViews)
		      (createRenderPass)
		      (createDescriptorSetLayout)
		      (createGraphicsPipeline)
		      (createCommandPool)
		      ;; create texture image needs command pools
		      (createColorResources)
		      (createDepthResources)
		      (createFramebuffers)
		      (createTextureImage)
		      (createTextureImageView)
		      (createTextureSampler)
		      (loadModel)
		      (createVertexBuffer)
		      (createIndexBuffer)
		      (createUniformBuffers)
		      (createDescriptorPool)
		      (createDescriptorSets)
		      (createCommandBuffers)
		      (createSyncObjects))))
	      `(do0
		,@(loop for (e) in l collect
		       `(do0
			 
			 ,(vkprint (format nil " call ~a" e))
			 (,e)))))))))
  (define-module
      `(glfw_window
	((_window :direction 'out :type GLFWwindow* ) )
	(do0
	 (defun framebufferResizeCallback (window width height)
	   (declare (values "static void")
		    ;; static because glfw doesnt know how to call a member function with a this pointer
		    (type GLFWwindow* window)
		    (type int width height))
	   (let ((app ("(State*)" (glfwGetWindowUserPointer window))))
	     (setf app->_framebufferResized true)))
	 (defun initWindow ()
	   (declare (values void))
	   (glfwInit)
	   (glfwWindowHint GLFW_CLIENT_API GLFW_NO_API)
	   (glfwWindowHint GLFW_RESIZABLE GLFW_FALSE)
	   (setf ,(g `_window) (glfwCreateWindow 800 600
						 (string "vulkan window")
						 NULL
						 NULL))
	   ;; store this pointer to the instance for use in the callback
	   (glfwSetWindowUserPointer ,(g `_window) (ref state))
	   (glfwSetFramebufferSizeCallback ,(g `_window)
					   framebufferResizeCallback))
	 (defun cleanupWindow ()
	   (declare (values void))
	   (glfwDestroyWindow ,(g `_window))
	   (glfwTerminate)
	   ))))
  (define-module
      `(surface
	()
	(do0
	 (defun cleanupSurface ()
	   (vkDestroySurfaceKHR ,(g `_instance) ,(g `_surface) NULL)
	   )
	 (defun createSurface ()
	   "// initialize _surface member"
	   "// must be destroyed before the instance is destroyed"
	   ,(vkthrow `(glfwCreateWindowSurface
		       ,(g `_instance) ,(g `_window)
		       NULL (ref ,(g `_surface))))))))
  (define-module
      `(physical_device
	()
	(do0
	 (include <stdlib.h>)
	 (include <string.h>)
	 (defun cleanupPhysicalDevice ()
	   
	   )
	 (do0
	  ,(emit-utils :code
		       `(defstruct0 QueueFamilyIndices
			    ;;"// initialized to -1, i.e. no value"
			    (graphicsFamily int)
			  (presentFamily int)
			  ))
	  (defun "QueueFamilyIndices_isComplete" (q)
	    (declare (values _Bool)
		     (type QueueFamilyIndices q))
	    (return (and
		     (!= -1 q.graphicsFamily)
		     (!= -1 q.presentFamily))))
	  (defun "QueueFamilyIndices_make" ()
	    (declare (values QueueFamilyIndices))
	    (let ((q		 ;(malloc (sizeof QueueFamilyIndices))
		   ))
	      (declare (type QueueFamilyIndices q))
	      (setf q.graphicsFamily -1
		    q.presentFamily -1)
	      (return q)))
	  (defun "QueueFamilyIndices_destroy" (q)
	    (declare (type QueueFamilyIndices* q))
	    (free q)
	    )
	  (defun findQueueFamilies (device)
	    (declare (type VkPhysicalDevice device)
		     (values QueueFamilyIndices))
	    (let ((indices (QueueFamilyIndices_make))
		  (queueFamilyCount 0))
	      (declare		    ;(type QueueFamilyIndices indices)
	       (type uint32_t queueFamilyCount))
	      (vkGetPhysicalDeviceQueueFamilyProperties
	       device &queueFamilyCount NULL)
	      (let ((queueFamilies[queueFamilyCount]))
		(declare (type VkQueueFamilyProperties
			       queueFamilies[queueFamilyCount]))
		(vkGetPhysicalDeviceQueueFamilyProperties
		 device
		 &queueFamilyCount
		 queueFamilies)
		(let ((i 0))
		  (foreach
		   (family queueFamilies)
		   (when (and (< 0 family.queueCount)
			      (logand family.queueFlags
				      VK_QUEUE_GRAPHICS_BIT))
		     (setf indices.graphicsFamily i))
		   (let ((presentSupport false))
		     (declare (type VkBool32 presentSupport))
		     (vkGetPhysicalDeviceSurfaceSupportKHR
		      device i ,(g `_surface) &presentSupport)
		     (when (and (< 0 family.queueCount)
				presentSupport)
		       (setf indices.presentFamily i))
		     (when (QueueFamilyIndices_isComplete indices)
		       break))
		   (incf i))))
	      (return indices))))
	 (do0
	  ,(emit-utils :code
		       `(defstruct0 SwapChainSupportDetails
			    (capabilities VkSurfaceCapabilitiesKHR)
			  (formatsCount int)
			  (formats VkSurfaceFormatKHR*)
			  (presentModesCount int)
			  (presentModes VkPresentModeKHR*)
			  ))
	  (defun cleanupSwapChainSupport (details)
	    (declare (type SwapChainSupportDetails* details))
	    (free details->formats)
	    (free details->presentModes)
	    (setf details->formatsCount 0
		  details->presentModesCount 0))
	  (defun querySwapChainSupport (device )
	    (declare (values SwapChainSupportDetails)
		     (type VkPhysicalDevice device))
	    (let ((details (curly (= ,(format nil ".~a" `formatsCount) 0)
				  (= ,(format nil ".~a" `presentModesCount) 0)))
		  (s ,(g `_surface)))
	      (declare (type SwapChainSupportDetails details))
	      (vkGetPhysicalDeviceSurfaceCapabilitiesKHR
	       device
	       s
	       &details.capabilities)
	      (let ((formatCount 0))
		(declare (type uint32_t formatCount))
		(vkGetPhysicalDeviceSurfaceFormatsKHR device s &formatCount
						      NULL)
		(unless (== 0 formatCount)
		  (let ((n_bytes_details_format (* (sizeof VkSurfaceFormatKHR) formatCount)))
		    ,(vkprint "malloc" `(n_bytes_details_format))
		   (setf details.formatsCount formatCount
			 details.formats (malloc n_bytes_details_format)))
		  
		  (vkGetPhysicalDeviceSurfaceFormatsKHR
		   device s &formatCount
		   details.formats)))
	      (let ((presentModeCount 0))
		(declare (type uint32_t presentModeCount))
		(vkGetPhysicalDeviceSurfacePresentModesKHR
		 device s &presentModeCount
		 NULL)
		(unless (== 0 presentModeCount)
		  (let ((n_bytes_presentModeCount (* (sizeof VkPresentModeKHR) presentModeCount)))
		    ,(vkprint "malloc" `(n_bytes_presentModeCount))
		    (setf details.presentModesCount presentModeCount
			  details.presentModes ("(VkPresentModeKHR*)" (malloc n_bytes_presentModeCount))))
		  (vkGetPhysicalDeviceSurfacePresentModesKHR
		   device s &presentModeCount
		   details.presentModes)))
	      (return details))))

	 (defun isDeviceSuitable (device)
	   (declare (values bool)
		    (type VkPhysicalDevice device))
	       
	   (let ((extensionsSupported (checkDeviceExtensionSupport device))
		 (swapChainAdequate false))
	     (declare (type bool swapChainAdequate))
	     (when extensionsSupported
	       (let ((swapChainSupport (querySwapChainSupport device)))
		 (setf swapChainAdequate
		       (and (not (== 0 swapChainSupport.formatsCount))
			    (not (== 0 swapChainSupport.presentModesCount))))
		 (cleanupSwapChainSupport &swapChainSupport))))
	   (let ((indices (findQueueFamilies device))
		 (supportedFeatures))
	     (declare		    ;(type QueueFamilyIndices indices)
	      (type VkPhysicalDeviceFeatures
		    supportedFeatures))
	     (vkGetPhysicalDeviceFeatures device
					  &supportedFeatures)
	     (let ((res (and (QueueFamilyIndices_isComplete indices)
			     supportedFeatures.samplerAnisotropy
			     (and 
			      extensionsSupported
			      swapChainAdequate))))
					;(QueueFamilyIndices_destroy indices)
	       (return res))
	     #+nil (return (indices.isComplete))))
	 (defun checkDeviceExtensionSupport (device)
	   (declare (values bool)
		    (type VkPhysicalDevice device)
		    )
	   (let ((extensionCount 0))
	     (declare (type uint32_t extensionCount))
	     (vkEnumerateDeviceExtensionProperties
	      device NULL &extensionCount NULL)
	     (let ((availableExtensions[extensionCount]))
	       (declare (type
			 VkExtensionProperties
			 availableExtensions[extensionCount]))
	       (vkEnumerateDeviceExtensionProperties
		device
		NULL
		&extensionCount
		availableExtensions)
	       (foreach (required ,(g `_deviceExtensions))
			(let ((found false))
			  (declare (type bool found))
			  ,(vkprint "check for extension" `(required))
			  (foreach (extension availableExtensions)
					;,(vkprint "check for extension" `(required extension.extensionName))
				   (when (== 0
					     (strcmp extension.extensionName
						     required))
				     (setf found true)
				     ,(vkprint "check for extension" `(found))
				     break))
			  (unless found
			    ,(vkprint "not all of the required extensions were found" `(required found))
			    (return false))))
	       (return true))))
	 (defun getMaxUsableSampleCount ()
	   (declare (values VkSampleCountFlagBits))
	   (let ((physicalDeviceProperties))
	     (declare (type
		       VkPhysicalDeviceProperties
		       physicalDeviceProperties))
	     (vkGetPhysicalDeviceProperties
	      ,(g `_physicalDevice)
	      &physicalDeviceProperties)
	     (let ((count
		    (min
		     physicalDeviceProperties.limits.framebufferColorSampleCounts
		     physicalDeviceProperties.limits.framebufferDepthSampleCounts)))
	       (declare (type VkSampleCountFlags counts
			      ))
	       ,(vkprint "min" `(count
				 physicalDeviceProperties.limits.framebufferColorSampleCounts
				 physicalDeviceProperties.limits.framebufferDepthSampleCounts))
	       ,@(loop for e in `(64 32 16 8 4 2) collect
		      `(when (logand count
				     ,(format nil "VK_SAMPLE_COUNT_~a_BIT" e))
			 (return ,(format nil "VK_SAMPLE_COUNT_~a_BIT" e))))
	       (return VK_SAMPLE_COUNT_1_BIT))))
	 (defun pickPhysicalDevice ()
	   "// initialize member _physicalDevice" 
	   (let ((deviceCount 0))
	     (declare (type uint32_t deviceCount))
	     (vkEnumeratePhysicalDevices ,(g `_instance) &deviceCount NULL)
	     (when (== 0 deviceCount)
	       "// throw"
	       ,(vkprint "failed to find gpu with vulkan support.")
	       #+nil (throw ("std::runtime_error"
			     )))
	     (let ((devices[deviceCount]))
	       (declare (type VkPhysicalDevice
			      devices[deviceCount]))
	       (vkEnumeratePhysicalDevices ,(g `_instance) &deviceCount
					   devices)
	       (foreach (device devices)
			(when (isDeviceSuitable device)
			  (setf ,(g `_physicalDevice) device
				,(g `_msaaSamples) (getMaxUsableSampleCount))
			  break))
	       (when (== VK_NULL_HANDLE
			 ,(g `_physicalDevice))
		 "// throw"
		 ,(vkprint "failed to find a suitable gpu." )
		 #+nil(throw ("std::runtime_error"
			      (string )))))))
	 )))
 
  (define-module
      `(logical_device
	()
	(do0
	 (defun cleanupLogicalDevice ()
	   (vkDestroyDevice ,(g `_device) NULL)
	   )
	 (defun createLogicalDevice ()
	   "// initialize members _device and _graphicsQueue"
	   (let ((indices (findQueueFamilies ,(g `_physicalDevice)))
		 (queuePriority 1s0))
	     (declare (type float queuePriority))
	     (let ((allQueueFamilies[]
		    (curly
		     indices.graphicsFamily
		     indices.presentFamily))
		   (qNumber (length allQueueFamilies))
		   (qSeen[qNumber])
		   (qSeenCount 0))
	       (declare (type int allQueueFamilies[])
			(type "const int" qNumber)
			(type uint32_t qSeen[qNumber]))
	       (foreach (q allQueueFamilies)
			,(vkprint "check if queue is valid and was seen before" `(q))
			(unless (== -1 q)
			  (let ((n qSeenCount))
			    (if (== n 0)
				(do0
				 ,(vkprint "first entry" `(n))
				 (setf (aref qSeen 0) q)
				 (incf qSeenCount))
				(do0
				 (dotimes (i n)
				   ,(vkprint "loop through all queue indeces that have been seen before" `((aref qSeen i) i n))
				   (if (== q (aref qSeen i))
				       (do0 ;; we saw this before
					,(vkprint "seen before" `(q qSeenCount n))
					break)
				       (do0
					,(vkprint "not seen before" `(q qSeenCount n))
					(setf (aref qSeen (+ i 1)) q)
					(incf qSeenCount))))))))
			)
	       ,(vkprint "seen" `(qSeenCount) )
	       (let ((queueCreateInfos[qSeenCount])
		     (uniqueQueueFamilies[qSeenCount])
		     (info_count 0))
		 (declare (type VkDeviceQueueCreateInfo
				queueCreateInfos[qSeenCount])
			  (type int uniqueQueueFamilies[qSeenCount]))
		 (dotimes (i qSeenCount)
		   (let ((q (aref qSeen i)))
		     ,(vkprint "copy qSeen into uniquQueueFamilies" `(i q qSeenCount))
		     (setf (aref uniqueQueueFamilies i)
			   q)))
		 (foreach (queueFamily uniqueQueueFamilies)
			  ,(vkprint "create unique queue" `(queueFamily info_count))
			  ,(vk `(VkDeviceQueueCreateInfo
				 queueCreateInfo
				 :sType VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO
				 :queueFamilyIndex queueFamily
				 :queueCount 1
				 :pQueuePriorities &queuePriority))
			  (setf (aref queueCreateInfos info_count) queueCreateInfo)
			  (incf info_count)
			  (progn
			    ,(vkprint "created unique queue" `(queueFamily info_count queueCreateInfo))))))
	     (let ((deviceFeatures (curly)))
	       (declare (type VkPhysicalDeviceFeatures deviceFeatures))
	       (setf deviceFeatures.samplerAnisotropy VK_TRUE)
	       ,(vkcall
		 `(create
		   device
		   (:pQueueCreateInfos queueCreateInfos
				       :queueCreateInfoCount (length queueCreateInfos)
				       :pEnabledFeatures &deviceFeatures
				       :enabledExtensionCount 
				       (length ,(g `_deviceExtensions))
				       :ppEnabledExtensionNames 
				       ,(g `_deviceExtensions)
				       :enabledLayerCount
				       #-nolog (length ,(g `_validationLayers))
				       #+nolog 0
				       :ppEnabledLayerNames
				       #+nolog NULL
				       #-nolog ,(g `_validationLayers))
		   (,(g `_physicalDevice)
		     &info
		     NULL
		     (ref ,(g `_device)))
		   ,(g `_physicalDevice))
		 :throw t)
	       (progn
		 ,(vkprint "create graphics queue" `(indices.graphicsFamily))
		 (vkGetDeviceQueue ,(g `_device) indices.graphicsFamily
				   0 (ref ,(g `_graphicsQueue))))
	       (progn
		 ,(vkprint "create present queue" `(indices.presentFamily))
		 (vkGetDeviceQueue ,(g `_device) indices.presentFamily
				   0 (ref ,(g `_presentQueue))))
	       ))
					;(QueueFamilyIndices_destroy indices)
	   )
	 
	 )))

  (define-module
      `(swap_chain
	()
	(do0
	 (include <stdlib.h>
		  <assert.h>)
	 
	 (do0
	  (defun chooseSwapSurfaceFormat (availableFormats n)
	    (declare (values VkSurfaceFormatKHR)
		     (type "const VkSurfaceFormatKHR*"
			   availableFormats)
		     (type int n))
	    (dotimes (i n)	    ;foreach (format availableFormats)
	      (let ((format (aref availableFormats i)))
		(when (and (== VK_FORMAT_B8G8R8A8_UNORM
			       format.format)
			   (== VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
			       format.colorSpace))
		  (return format))))
	    (return (aref availableFormats 0)))
	  (defun chooseSwapPresentMode (modes n)
	    (declare (values VkPresentModeKHR)
		     (type "const VkPresentModeKHR*"
			   modes)
		     (type int n))
	    "// prefer triple buffer (if available)"
	    (dotimes (i n)		;foreach (mode modes)
	      (let ((mode (aref modes i)))
		(when (== VK_PRESENT_MODE_MAILBOX_KHR mode)
		  (return mode))))
	    (return VK_PRESENT_MODE_FIFO_KHR))
	  (defun chooseSwapExtent (capabilities)
	    (declare (values VkExtent2D)
		     (type "const VkSurfaceCapabilitiesKHR*"
			   capabilities))
	    (if (!= UINT32_MAX capabilities->currentExtent.width)
		(do0
		 (return capabilities->currentExtent))
		(do0
		 (let ((width 0)
		       (height 0)
		       )
		   (declare (type int width height))
		   (glfwGetFramebufferSize ,(g `_window) &width &height)
		   (let ((actualExtent (curly width
					      height))
			 )
		     (declare (type VkExtent2D actualExtent))

		     ,@(loop for e in `(width height) collect
			    `(setf (dot actualExtent ,e)
				   (max (dot capabilities->minImageExtent ,e)
					(min
					 (dot capabilities->maxImageExtent ,e)
					 (dot actualExtent ,e)))))
			
		     (return actualExtent)))))))
	 #+nil(defun cleanupSwapChain ()
					;(free ,(g `_swapChainImages))
	   )
	 (defun createSwapChain ()
	   (declare (values void))
	   
	   (let ((swapChainSupport
		  (querySwapChainSupport ,(g `_physicalDevice)))
		 (surfaceFormat
		  (chooseSwapSurfaceFormat
		   swapChainSupport.formats
		   swapChainSupport.formatsCount
		   ))
		 (presentMode
		  (chooseSwapPresentMode
		   swapChainSupport.presentModes
		   swapChainSupport.presentModesCount))
		 (extent
		  (chooseSwapExtent
		   &swapChainSupport.capabilities
		   ))
		 (imageCount_
		  (+ swapChainSupport.capabilities.minImageCount 1))
		 (imageCount (max imageCount_
				  _N_IMAGES))
		 (indices (findQueueFamilies ,(g `_physicalDevice)))
		 (queueFamilyIndices[] (curly
					indices.graphicsFamily
					indices.presentFamily))
		 ;; best performance mode:
		 (imageSharingMode VK_SHARING_MODE_EXCLUSIVE)
		 (queueFamilyIndexCount 0)
		 (pQueueFamilyIndices NULL))
	     (declare (type "__typeof__(indices.graphicsFamily)" queueFamilyIndices[]))
	     
	     ,(vkprint "create swap chain" `(imageCount_ imageCount
							 swapChainSupport.capabilities.minImageCount
							 swapChainSupport.capabilities.maxImageCount))
	     (unless (== 0 swapChainSupport.capabilities.maxImageCount)
	       (assert (<= imageCount swapChainSupport.capabilities.maxImageCount)))
	     (unless (== indices.presentFamily
			 indices.graphicsFamily)
	       "// this could be improved with ownership stuff"
	       (setf imageSharingMode VK_SHARING_MODE_CONCURRENT
		     queueFamilyIndexCount 2
		     pQueueFamilyIndices pQueueFamilyIndices))
	     (when (and (< 0 swapChainSupport.capabilities.maxImageCount)
			(< swapChainSupport.capabilities.maxImageCount
			   imageCount))
	       (setf imageCount swapChainSupport.capabilities.maxImageCount))
	     ,(vkcall
	       `(create
		 swapchain
		 (:surface ,(g `_surface)
			   :minImageCount imageCount
			   :imageFormat surfaceFormat.format
			   :imageColorSpace surfaceFormat.colorSpace
			   :imageExtent extent
			   :imageArrayLayers 1
			   :imageUsage VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
			   ;; we could use VK_IMAGE_USAGE_TRANSFER_DST_BIT
			   ;; if we want to enable post processing
			   :imageSharingMode imageSharingMode
			   :queueFamilyIndexCount queueFamilyIndexCount
			   :pQueueFamilyIndices pQueueFamilyIndices
			   :preTransform swapChainSupport.capabilities.currentTransform
			   ;; ignore alpha
			   :compositeAlpha VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
			   :presentMode presentMode
			   ;; turning on clipping can help performance
			   ;; but reading values back may not work
			   :clipped VK_TRUE
			   ;; resizing the window might need a new swap
			   ;; chain, complex topic
			   :oldSwapchain VK_NULL_HANDLE
			   )
		 (,(g `_device)
		   &info
		   NULL
		   (ref ,(g `_swapChain))
		   )
		 ,(g `_swapChain))
	       :throw t
	       :khr "KHR")
	     (cleanupSwapChainSupport &swapChainSupport)
	     (do0
	      "// now get the images, note will be destroyed with the swap chain"
	      (vkGetSwapchainImagesKHR ,(g `_device)
				       ,(g `_swapChain)
				       &imageCount
				       NULL)
	      #+nil (setf ,(g `_swapChainImages) (malloc (* (sizeof (deref ,(g `_swapChainImages)))
							    imageCount)))
	      ,(vkprint "create swapChainImages" `(imageCount _N_IMAGES))
	      (vkGetSwapchainImagesKHR ,(g `_device)
				       ,(g `_swapChain)
				       &imageCount
				       ,(g `_swapChainImages))
	      (setf ,(g `_swapChainImageFormat) surfaceFormat.format
		    ,(g `_swapChainExtent) extent)))
					;(QueueFamilyIndices_destroy indices)
	   )
	 )))

  (define-module
      `(image_view
	()
	(do0
	 (defun cleanupImageView ()
					
	   )
	 (defun createImageView (image format aspectFlags mipLevels)
	   (declare (values VkImageView)
		    (type VkImage image)
		    (type VkFormat format)
		    (type VkImageAspectFlags aspectFlags)
		    (type uint32_t mipLevels))
	   (let ((imageView))
	     (declare (type VkImageView imageView))
	     ,(vkcall
	       `(create
		 image-view
		 (:image
		  image
		  :viewType VK_IMAGE_VIEW_TYPE_2D
		  :format format
		  ;; color targets without mipmapping or
		  ;; multi layer (stereo)
		  :subresourceRange.aspectMask aspectFlags
		  :subresourceRange.baseMipLevel 0
		  :subresourceRange.levelCount mipLevels
		  :subresourceRange.baseArrayLayer 0
		  :subresourceRange.layerCount 1
		  )
		 (,(g `_device)
		   &info
		   NULL
		   &imageView)
		 imageView)
	       :throw t)
	     (return imageView)))
	 (defun createImageViews ()
	   (declare (values void))
	   #+nil (_swapChainImageViews.resize
		  (_swapChainImages.size))
	   (dotimes (i (length ,(g `_swapChainImages)))
	     ,(vkprint "createImageView" `(i (length ,(g `_swapChainImages))))
	     (setf
	      (aref ,(g `_swapChainImageViews) i)
	      (createImageView
	       (aref ,(g `_swapChainImages) i)
	       ,(g `_swapChainImageFormat)
	       VK_IMAGE_ASPECT_COLOR_BIT
	       1 ;; mipLevels
	       ))))
	 )))
  
  (define-module
      `(render_pass
	()
	(do0
	 (defun cleanupRenderPass ()
	   )
	 (defun findSupportedFormat (candidates n
				     tiling
				     features)
	   (declare (values VkFormat)
		    (type VkFormat* candidates)
		    (type int n)
		    (type VkImageTiling tiling)
		    (type VkFormatFeatureFlags features))
	   (dotimes (i n)		; foreach (format candidates)
	     (let ((format (aref candidates i))
		   (props))
	       (declare (type VkFormatProperties props))
	       (vkGetPhysicalDeviceFormatProperties ,(g `_physicalDevice)
						    format &props)
	       (when (and
		      (== VK_IMAGE_TILING_LINEAR tiling)
		      (== features
			  (logand
			   features
			   props.linearTilingFeatures)))
		 (return format))
	       (when (and
		      (== VK_IMAGE_TILING_OPTIMAL tiling)
		      (== features
			  (logand
			   features
			   props.optimalTilingFeatures)))
		 (return format))))
			  
	   ,(vkprint "failed to find supported format!")
	   )
	 (defun findDepthFormat ()
	   (declare (values VkFormat))
	   (let ((candidates[] (curly
				VK_FORMAT_D32_SFLOAT
				VK_FORMAT_D32_SFLOAT_S8_UINT
				VK_FORMAT_D24_UNORM_S8_UINT)))
	     (declare (type "VkFormat" candidates[]
			    ))
	     (return (findSupportedFormat
		      candidates
		      (length candidates)
		      VK_IMAGE_TILING_OPTIMAL
		      VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))))
	 (defun createRenderPass ()
	   ,(vk
	     `(VkAttachmentDescription
	       colorAttachment
	       :format ,(g `_swapChainImageFormat)
	       :samples ,(g `_msaaSamples)
	       :loadOp VK_ATTACHMENT_LOAD_OP_CLEAR
	       :storeOp VK_ATTACHMENT_STORE_OP_STORE
	       :stencilLoadOp VK_ATTACHMENT_LOAD_OP_DONT_CARE
	       :stencilStoreOp VK_ATTACHMENT_STORE_OP_DONT_CARE
	       :initialLayout VK_IMAGE_LAYOUT_UNDEFINED
	       ;; image to be presented in swap chain
	       :finalLayout VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL))
	   ,(vk
	     `(VkAttachmentDescription
	       depthAttachment
	       :format (findDepthFormat)
	       :samples ,(g `_msaaSamples)
	       :loadOp VK_ATTACHMENT_LOAD_OP_CLEAR
	       :storeOp VK_ATTACHMENT_STORE_OP_DONT_CARE
	       :stencilLoadOp VK_ATTACHMENT_LOAD_OP_DONT_CARE
	       :stencilStoreOp VK_ATTACHMENT_STORE_OP_DONT_CARE
	       :initialLayout VK_IMAGE_LAYOUT_UNDEFINED
	       ;; image to be presented in swap chain
	       :finalLayout VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL))
	   ,(vk
	     `(VkAttachmentDescription
	       colorAttachmentResolve
	       :format ,(g `_swapChainImageFormat)
	       :samples VK_SAMPLE_COUNT_1_BIT
	       :loadOp VK_ATTACHMENT_LOAD_OP_DONT_CARE
	       :storeOp VK_ATTACHMENT_STORE_OP_STORE
	       :stencilLoadOp VK_ATTACHMENT_LOAD_OP_DONT_CARE
	       :stencilStoreOp VK_ATTACHMENT_STORE_OP_DONT_CARE
	       :initialLayout VK_IMAGE_LAYOUT_UNDEFINED
	       ;; image to be presented in swap chain
	       :finalLayout VK_IMAGE_LAYOUT_PRESENT_SRC_KHR))
	   ,(vk
	     `(VkAttachmentReference
	       colorAttachmentRef
	       ;; we only have one attachment description
	       :attachment 0
	       ;; choose best layout for use case color buffer
	       :layout VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL))
	   ,(vk
	     `(VkAttachmentReference
	       depthAttachmentRef
	       :attachment 1
	       :layout VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL))

	   ,(vk
	     `(VkAttachmentReference
	       colorAttachmentResolveRef
	       :attachment 2
	       :layout VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL))
	   ,(vk
	     `(VkSubpassDescription
	       subpass
	       :pipelineBindPoint VK_PIPELINE_BIND_POINT_GRAPHICS
	       :colorAttachmentCount 1
	       ;; frag shader references this as outColor
	       :pColorAttachments &colorAttachmentRef
	       :pDepthStencilAttachment &depthAttachmentRef
	       :pResolveAttachments &colorAttachmentResolveRef))
	   ,(vk
	     `(VkSubpassDependency
	       dependency
	       :srcSubpass VK_SUBPASS_EXTERNAL
	       :dstSubpass 0
	       :srcStageMask VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
	       :srcAccessMask 0
	       :dstStageMask VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
	       :dstAccessMask (logior
			       VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
			       VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT)))
	   (let ((attachments[] (curly colorAttachment
				       depthAttachment
				       colorAttachmentResolve)))
	     (declare (type VkAttachmentDescription attachments[]))
	     
	     ,(vkcall
	       `(create
		 render-pass
		 (:attachmentCount 
		  (length attachments)
		  :pAttachments attachments
		  :subpassCount 1
		  :pSubpasses &subpass
		  ;; wait with writing of the color attachments
		  :dependencyCount 1
		  :pDependencies &dependency)
		 (,(g `_device)
		   &info
		   NULL
		   (ref ,(g `_renderPass)))
		 ,(g `_renderPass))
	       :throw t)))
	 )))
  (define-module
      `(descriptor_set_layout
	()
	(do0
	 (defun createDescriptorSetLayout ()
	  
	   ,(vk
	     `(VkDescriptorSetLayoutBinding
	       samplerLayoutBinding
	       :binding 1
	       :descriptorCount 1
	       :descriptorType VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
	       :pImmutableSamplers NULL
	       :stageFlags VK_SHADER_STAGE_FRAGMENT_BIT
	       ))
	   ,(vk
	     `(VkDescriptorSetLayoutBinding
	       uboLayoutBinding
	       :binding 0
	       :descriptorType VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
	       :descriptorCount 1
	       :stageFlags VK_SHADER_STAGE_VERTEX_BIT
	       :pImmutableSamplers NULL))

	   (let ((bindings[] (curly
			      uboLayoutBinding
			      samplerLayoutBinding)))
	     (declare (type
		       VkDescriptorSetLayoutBinding
		       bindings[]))
	     ,(vkcall
	       `(create
		 descriptor-set-layout
		 (:bindingCount (length bindings)
				:pBindings bindings)
		 ( ,(g `_device) &info NULL (ref ,(g `_descriptorSetLayout)))
		 ,(g `_descriptorSetLayout)
		 )
	       :throw t))
	   
	   ))))


  (define-module
      `(graphics_pipeline
	()
	(do0
	 (include <stdlib.h>
		  <string.h>)
	 ,(emit-utils :code
		      `(do0
			
			(defstruct0 Vertex
			    (pos vec3)
			  (color vec3)
			  (texCoord vec2)
					;(_binding VkVertexInputBindingDescription)
					;(_bindingp uint8_t)
			  )))
	 #+nil (do0
		"bool Vertex::operator==(const Vertex& other) const"
		(progn		   ;defun "Vertex::operator==" (other)
		  #+nil(declare (type "const Vertex&" other)
				(values bool))
		  (return (and
			   (== pos other.pos)
			   (== color other.color)
			   (== texCoord other.texCoord))))
		(do0
		 "template<> struct std::hash<Vertex>"
		 (progn
		   "size_t operator()(Vertex const& vertex) const"
		   (progn ;;defun "operator()" (vertex)
		     #+nil (declare (values size_t)
				    (type "Vertex const&" vertex))
		     (return (logxor
			      ("std::hash<glm::vec3>()"
			       vertex.pos)
			      (>> (<< ("std::hash<glm::vec3>()"
				       vertex.color)
				      1)
				  1)
			      (<< ("std::hash<glm::vec2>()"
				   vertex.texCoord)
				  1)))
		     ))))
	 (defun "Vertex_getBindingDescription" ( ;vertex
						)
	   (declare (values "VkVertexInputBindingDescription")
					;(type Vertex* vertex)
		    )
	   (do0				;if vertex->_bindingp
	    #+nil(do0
		  (return vertex->_binding))
	    (do0
	     ,(vk
	       `(VkVertexInputBindingDescription
		 bindingDescription
		 :binding 0
		 :stride (sizeof Vertex)
		 ;; move to next data after each vertex
		 :inputRate VK_VERTEX_INPUT_RATE_VERTEX))
					;(setf vertex->_binding bindingDescription)
	     (return bindingDescription))))

	 (do0
	  ,(emit-utils :code
		       `(do0
	    
			 (defstruct0 VertexInputAttributeDescription3
			     (data[3] VkVertexInputAttributeDescription)
			   )))
	  (defun "Vertex_getAttributeDescriptions" ()
	    (declare (values "VertexInputAttributeDescription3"))
	    (let ((attributeDescriptions (curly)))
	      (declare (type "VertexInputAttributeDescription3"
			     attributeDescriptions))
	      ,(set-members `((aref attributeDescriptions.data 0)
			      :binding 0
			      :location 0
			      :format VK_FORMAT_R32G32B32_SFLOAT
			      :offset (offsetof Vertex pos)))
	      ,(set-members `((aref attributeDescriptions.data 1)
			      :binding 0
			      :location 1
			      :format VK_FORMAT_R32G32B32_SFLOAT
			      :offset (offsetof Vertex color)))
	      ,(set-members `((aref attributeDescriptions.data 2)
			      :binding 0
			      :location 2
			      :format VK_FORMAT_R32G32_SFLOAT
			      :offset (offsetof Vertex texCoord)))
	      (return attributeDescriptions))))
	 ,(emit-utils :code
		      `(defstruct0 Array_u8
			   (size int)
			 (data[] uint8_t*))
		      
		      )
	 (defun makeArray_u8 (n)
	   (declare (type int n)
		    (values Array_u8*))
	   (let ((n_bytes_Array_u8 (+ (sizeof Array_u8)
				      (* n (sizeof uint8_t)))))
	     ,(vkprint "malloc" `(n_bytes_Array_u8))
	    (let ((a (malloc n_bytes_Array_u8)))
	      (declare (type Array_u8* a))
	      (setf a->size n)
	      (return a))))
	 (defun destroyArray_u8 (a)
	   (declare (type Array_u8* a))
	   (free a))
	 (defun readFile (filename)
	   (declare (type "const char*" filename)
		    (values Array_u8*	;"char*"
			    ))
	   (let ((file (fopen filename (string "r"))))
	     (unless file
	       ,(vkprint "failed to open file."))
	     (when file
	       (fseek file 0L SEEK_END)
	       (let ((filesize (ftell file))
		     (buffer (makeArray_u8 filesize) ;("(char*)" (malloc (+ 1 filesize)))
		       ))
		 (rewind file)
		 (let ((read_status (fread buffer->data 1 filesize file)))
					;(setf (aref buffer filesize) 0)
		   ,(vkprint "readFile" `(read_status filename filesize file)))
		 (return buffer)))))
	 
	 (defun createShaderModule (code)
	   (declare (values VkShaderModule)
		    (type "const Array_u8*" code)
		    )
	   ;;std::vector<char> fullfills alignment requirements of uint32_t
	   
	   (let ((shaderModule)
		 (codeSize code->size)
		 (pCode  ("(const uint32_t*)"
			  code->data)))
	     (declare (type VkShaderModule shaderModule))
	     ,(vkprint "createShader" `(codeSize pCode))
	     ,(vkcall
	       `(create
		 shader-module
		 (:codeSize codeSize
			    :pCode pCode)
		 (,(g `_device)
		   &info
		   NULL
		   &shaderModule)
		 shaderModule)
	       :throw t)
	     (return shaderModule)))
	 
	 (defun createGraphicsPipeline ()
	   (declare (values void))
	   (let ((fv (readFile (string "vert.spv")))
		 (vertShaderModule (createShaderModule
				    fv))
		 (ff (readFile (string "frag.spv")))
		 (fragShaderModule (createShaderModule
				    ff)))
	     (destroyArray_u8 fv)
	     (destroyArray_u8 ff)
	     
	     ,@(loop for e in `(frag vert)
		  collect
		    (vk
		     `(VkPipelineShaderStageCreateInfo
		       ,(format nil "~aShaderStageInfo" e)
		       :sType VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO
		       :stage ,(case e
				 (vert 'VK_SHADER_STAGE_VERTEX_BIT)
				 (frag 'VK_SHADER_STAGE_FRAGMENT_BIT))
		       :module ,(format nil "~aShaderModule" e)
		       ;; entrypoint
		       :pName (string "main")
		       ;; this would allow specification of constants:
		       :pSpecializationInfo NULL)))
	     (let (("shaderStages[]"
		    (curly vertShaderStageInfo
			   fragShaderStageInfo))
		   (bindingDescription ("Vertex_getBindingDescription"))
		   (attributeDescriptions ("Vertex_getAttributeDescriptions")))
	       (declare (type VkPipelineShaderStageCreateInfo
			      "shaderStages[]"))
	       ,(vk
		 `(VkPipelineVertexInputStateCreateInfo
		   vertexInputInfo
		   :sType VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO
		   :vertexBindingDescriptionCount 1
		   :pVertexBindingDescriptions &bindingDescription
		   :vertexAttributeDescriptionCount (length attributeDescriptions.data)
		   :pVertexAttributeDescriptions attributeDescriptions.data))
	       ,(vk
		 `(VkPipelineInputAssemblyStateCreateInfo
		   inputAssembly
		   :sType VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO
		   :topology VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST



		   ;VK_PRIMITIVE_TOPOLOGY_POINT_LIST
		   ;VK_PRIMITIVE_TOPOLOGY_LINE_LIST
		   ;; this would allow to break up lines
		   ;; and strips with 0xfff or 0xffffff
		   :primitiveRestartEnable VK_FALSE))
	       ,(vk
		 `(VkViewport
		   viewport
		   :x 0s0
		   :y 0s0
		   :width (* 1s0 ,(g `_swapChainExtent.width))
		   :height (* 1s0 ,(g `_swapChainExtent.height))
		   :minDepth 0s0
		   :maxDepth 1s0))
	       ,(vk
		 `(VkRect2D
		   scissor
		   :offset "(__typeof__(scissor.offset)){0,0}" ;(curly 0 0)
				   
		   :extent ,(g `_swapChainExtent)))
	       ,(vk
		 `(VkPipelineViewportStateCreateInfo
		   viewPortState
		   :sType VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO
		   :viewportCount 1
		   :pViewports &viewport
		   :scissorCount 1
		   :pScissors &scissor))
	       ,(vk
		 `(VkPipelineRasterizationStateCreateInfo
		   rasterizer
		   :sType VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO
				 
		   :depthClampEnable VK_FALSE
		   ;; _LINE could do wireframe rendering
		   :polygonMode VK_POLYGON_MODE_FILL
		   ;; thicker than 1s0 needs wideLines GPU feature
		   :lineWidth 1s0
		   :cullMode VK_CULL_MODE_BACK_BIT
		   :frontFace VK_FRONT_FACE_COUNTER_CLOCKWISE
		   :depthBiasEnable VK_FALSE
		   :depthBiasConstantFactor 0s0
		   ;; sometimes used for shadow mapping:
		   :depthBiasClamp 0s0
		   :depthBiasSlopeFactor 0s0))
	       ,(vk ;; for now disable multisampling
		 `(VkPipelineMultisampleStateCreateInfo
		   multisampling
		   :sType VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO
		   :sampleShadingEnable VK_FALSE
		   :rasterizationSamples ,(g `_msaaSamples)
		   :minSampleShading 1s0
		   :pSampleMask NULL
		   :alphaToCoverageEnable VK_FALSE
		   :alphaToOneEnable VK_FALSE))
	       ,(vk
		 `(VkPipelineDepthStencilStateCreateInfo
		   depthStencil
		   :sType VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO
		   :depthTestEnable VK_TRUE
		   ;; use this for transparent objects:
		   :depthWriteEnable VK_TRUE
		   ;; lower depth is closer
		   :depthCompareOp VK_COMPARE_OP_LESS
		   :depthBoundsTestEnable VK_FALSE
		   :minDepthBounds 0s0
		   :maxDepthBounds 1s0
		   :stencilTestEnable VK_FALSE
		   :front "(__typeof__(depthStencil.front)){}" ;(curly)
		   :back "(__typeof__(depthStencil.back)){}" ;(curly)
		   ))
	       ,(vk
		 `(VkPipelineColorBlendAttachmentState
		   colorBlendAttachment
				 
		   :colorWriteMask
		   (logior VK_COLOR_COMPONENT_R_BIT
			   VK_COLOR_COMPONENT_G_BIT
			   VK_COLOR_COMPONENT_B_BIT
			   VK_COLOR_COMPONENT_A_BIT
			   )
		   :blendEnable VK_FALSE
		   :srcColorBlendFactor VK_BLEND_FACTOR_ONE
		   :dstColorBlendFactor VK_BLEND_FACTOR_ZERO
		   :colorBlendOp VK_BLEND_OP_ADD
		   :srcAlphaBlendFactor VK_BLEND_FACTOR_ONE
		   :dstAlphaBlendFactor VK_BLEND_FACTOR_ZERO
		   :alphaBlendOp VK_BLEND_OP_ADD))
	       ,(vk
		 `(VkPipelineColorBlendStateCreateInfo
		   colorBlending
		   :sType
		   VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO
		   :logicOpEnable VK_FALSE
		   :logicOp VK_LOGIC_OP_COPY
		   :attachmentCount 1
		   :pAttachments &colorBlendAttachment
		   :blendConstants[0] 0s0
		   :blendConstants[1] 0s0
		   :blendConstants[2] 0s0
		   :blendConstants[3] 0s0))
	       #+nil (let (("dynamicStates[]"
			    (curly VK_DYNAMIC_STATE_VIEWPORT
				   VK_DYNAMIC_STATE_LINE_WIDTH)))
		       (declare (type VkDynamicState "dynamicStates[]"))
		       ;; viewport, line width and blend constants
		       ;; can be changed dynamically
				     
		       ,(vk
			 `(VkPipelineDynamicStateCreateInfo
			   dynamicState
			   :sType VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO
			   :dynamicStateCount 2
			   :pDynamicStates dynamicStates)))
			     
			     
			     
	       )
	     (do0
	      ,(vkcall
		`(create
		  pipeline-layout
		  (:setLayoutCount 1
				   :pSetLayouts (ref ,(g `_descriptorSetLayout))
				   ;; another way of passing dynamic values to shaders
				   :pushConstantRangeCount 0
				   :pPushConstantRanges NULL
				   )
		  (,(g `_device)
		    &info
		    NULL
		    (ref ,(g `_pipelineLayout)))
		  ,(g `_pipelineLayout))
		:throw t))
	     ,(vkcall
	       `(create
		 graphics-pipeline
		 (:stageCount 2
			      :pStages shaderStages
			      :pVertexInputState &vertexInputInfo
			      :pInputAssemblyState &inputAssembly
			      :pViewportState &viewPortState
			      :pRasterizationState &rasterizer
			      :pMultisampleState &multisampling
			      :pDepthStencilState &depthStencil
			      :pColorBlendState &colorBlending
			      ;; if we want to change linewidth:
			      :pDynamicState NULL
			      :layout ,(g `_pipelineLayout)
			      :renderPass ,(g `_renderPass)
			      :subpass 0
			      ;; similar pipelines can be derived
			      ;; from each other to speed up
			      ;; switching
			      :basePipelineHandle VK_NULL_HANDLE
			      :basePipelineIndex -1)
		 (,(g `_device)
		   VK_NULL_HANDLE ;; pipline cache
		   1
		   &info
		   NULL
		   (ref ,(g `_graphicsPipeline)))
		 ,(g `_graphicsPipeline))
	       :plural t
	       :throw t)			   
	     (vkDestroyShaderModule ,(g `_device)
				    fragShaderModule
				    NULL)
	     (vkDestroyShaderModule ,(g `_device)
				    vertShaderModule
				    NULL))))))
  (define-module
      `(command_pool
	()
	(do0
	 
	 
	 (defun createCommandPool ()
	   (declare (values void))
	   (let ((queueFamilyIndices (findQueueFamilies
				      ,(g `_physicalDevice))))
	     ,(vkcall
	       `(create
		 command-pool
		 (;; cmds for drawing go to graphics queue
		  :queueFamilyIndex queueFamilyIndices.graphicsFamily
		  :flags 0)
		 (,(g `_device)
		   &info
		   NULL
		   (ref ,(g `_commandPool)))
		 ,(g `_commandPool))
	       :throw t)
					;(QueueFamilyIndices_destroy queueFamilyIndices)
	     )))))

  (define-module
      `(color_resource
	()
	(do0

	 (defun hasStencilComponent (format)
	   (declare (values bool)
		    (type VkFormat format))
	   (return (or
		    (== VK_FORMAT_D32_SFLOAT_S8_UINT format)
		    (== VK_FORMAT_D24_UNORM_S8_UINT format))))
	 (defun beginSingleTimeCommands ()
	   (declare (values VkCommandBuffer))
	   (let ((commandBuffer))
	     (declare (type VkCommandBuffer commandBuffer))
	     ,(vkcall
	       `(allocate
		 command-buffer
		 (:level VK_COMMAND_BUFFER_LEVEL_PRIMARY
			 :commandPool ,(g `_commandPool)
			 :commandBufferCount 1
			 )
		 ( ,(g `_device) &info &commandBuffer)
		 
		 )
	       :throw nil
	       :plural t))
	   ,(vkcall
	     `(begin
	       command-buffer
	       (:flags VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
	       (commandBuffer &info)
	       commandBuffer)
	     :throw nil)
	   (return commandBuffer)
	   )
	 (defun endSingleTimeCommands (commandBuffer)
	   (declare (values void)
		    (type VkCommandBuffer commandBuffer))
	   (vkEndCommandBuffer commandBuffer)
	   ,(vk
	     `(VkSubmitInfo
	       submitInfo
	       :sType VK_STRUCTURE_TYPE_SUBMIT_INFO
	       :commandBufferCount 1
	       :pCommandBuffers &commandBuffer))
	   (vkQueueSubmit ,(g `_graphicsQueue)
			  1
			  &submitInfo
			  VK_NULL_HANDLE)
	   (vkQueueWaitIdle ,(g `_graphicsQueue))
	   (vkFreeCommandBuffers
	    ,(g `_device)
	    ,(g `_commandPool)
	    1
	    &commandBuffer)
	   ,(vkprint "endSingleTimeCommands " `(commandBuffer)))
	 (defun transitionImageLayout (image
				       format
				       oldLayout
				       newLayout
				       mipLevels)
	   (declare (values void)
		    (type VkImage image)
		    (type VkFormat format)
		    (type VkImageLayout oldLayout newLayout)
		    (type uint32_t mipLevels))
	   (let ((commandBuffer
		  (beginSingleTimeCommands)))
	     ;; use memory barriers in combination with
	     ;; exclusive sharing mode to transition
	     ;; image layout
	     ,(vk
	       `(VkImageMemoryBarrier
		 barrier
		 :sType VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER
		 ;; old could be undefined if yout dont
		 ;; care about existing contents
		 :oldLayout oldLayout
		 :newLayout newLayout
		 :srcQueueFamilyIndex
		 VK_QUEUE_FAMILY_IGNORED
		 :dstQueueFamilyIndex
		 VK_QUEUE_FAMILY_IGNORED
		 :image image
		 :subresourceRange.aspectMask
		 VK_IMAGE_ASPECT_COLOR_BIT
		 :subresourceRange.baseMipLevel 0
		 :subresourceRange.levelCount mipLevels
		 :subresourceRange.baseArrayLayer 0
		 :subresourceRange.layerCount 1
		 :srcAccessMask 0
		 :dstAccessMask 0))

	     (if (== VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
		     newLayout)
		 (do0
		  (setf barrier.subresourceRange.aspectMask VK_IMAGE_ASPECT_DEPTH_BIT)
		  (when (hasStencilComponent format)
		    (setf
		     barrier.subresourceRange.aspectMask (logior barrier.subresourceRange.aspectMask
								 VK_IMAGE_ASPECT_STENCIL_BIT))))
		 (do0
		  (setf
		   barrier.subresourceRange.aspectMask VK_IMAGE_ASPECT_COLOR_BIT)))
	     
	     (let ((srcStage )
		   (dstStage))
	       (declare (type VkPipelineStageFlags
			      srcStage
			      dstStage))
	       (if (and
		    (== VK_IMAGE_LAYOUT_UNDEFINED
			oldLayout)
		    (== VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
			newLayout))
		   (do0
		    (setf barrier.srcAccessMask 0
			  barrier.dstAccessMask VK_ACCESS_TRANSFER_WRITE_BIT
			  srcStage VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
			  dstStage VK_PIPELINE_STAGE_TRANSFER_BIT
			  ))
		   (if (and
			(== VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
			    oldLayout)
			(== VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
			    newLayout))
		       (do0
			(setf barrier.srcAccessMask VK_ACCESS_TRANSFER_WRITE_BIT
			      barrier.dstAccessMask VK_ACCESS_SHADER_READ_BIT
			      srcStage VK_PIPELINE_STAGE_TRANSFER_BIT
			      dstStage VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
			      ))
		       (if (and
			    (== VK_IMAGE_LAYOUT_UNDEFINED
				oldLayout)
			    (== VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
				newLayout))
			   (do0
			    (setf barrier.srcAccessMask 0
				  barrier.dstAccessMask (logior VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
								VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
				  srcStage VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
				  dstStage VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
				  )
			    )
			   (if (and
				(== VK_IMAGE_LAYOUT_UNDEFINED
				    oldLayout)
				(== VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
				    newLayout))
			       (do0
				(setf barrier.srcAccessMask 0
				      barrier.dstAccessMask (logior VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
								    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT)
				      srcStage VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
				      dstStage VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
				      )
				)
			       (do0
				,(vkprint "unsupported layout transition.")
				#+nil (throw
					  ("std::invalid_argument"
					   (string )))))))))
	     (vkCmdPipelineBarrier
	      commandBuffer
	      ;; https://www.khronos.org/registry/vulkan/specs/1.0/html/vkspec.html#synchronization-access-types-supported
	      srcStage	 ;; stage that should happen before barrier
	      dstStage	 ;; stage that will wait
	      0		 ;; per-region
	      0 NULL	 ;; memory barrier
	      0 NULL	 ;; buffer memory barrier
	      1 &barrier ;; image memory barrier
	      )
	     
	     (endSingleTimeCommands commandBuffer)))
	 
	 (defun findMemoryType (typeFilter properties)
	   (declare (values uint32_t)
		    (type uint32_t typeFilter)
		    (type VkMemoryPropertyFlags properties))
	   (let ((ps))
	     (declare (type VkPhysicalDeviceMemoryProperties ps))
	     (vkGetPhysicalDeviceMemoryProperties ,(g `_physicalDevice)
						  &ps)
	     (dotimes (i ps.memoryTypeCount)
	       (when (and (logand (<< 1 i)
				  typeFilter)
			  (== properties
			      (logand properties
				      (dot (aref ps.memoryTypes i)
					   propertyFlags))))
		 (return i)))
	     ,(vkprint  "failed to find suitable memory type.")))
	 ,(emit-utils :code
		      `(defstruct0 Tuple_Image_DeviceMemory
			   (image VkImage)
			 (memory VkDeviceMemory)))
	 (defun makeTuple_Image_DeviceMemory (image memory)
	   (declare (values Tuple_Image_DeviceMemory)
		    (type VkImage image)
		    (type VkDeviceMemory memory))
	   (let ((tup (curly image memory)))
	     (declare (type Tuple_Image_DeviceMemory tup))
	     (return tup)))
	 (defun createImage (width height
			     mipLevels
			     numSamples
			     format tiling
			     usage
			     properties)
	   (declare (values
		     Tuple_Image_DeviceMemory
					;"std::tuple<VkImage,VkDeviceMemory>"
		     )
		    (type uint32_t width height mipLevels)
		    (type VkSampleCountFlagBits numSamples)
		    (type VkFormat format)
		    (type VkImageTiling tiling)
		    (type VkImageUsageFlags usage)
		    (type VkMemoryPropertyFlags properties))
	   (let ((image)
		 (imageMemory)
		 )
	     (declare (type VkImage image)
		      (type VkDeviceMemory imageMemory))
	     ,(vkcall
	       `(create
		 image
		 (:imageType
		  VK_IMAGE_TYPE_2D
		  :extent.width width
		  :extent.height height
		  :extent.depth 1
		  :mipLevels mipLevels
		  :arrayLayers 1
		  :format format
		  ;; if you need direct access, use linear tiling for row major
		  :tiling tiling
		  :initialLayout VK_IMAGE_LAYOUT_UNDEFINED
		  :usage usage
			      
		  :sharingMode
		  VK_SHARING_MODE_EXCLUSIVE
		  :samples numSamples
		  :flags 0)
		 ( ,(g `_device)
		    &info
		    NULL
		    &image)
		 image)
	       :throw t))
	   (let ((memReq))
	     (declare (type VkMemoryRequirements memReq))
	     (vkGetImageMemoryRequirements
	      ,(g `_device)
	      image
	      &memReq)
	     ,(vkcall
	       `(allocate
		 memory
		 (:allocationSize
		  memReq.size
		  :memoryTypeIndex
		  (findMemoryType
		   memReq.memoryTypeBits
		   properties))
		 ( ,(g `_device)
		    &info
		    NULL
		    &imageMemory)
		 imageMemory
		 )
	       :throw t)
	     (vkBindImageMemory ,(g `_device)
				image
				imageMemory
				0)
	     (return (makeTuple_Image_DeviceMemory ;"std::make_tuple"
		      image
		      imageMemory)))
			
	   )
	 (defun createColorResources ()
	   ;; for msaa
	   (let ((colorFormat ,(g `_swapChainImageFormat))
		 (colorTuple #+nil (bracket colorImage
					    colorImageMemory)
			     (createImage
			      ,(g `_swapChainExtent.width)
			      ,(g `_swapChainExtent.height)
			      1
			      ,(g `_msaaSamples)
			      colorFormat
			      VK_IMAGE_TILING_OPTIMAL
			      (logior
			       VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT
			       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
			      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
			      )))
	     (declare (type VkFormat colorFormat))
	     (setf ,(g `_colorImage) colorTuple.image ;colorImage
		   ,(g `_colorImageMemory) colorTuple.memory ;colorImageMemory
		   )
	     (setf ,(g `_colorImageView)
		   (createImageView
		    ,(g `_colorImage)
		    colorFormat
		    VK_IMAGE_ASPECT_COLOR_BIT
		    1))
	     (transitionImageLayout
	      ,(g `_colorImage)
	      colorFormat
	      VK_IMAGE_LAYOUT_UNDEFINED
	      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
	      1))))))
  (define-module
      `(depth_resources
	()
	(do0
	 (defun createDepthResources ()
	   (declare (values void))
	   (let ((depthFormat (findDepthFormat))
		 (depthTuple 
		  (createImage ,(g `_swapChainExtent.width)
			       ,(g `_swapChainExtent.height)
			       1 ;; mipLevels
			       ,(g `_msaaSamples)
			       depthFormat
			       VK_IMAGE_TILING_OPTIMAL
			       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
			       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
			       
			       )))
	     (setf ,(g `_depthImage) depthTuple.image
		   ,(g `_depthImageMemory) depthTuple.memory
		   ,(g `_depthImageView)
		   (createImageView ,(g `_depthImage)
				    depthFormat
				    VK_IMAGE_ASPECT_DEPTH_BIT
				    1 ;; mipLevels
				    ))
	     (transitionImageLayout
	      ,(g `_depthImage)
	      depthFormat
	      VK_IMAGE_LAYOUT_UNDEFINED
	      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL 1))))))
  (define-module
      `(framebuffer
	()
	(do0
	 ,(emit-utils :code
		      `(defstruct0 Triple_FrambufferViews
			   (image VkImageView)
			 (depth VkImageView)
			 (swap VkImageView)))
	 (defun createFramebuffers ()
	   (declare (values void))
	   (let ((n (length ,(g `_swapChainImageViews))))
					;(_swapChainFramebuffers.resize n)
	     (dotimes (i n)
	       ;; color attachment differs for each
	       ;; swap chain image, but depth image can
	       ;; be reused. at any time only one
	       ;; subpass is running
	       (let ((attachments (cast Triple_FrambufferViews
					(curly ,(g `_colorImageView)
					       ,(g `_depthImageView)
					       (aref ,(g `_swapChainImageViews) i)
					       ))))
				
		 ,(vkcall
		   `(create
		     framebuffer
		     (:renderPass
		      ,(g `_renderPass)
		      :attachmentCount 3
		      :pAttachments (cast VkImageView* &attachments) ;; FIXME: perhaps use flex array
		      :width ,(g `_swapChainExtent.width)
		      :height ,(g `_swapChainExtent.height)
		      :layers 1)
		     ( ,(g `_device)
			&info
			NULL
			(ref
			 (aref ,(g `_swapChainFramebuffers) i)))
		     (aref ,(g `_swapChainFramebuffers) i))
		   :throw t))))))))
  (define-module
      `(texture_image
	()
	(do0
	 "#pragma GCC optimize (\"O3\")"
	 " "
	 
	 (include <string.h>)
	 (do0
	  "#define STB_IMAGE_IMPLEMENTATION"
	  " "
	  "#define STBI_ONLY_JPEG"
	  " "
	  "#define STBI_NO_HDR"
	  " "
	  "#define STBI_NO_LINEAR"
	  " "
	  "#define STBI_NO_FAILURE_STRINGS"
	  " "
	  (include "stb_image.h")
	  " "
	  (include <math.h>)
	  " ")
	 ,(emit-utils :code
		      `(defstruct0 Tuple_Buffer_DeviceMemory
			   (buffer VkBuffer)
			 (memory VkDeviceMemory)))
	 (defun createBuffer (size usage properties )
	   ;; https://www.fluentcpp.com/2018/06/19/3-simple-c17-features-that-will-make-your-code-simpler/
	   (declare (values Tuple_Buffer_DeviceMemory ;"std::tuple<VkBuffer,VkDeviceMemory>"
			    )
		    (type VkDeviceSize size)
		    (type VkBufferUsageFlags usage)
		    (type VkMemoryPropertyFlags properties)
		    )
	   (let ((buffer)
		 (bufferMemory)
		 )
	     (declare (type VkBuffer  buffer)
		      (type VkDeviceMemory bufferMemory)
		      )
	     ,(vkcall
	       `(create
		 buffer
		 (;; buffer size in bytes
		  :size size
		  :usage usage
		  ;; only graphics queue is using this buffer
		  :sharingMode VK_SHARING_MODE_EXCLUSIVE
		  ;; flags could indicate sparse memory (we
		  ;; don't use that)
		  :flags 0
		  )
		 ( ,(g `_device)
		    &info
		    NULL
		    &buffer)
		 buffer
		 )
	       :throw t)
			 
	     (let ((memReq))
	       (declare (type VkMemoryRequirements memReq))
	       (vkGetBufferMemoryRequirements ,(g `_device)
					      buffer
					      &memReq)
	       ,(vkcall
		 `(allocate
		   memory
		   (:allocationSize memReq.size
				    :memoryTypeIndex (findMemoryType
						      memReq.memoryTypeBits
						      properties
						      ))
		   ( ,(g `_device)
		      &info
		      NULL
		      &bufferMemory)
		   bufferMemory)
		 :throw t)
			  
	       (vkBindBufferMemory ,(g `_device)
				   buffer
				   bufferMemory
				   0)
	       (return (cast Tuple_Buffer_DeviceMemory
			     (curly
			      buffer
			      bufferMemory))))))
	 (defun generateMipmaps (image imageFormat
				 texWidth texHeight mipLevels)
	   (declare (values void)
		    (type VkImage image)
		    (type VkFormat imageFormat)
		    (type int32_t texHeight mipLevels texWidth))

	   ,(vkprint "generateMipmaps")
	   
	   (let ((formatProperties))
	     (declare (type VkFormatProperties formatProperties))
	     ;; check if format blit can do linear
	     (vkGetPhysicalDeviceFormatProperties
	      ,(g `_physicalDevice)
	      imageFormat
	      &formatProperties)
	     (unless
		 (logand formatProperties.optimalTilingFeatures
			 VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)
	       ,(vkprint "texture image format does not support linear blitting!")
	       #+nil (throw ("std::runtime_error"
			     (string ))))
	     )

	   (let ((commandBuffer (beginSingleTimeCommands)))
	     ,(vk
	       `(VkImageMemoryBarrier
		 barrier
		 :sType VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER
		 :image image
		 :srcQueueFamilyIndex VK_QUEUE_FAMILY_IGNORED
		 :dstQueueFamilyIndex VK_QUEUE_FAMILY_IGNORED
		 :subresourceRange.aspectMask VK_IMAGE_ASPECT_COLOR_BIT
		 :subresourceRange.baseArrayLayer 0
		 :subresourceRange.layerCount 1
		 :subresourceRange.levelCount 1
		 ))
	     (let ((mipWidth texWidth)
		   (mipHeight texHeight))
	       (for ((= "int i" 1) (< i mipLevels) (incf i))
		    (do0
		     
		     ,(set-members
		       `(barrier
			    :subresourceRange.baseMipLevel (- i 1)
			    :oldLayout VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
			    :newLayout VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
			    :srcAccessMask VK_ACCESS_TRANSFER_WRITE_BIT
			    :dstAccessMask VK_ACCESS_TRANSFER_READ_BIT))
		     ,(vkprint " vkCmdPipelineBarrier " `(i))
		     
		     (vkCmdPipelineBarrier
		      commandBuffer
		      VK_PIPELINE_STAGE_TRANSFER_BIT
		      VK_PIPELINE_STAGE_TRANSFER_BIT
		      0
		      0 NULL
		      0 NULL
		      1 &barrier))
		    (let ((dstOffsetx 1)
			  (dstOffsety 1))
		      (when (< 1 mipWidth)
			(setf dstOffsetx (/ mipWidth 2)))
		      (when (< 1 mipHeight)
			(setf dstOffsety (/ mipHeight 2)))
		      ,(vk
			`(VkImageBlit
			  blit
			  :srcOffsets[0] (cast (__typeof__ *blit.srcOffsets)
					       (curly 0 0 0))
			  :srcOffsets[1] (cast (__typeof__ *blit.srcOffsets)
					       (curly mipWidth
						      mipHeight
						      1))
			  :srcSubresource.aspectMask VK_IMAGE_ASPECT_COLOR_BIT
			  :srcSubresource.mipLevel (- i 1)
			  :srcSubresource.baseArrayLayer 0
			  :srcSubresource.layerCount 1
			  :dstOffsets[0] (cast (__typeof__ *blit.dstOffsets)
					       (curly 0 0 0))
			  :dstOffsets[1] (cast (__typeof__ *blit.dstOffsets)
					       (curly dstOffsetx
						      dstOffsety
						      1))
			  :dstSubresource.aspectMask VK_IMAGE_ASPECT_COLOR_BIT
			  :dstSubresource.mipLevel i
			  :dstSubresource.baseArrayLayer 0
			  :dstSubresource.layerCount 1
			  ))
		      ,(vkprint " vkCmdBlitImage" `(i))
		      
		      (vkCmdBlitImage
		       commandBuffer
		       image
		       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
		       image
		       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
		       1
		       &blit
		       VK_FILTER_LINEAR))

		    (do0
		     ,(set-members
		       `(barrier
			    :oldLayout VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
			    :newLayout VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
			    :srcAccessMask VK_ACCESS_TRANSFER_READ_BIT
			    :dstAccessMask VK_ACCESS_SHADER_READ_BIT))
		     ;; wait on blit command
		     ;; transition mip level i-1 to shader_ro
		     ,(vkprint " vkCmdPipelineBarrier" `(i))
		     
		     (vkCmdPipelineBarrier
		      commandBuffer
		      VK_PIPELINE_STAGE_TRANSFER_BIT
		      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
		      0
		      0 NULL
		      0 NULL
		      1 &barrier))

		    (do0
		     ;; handle non-square images
		     ;; ensure mip dimensions never become 0
		     (when (< 1 mipWidth)
		       (setf mipWidth (/ mipWidth 2)))
		     (when (< 1 mipHeight)
		       (setf mipHeight (/ mipHeight 2))))
		    
		    
		    ))

	     (do0
	      ;; transition last miplevel because the
	      ;; last mipmap was never blitted from
	      ,(set-members
		`(barrier
		     :subresourceRange.baseMipLevel
		   (- ,(g `_mipLevels) 1)
		   :oldLayout VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
		   :newLayout VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		   :srcAccessMask VK_ACCESS_TRANSFER_WRITE_BIT
		   :dstAccessMask VK_ACCESS_SHADER_READ_BIT))
	      (vkCmdPipelineBarrier
	       commandBuffer
	       VK_PIPELINE_STAGE_TRANSFER_BIT
	       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
	       0
	       0 NULL
	       0 NULL
	       1 &barrier))
	     
	     (endSingleTimeCommands commandBuffer)))
	 (defun copyBufferToImage (buffer
				   image
				   width
				   height)
	   (declare (values void)
		    (type VkBuffer buffer)
		    (type VkImage image)
		    (type uint32_t width height))
	   ,(with-single-time-commands
		`((commandBuffer)
		  ,(vk
		    `(VkBufferImageCopy
		      region
		      :bufferOffset 0
		      :bufferRowLength 0
		      :bufferImageHeight 0
		      :imageSubresource.aspectMask
		      VK_IMAGE_ASPECT_COLOR_BIT
		      :imageSubresource.mipLevel 0
		      :imageSubresource.baseArrayLayer 0
		      :imageSubresource.layerCount 1
		      :imageOffset (cast (__typeof__ region.imageOffset)
					 (curly 0 0 0))
		      :imageExtent (cast (__typeof__ region.imageExtent) (curly width height 1))))
		  (vkCmdCopyBufferToImage
		   commandBuffer
		   buffer
		   image
		   ;; assuming the layout has already
		   ;; been transitioned into a format
		   ;; that is optimal for copying pixels
		   ;; to
		   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
		   1
		   &region))))
	 (defun createTextureImage ()
	   (declare (values void))
	   "// uses command buffers "
	   ,(vkprint "start loading texture")
	   (let ((texWidth 0)
		 (texHeight 0)
		 (texChannels 0)
		 (texFilename (string "chalet.jpg"))
		 (pixels
		  (stbi_load
		   texFilename
		   &texWidth
		   &texHeight
		   &texChannels
		   STBI_rgb_alpha))
		 (imageSize (* texWidth texHeight 4)))
	     (declare (type int
			    texWidth
			    texHeight
			    texChannels
			    )
		      (type VkDeviceSize
			    imageSize))
	     
	     (unless pixels
	       ,(vkprint "failed to load texture image." `(texFilename))
	       #+nil (throw ("std::runtime_error"
			     (string ))))
	     (setf ,(g `_mipLevels)
		   (cast uint32_t
			 (+ 1
			    (floor
			     (log2
			      (max
			       texWidth
			       texHeight))))))
	     ,(vkprint "loaded texture" `(texWidth texHeight texChannels texFilename ,(g `_mipLevels)))
	     (do0 ;; print comment with example mip levels
	      ,(format nil "// ~8a ~8a" "width" "mipLevels")
	      ,@(loop for i in `(2 4 16 32 128 255 256 257 512 1024) collect
		     (format nil "// ~8a ~8a"
			     i (+ 1 (floor (log i 2))))))
	     
	     (let ((stagingBufferTuple #+nil (bracket stagingBuffer
						      stagingBufferMemory)
				       (createBuffer
					imageSize
					VK_BUFFER_USAGE_TRANSFER_SRC_BIT
					(logior
					 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
					 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
					 )))
		   (data NULL))
	       (declare (type void* data))
	       ,(vkprint "map staging")
	       (vkMapMemory ,(g `_device)
			    stagingBufferTuple.memory
			    0
			    imageSize
			    0
			    &data)
	       ,(vkprint "copy pixels")
	       (memcpy data pixels
		       imageSize)
	       ,(vkprint "unmap staging")
	       (vkUnmapMemory ,(g `_device)
			      stagingBufferTuple.memory)
	       (stbi_image_free pixels))
	     ,(vkprint "create image")
	     (let ((imageTuple
		    (createImage
		     texWidth
		     texHeight
		     ,(g `_mipLevels)
		     VK_SAMPLE_COUNT_1_BIT
		     VK_FORMAT_R8G8B8A8_UNORM
		     VK_IMAGE_TILING_OPTIMAL
		     (logior
		      VK_IMAGE_USAGE_TRANSFER_DST_BIT
		      VK_IMAGE_USAGE_TRANSFER_SRC_BIT ;; for mipLevel computation
		      VK_IMAGE_USAGE_SAMPLED_BIT)
		     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
		     )))
	       (setf ,(g `_textureImage) imageTuple.image
		     ,(g `_textureImageMemory) imageTuple.memory)
	       ,(vkprint "transition image layout")
	       (transitionImageLayout
		,(g `_textureImage)
		VK_FORMAT_R8G8B8A8_UNORM
		VK_IMAGE_LAYOUT_UNDEFINED
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
		,(g `_mipLevels))
	       (copyBufferToImage
		stagingBufferTuple.buffer
		,(g `_textureImage)
		texWidth
		texHeight)

	       
	       #+nil(transitionImageLayout
		     ,(g `_textureImage)
		     VK_FORMAT_R8G8B8A8_UNORM
		     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
		     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		     )
	       ;; will be transitioned to
	       ;; READ_ONLY_OPTIMAL while generating
	       ;; mipmaps
	       ,(vkprint "destroy staging")
	       (do0
		(vkDestroyBuffer
		 ,(g `_device) stagingBufferTuple.buffer NULL)
		(vkFreeMemory ,(g `_device)
			      stagingBufferTuple.memory
			      NULL))
	       ,(vkprint "start mip maps")
	       (generateMipmaps ,(g `_textureImage)
				VK_FORMAT_R8G8B8A8_UNORM
				texWidth
				texHeight
				,(g `_mipLevels))
	       ,(vkprint "finished mip maps")
	       ))))))
  (define-module
      `(texture_image_view
	()
	(do0
	 (defun createTextureImageView ()
	   
	   (setf
	    ,(g `_textureImageView)
	    (createImageView ,(g `_textureImage)
			     VK_FORMAT_R8G8B8A8_UNORM
			     VK_IMAGE_ASPECT_COLOR_BIT
			     ,(g `_mipLevels)))))))
  (define-module
      `(texture_sampler
	()
	(do0
	 (defun createTextureSampler ()
	   
	   ,(vkcall
	     `(create
	       sampler
	       (:magFilter
		VK_FILTER_LINEAR
		:minFilter VK_FILTER_LINEAR
		:addressModeU VK_SAMPLER_ADDRESS_MODE_REPEAT
		:addressModeV VK_SAMPLER_ADDRESS_MODE_REPEAT
		:addressModeW VK_SAMPLER_ADDRESS_MODE_REPEAT
		:anisotropyEnable VK_TRUE
		:maxAnisotropy 16
		:borderColor VK_BORDER_COLOR_INT_OPAQUE_BLACK
		:unnormalizedCoordinates VK_FALSE
		:compareEnable VK_FALSE
		:compareOp VK_COMPARE_OP_ALWAYS
		:mipmapMode VK_SAMPLER_MIPMAP_MODE_LINEAR
		:mipLodBias 0s0
		:minLod
					;(static_cast<float> (*  _mipLevels 2))
		0s0
		:maxLod (cast float ,(g `_mipLevels)))
	       (,(g `_device)
		 &info
		 NULL
		 (ref ,(g `_textureSampler)))
	       ,(g `_textureSampler)
	       )
	     :throw t)))))
  (define-module
      `(load_object
	()
	(do0
	 "#pragma GCC optimize (\"O3\")"
	 " "
	 
	 "#define TINYOBJ_LOADER_C_IMPLEMENTATION"
	 (include "tinyobj_loader_c.h")
	 (include <fcntl.h>
		  <sys/mman.h>
		  <sys/stat.h>
		  <sys/types.h>
		  <unistd.h>
		  )

	 
	 ,(emit-utils
	   :code
	   `(defstruct0 mmapPair
		(n int)
	      (data char*)
	      ))
	 (defun munmapFile (pair)
	   (declare (type mmapPair pair)
		    )
	   (munmap pair.data pair.n))
	 (defun mmapFile (filename)
	   (declare (type char* filename)
		    (values mmapPair))
	   (let ((f (fopen filename (string "r"))))
	     (unless f
	       ,(vkprint "can't open file" `(filename)))
	     (fseek f 0 SEEK_END)
	     (let ((filesize (ftell f)))
	       (fclose f)))
	   (let ((fd (open filename O_RDONLY))
		 (sb))
	     (declare (type "struct stat" sb))
	     (when (== -1 fd)
	       ,(vkprint "can't open file for mmap" `(filename)))
	     (when (== -1 (fstat fd &sb))
	       ,(vkprint "can't fstat file" `(filename)))
	     (unless (S_ISREG sb.st_mode)
	       ,(vkprint "not a file" `(filename sb.st_mode)))
	     (let ((p (mmap 0 filesize
			    PROT_READ
			    MAP_SHARED
			    fd
			    0)))
	       (if (== MAP_FAILED p)
		   ,(vkprint "mmap failed" `(filename)))
	       (if (== -1 (close fd))
		   ,(vkprint "close failed" `(filename)))
	       (let ((map (cast mmapPair
				(curly filesize
				       p))))
		 (return map)))
	     )
	   )
	 (defun cleanupModel ()
	   (free ,(g `_vertices) )
	   (free ,(g `_indices) )
	   (setf
	    ,(g `_num_vertices) 0
	    ,(g `_num_indices) 0
	    ))

	 (defun loadModel ()
	   ;; https://en.wikipedia.org/wiki/Wavefront_.obj_file the
	   ;; obj file that i use contains lists of vertex positions
	   ;; v -0.587606 0.129110 0.826194
	   ;; v -0.586717 0.122363 0.824548

	   ;; texture coordinates
	   ;; vt 0.6716 0.5600
	   ;; vt 0.6724 0.5607

	   ;; and a list of triangles with corresponding vertex index
	   ;; and texture coordinate index
	   ;; f 1/f 1/1 2/2 3/3
	   ;; f 4/4 5/5 6/6
	   (let ((map (mmapFile (string "chalet.obj")))
		  (attrib)
		  (shapes NULL)
		  (num_shapes)
		  (materials NULL)
		  (num_materials)

		 )
	     (declare (type tinyobj_attrib_t attrib)
		       (type tinyobj_shape_t* shapes)
		       (type tinyobj_material_t* materials)
		       (type size_t num_shapes
			     num_materials))
	     (tinyobj_attrib_init &attrib)
	    (let  ((res (tinyobj_parse_obj
			&attrib
			&shapes
			&num_shapes
			&materials
			&num_materials
			map.data
			map.n
			TINYOBJ_FLAG_TRIANGULATE)))
		
	      (unless (== TINYOBJ_SUCCESS res)
		,(vkprint "tinyobj failed to open" `(res)))
	      #+nil,(vkprint "tinyobj opened" `(num_shapes
						num_materials
						attrib.num_face_num_verts))
	      ;; num_shapes=                       1 (unsigned long int)
	      ;; num_materials=                    0 (unsigned long int)
	      ;; attrib.num_face_num_verts=   500000 (unsigned int)   ;; i think  this is the number of vertices belonging to each face. here each face has 3 vertices 
	      ;; attrib.num_vertices=         234246 (unsigned int)
	      ;; attrib.num_texcoords=        265645 (unsigned int)
	      ;; attrib.num_faces=           1500000 (unsigned int)
	      ;; attrib.num_normals=               0 (unsigned int)

	      ,(vkprint "model" `(num_shapes
				  num_materials
				  attrib.num_face_num_verts
				  attrib.num_vertices
				  attrib.num_texcoords
				  attrib.num_faces
				  attrib.num_normals))
	      ;; float attrib.vertices[3*num_vertices]
	      ;; float attrib.texcoords[3*num_texcoords]
	      ;; struct { int v_idx, vt_idx, vn_idx; } tinyobj_vertex_index_t
	      ;; vertex_index_t  attrib.faces[num_faces]


	      (do0
	       (setf
		  ,(g `_num_vertices) attrib.num_faces
		  )
	       (let ((n_bytes_vertices (* (sizeof (deref ,(g `_vertices)))
					  ,(g `_num_vertices))))
		 ,(vkprint "malloc" `(n_bytes_vertices))
		 (setf
		  
		  ,(g `_vertices) (malloc n_bytes_vertices)
		  )))
	      (do0
	       (setf
		,(g `_num_indices) attrib.num_faces
		)
	       (let ((n_bytes_indices (* (sizeof (deref ,(g `_indices)))
					    ,(g `_num_indices))))
		 ,(vkprint "malloc" `(n_bytes_indices))
		 (setf
		  ,(g `_indices) (malloc n_bytes_indices))))

	      (dotimes (i attrib.num_faces)
		(let (,@(loop for j below 3 collect
			     `(,(format nil "v~a" j)
				(aref attrib.vertices (+ ,j (* 3 (dot (aref (dot attrib faces) i)
								      v_idx))))))
		      ,@(loop for j below 2 collect
			     `(,(format nil "t~a" j)
				(aref attrib.texcoords (+ ,j (* 2 (dot (aref (dot attrib faces) i)
								       vt_idx))))))
			(vertex (cast Vertex
				      (curly
				       (curly v0 v1 v2)
				       (curly 1s0 1s0 1s0)
				       (curly t0 (- t1))))))
		  
		  (setf (aref ,(g `_vertices) i) vertex
			(aref ,(g `_indices) i) i)))
	      
	      
	  
	      (munmapFile map))
	    (do0
	     "// cleanup"
	     (tinyobj_attrib_free &attrib)
	     (when shapes
	       (tinyobj_shapes_free shapes num_shapes))
	     (when materials
	       (tinyobj_materials_free materials num_materials)))
	    ))
	 )))
  (define-module
	    `(vertex_buffer
	      ()
	      (do0
	       (include <string.h>)
	       (defun copyBuffer (srcBuffer
					  dstBuffer
					  size)
			 (declare (values void)
				  (type VkBuffer srcBuffer dstBuffer)
				  (type VkDeviceSize size))
		       
		       
		       
			 (let ((commandBuffer (beginSingleTimeCommands)))
			   ,(vk
			     `(VkBufferCopy
			       copyRegion
			       :srcOffset 0
			       :dstOffset 0
			       :size size))
			   (vkCmdCopyBuffer commandBuffer srcBuffer
					    dstBuffer 1 &copyRegion)
			   (endSingleTimeCommands commandBuffer)))
	       (defun createVertexBuffer ()
			 (declare (values void))
			 (let ((bufferSize (* (sizeof (aref ,(g `_vertices) 0))
					      ,(g `_num_vertices)))
			       (stagingBuffer
				(createBuffer
				 bufferSize
				 VK_BUFFER_USAGE_TRANSFER_SRC_BIT
				 (logior
				  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
				  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))))
			 			
			   (let ((data))
			     (declare (type void* data))
			     (vkMapMemory ,(g `_device) stagingBuffer.memory
					  0	     ;; offset
					  bufferSize ;; size
					  0	     ;; flags
					  &data)
			     (memcpy data
				     ,(g `_vertices)
				     bufferSize)
			     ;; without coherent bit, the changed memory
			     ;; might not immediatly be visible.
			     ;; alternatively: vkFlushMappedMemoryRanges
			     ;; or vkInvalidateMappedMemoryRanges; the
			     ;; memory transfer is defined to be
			     ;; complete as of the next call to
			     ;; vkQueueSubmit
			     (vkUnmapMemory ,(g `_device) stagingBuffer.memory)))

			 (let  ((vertexBuffer
				 (createBuffer
				  bufferSize
				  (logior
				   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
				   ;; can be a data transfer destination
				   VK_BUFFER_USAGE_TRANSFER_DST_BIT)
				  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)))
			   (setf ,(g `_vertexBuffer) vertexBuffer.buffer
				 ,(g `_vertexBufferMemory) vertexBuffer.memory)
			   (copyBuffer stagingBuffer.buffer
				       ,(g `_vertexBuffer)
				       bufferSize))
		       
			 (do0
			  (vkDestroyBuffer ,(g `_device) stagingBuffer.buffer NULL)
			  (vkFreeMemory ,(g `_device) stagingBuffer.memory NULL))))))
  (define-module
	    `(index_buffer
	      ()
	      (do0
	       (include <string.h>)
	       (defun createIndexBuffer ()
			 (declare (values void))
			 (let ((bufferSize (* (sizeof (aref ,(g `_indices) 0))
					      ,(g `_num_indices)))
			       (stagingBuffer
				(createBuffer
				 bufferSize
				 VK_BUFFER_USAGE_TRANSFER_SRC_BIT
				 (logior
				  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
				  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))))
			 			
			   (let ((data))
			     (declare (type void* data))
			     (vkMapMemory ,(g `_device) stagingBuffer.memory
					  0	     ;; offset
					  bufferSize ;; size
					  0	     ;; flags
					  &data)
			     (memcpy data
				     ,(g `_indices)
				     bufferSize)
			  
			     (vkUnmapMemory ,(g `_device) stagingBuffer.memory)))

			 (let  ((indexBuffer
				 (createBuffer
				  bufferSize
				  (logior
				   VK_BUFFER_USAGE_INDEX_BUFFER_BIT
				   ;; can be a data transfer destination
				   VK_BUFFER_USAGE_TRANSFER_DST_BIT)
				  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)))
			   (setf ,(g `_indexBuffer) indexBuffer.buffer
				 ,(g `_indexBufferMemory) indexBuffer.memory)
			   (copyBuffer stagingBuffer.buffer
				       ,(g `_indexBuffer)
				       bufferSize))
		       
			 (do0
			  (vkDestroyBuffer ,(g `_device) stagingBuffer.buffer NULL)
			  (vkFreeMemory ,(g `_device) stagingBuffer.memory NULL))))))
  (define-module
	    `(uniform_buffers
	      ()
	      (do0
	       ,(emit-utils :code
		 `(do0
		   (defstruct0 UniformBufferObject
		       ;; 32 needed for avx
		       ;; "alignas(16) mat4"
		       (model "mat4")
		     (view "mat4")
		     (proj "mat4"))))
	       (defun createUniformBuffers ()
		 
		 (let ((bufferSize (sizeof UniformBufferObject))
		       (n (length ,(g `_swapChainImages))))
					;(_uniformBuffers.resize n)
					;(_uniformBuffersMemory.resize n)
		   ,(vkprint "create uniform buffers" `(bufferSize
							(length ,(g `_swapChainImages))
							(length ,(g `_uniformBuffers))
							(length ,(g `_uniformBuffersMemory))
							))
		   (dotimes (i n)
		     (let ((uniformBuffer
			    (createBuffer
			     bufferSize
			     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
			     (logior
			      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
			      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
			     )))
		       (setf (aref ,(g `_uniformBuffers) i)
			     uniformBuffer.buffer
			     (aref ,(g `_uniformBuffersMemory) i)
			     uniformBuffer.memory))))))))
 (define-module
	    `(descriptor_pool
	      ()
	      (do0
	       (defun createDescriptorPool ()
			  (declare (values void))
			  (let ((n (length ,(g `_swapChainImages)))
				)
			    ,(vk
			      `(VkDescriptorPoolSize
				uboPoolSize
				:type VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
				:descriptorCount
				n
				))
			    ,(vk
			      `(VkDescriptorPoolSize
				samplerPoolSize
				:type VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
				:descriptorCount
				n
				))
			    (let ((poolSizes[] (curly uboPoolSize
						       samplerPoolSize)))
			      (declare (type
					VkDescriptorPoolSize poolSizes[]))
			      ,(vkcall
				`(create
				  descriptor-pool
				  (:poolSizeCount (length poolSizes)
						  :pPoolSizes poolSizes
						  :maxSets n
						  ;; we could allow to free the sets
						  :flags 0)
				  (,(g `_device)
				   &info
				   NULL
				   (ref ,(g `_descriptorPool)))
				  ,(g `_descriptorPool))
				:throw t)))))))
  (define-module
	    `(descriptor_sets
	      ()
	      (do0
	       (defun createDescriptorSets ()
			  (declare (values void))
			  (let ((n (length ,(g `_swapChainImages)))
				(layouts[] (curly ,(g `_descriptorSetLayout)
						   ,(g `_descriptorSetLayout)
						   ,(g `_descriptorSetLayout)
						   ,(g `_descriptorSetLayout))))
			    (declare (type VkDescriptorSetLayout layouts[])
				     (type "const int" n))
			    ;(_descriptorSets.resize n)
			    ,(vkcall
			      `(allocate
				descriptor-set
				(:descriptorPool ,(g `_descriptorPool)
						 :descriptorSetCount n
						 :pSetLayouts layouts)
				(,(g `_device)
				 &info
				 ,(g `_descriptorSets))
					;_descriptorSets
				)
			      :throw t
			      :plural t)
			    ;; the sets will be automatically freed when
			    ;; pool is destroyed
			    (dotimes (i n)
			      ,(vk
				`(VkDescriptorBufferInfo
				  bufferInfo
				  :buffer (aref ,(g `_uniformBuffers) i)
				  :offset 0
				  :range (sizeof UniformBufferObject)))
			    
			      ,(vk
				`(VkWriteDescriptorSet
				  uboDescriptorWrite
				  :sType VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
				  :dstSet (aref ,(g `_descriptorSets) i)
				  :dstBinding 0
				  :dstArrayElement 0 ;; if multiple
				  ;; descriptors
				  ;; should be
				  ;; updated at once,
				  ;; start here
				  :descriptorType VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
				  :descriptorCount 1
				  :pBufferInfo &bufferInfo ;; buffer data
				  :pImageInfo NULL	 ;; image data
				  :pTexelBufferView NULL ;; buffer views
				  ))
			      ,(vk
				`(VkDescriptorImageInfo
				  imageInfo
				  :imageLayout VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
				  :imageView ,(g `_textureImageView)
				  :sampler ,(g `_textureSampler)))
			      ,(vk
				`(VkWriteDescriptorSet
				  samplerDescriptorWrite
				  :sType VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
				  :dstSet (aref ,(g `_descriptorSets) i)
				  :dstBinding 1
				  :dstArrayElement 0
				  :descriptorType VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
				  :descriptorCount 1
				  :pBufferInfo NULL ;; buffer data
				  :pImageInfo &imageInfo ;; image data
				  :pTexelBufferView NULL ;; buffer views
				  ))
			      (let ((descriptorWrites[]
				      (curly
				       uboDescriptorWrite
				       samplerDescriptorWrite)))
				(declare (type
					  VkWriteDescriptorSet
					  descriptorWrites[]))
				(vkUpdateDescriptorSets
				 ,(g `_device)
				 (length descriptorWrites)
				 descriptorWrites
				 0
				 NULL ;; copy descriptor sets
				 )))
			    )))))
  (define-module
	    `(command_buffers
	      ()
	      (do0
	       (defun createCommandBuffers ()
		 #+nil(_commandBuffers.resize
		  (_swapChainFramebuffers.size))
		 ,(vkcall
		   `(allocate
		     command-buffer
		     (:commandPool ,(g `_commandPool)
				   :level VK_COMMAND_BUFFER_LEVEL_PRIMARY
				   :commandBufferCount (length ,(g `_commandBuffers)))
		     (,(g `_device)
		      &info
		      ,(g `_commandBuffers))
					;_commandBuffers
		     )
		   :throw t
		   :plural t)
			 
			 
		 (dotimes (i (length ,(g `_commandBuffers)))
		   ,(vkcall
		     `(begin
		       command-buffer
		       (;; flags can select if exectution is
			;; once, inside single renderpass, or
			;; resubmittable
			:flags 0 
			:pInheritanceInfo NULL)
		       ((aref ,(g `_commandBuffers) i) &info)
		       (aref ,(g `_commandBuffers) i))
		     :throw t)
			   

		   ,(vk
		     `(VkClearValue
		       clearColor
		       :color (cast (__typeof__ clearColor.color)  (curly 0s0 0s0 0s0 1s0))))
		   ,(vk
		     `(VkClearValue
		       clearDepth
		       ;; depth buffer far=1 by default
		       :depthStencil (cast (__typeof__ clearDepth.depthStencil) (curly 1s0 0))))
		   (let ((clearValues[] (curly clearColor clearDepth)))
		     (declare (type VkClearValue clearValues[]))
			      
		     ,(vk
		       `(VkRenderPassBeginInfo
			 renderPassInfo
			 :sType VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO
			 :renderPass ,(g `_renderPass)
			 :framebuffer (aref ,(g `_swapChainFramebuffers) i)
			 :renderArea.offset (cast (__typeof__ renderPassInfo.renderArea.offset) (curly  0 0))
			 :renderArea.extent ,(g `_swapChainExtent)
			 :clearValueCount (length clearValues)
			 :pClearValues clearValues
			 )))
		   (vkCmdBeginRenderPass
		    (aref ,(g `_commandBuffers) i)
		    &renderPassInfo
		    ;; dont use secondary command buffers
		    VK_SUBPASS_CONTENTS_INLINE)
		   (vkCmdBindPipeline
		    (aref ,(g `_commandBuffers) i)
		    VK_PIPELINE_BIND_POINT_GRAPHICS ,(g `_graphicsPipeline))
		   (let ((vertexBuffers[] (curly ,(g `_vertexBuffer)))
			 (offsets[] (curly 0))
			 )
		     (declare (type VkBuffer vertexBuffers[])
			      (type VkDeviceSize offsets[]))
		     (vkCmdBindVertexBuffers
		      (aref ,(g `_commandBuffers) i)
		      0
		      1
		      vertexBuffers
		      offsets
		      ))
		   (vkCmdBindIndexBuffer
		    (aref ,(g `_commandBuffers) i)
		    ,(g `_indexBuffer)
		    0
		    VK_INDEX_TYPE_UINT32)
		   (vkCmdBindDescriptorSets
		    (aref ,(g `_commandBuffers) i)
		    ;; descriptor could also be bound to compute
		    VK_PIPELINE_BIND_POINT_GRAPHICS
		    ,(g `_pipelineLayout)
		    0
		    1
		    (ref (aref ,(g `_descriptorSets) i))
		    0
		    NULL)
		   ;; draw the triangle
		   (vkCmdDrawIndexed
		    (aref ,(g `_commandBuffers) i)
		    ,(g `_num_indices
			) ;; count
		    1		;; no instance rendering
		    0		;; offset to first index into buffer
		    0		;; offset to add to index
		    0		;; firstInstance
		    )
		   (vkCmdEndRenderPass
		    (aref ,(g `_commandBuffers) i))
		   ,(vkthrow `(vkEndCommandBuffer
			       (aref ,(g `_commandBuffers) i)))
		   )))))
 (define-module
	    `(sync_objects
	      ()
	      (do0
	       (defun createSyncObjects ()
		 (declare (values void))
		 ;(_imageAvailableSemaphores.resize _MAX_FRAMES_IN_FLIGHT)
		 ;(_renderFinishedSemaphores.resize _MAX_FRAMES_IN_FLIGHT)
		 ;(_inFlightFences.resize _MAX_FRAMES_IN_FLIGHT)
		 ,(vk
		   `(VkSemaphoreCreateInfo
		     semaphoreInfo
		     :sType VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO))
			   ,(vk
			     `(VkFenceCreateInfo
			       fenceInfo
			       :sType VK_STRUCTURE_TYPE_FENCE_CREATE_INFO
			       :flags VK_FENCE_CREATE_SIGNALED_BIT)
			     )
			   (dotimes (i _MAX_FRAMES_IN_FLIGHT)
			     ,(vkthrow `(vkCreateSemaphore
					 ,(g `_device)
					 &semaphoreInfo
					 NULL
					 (ref (aref ,(g `_imageAvailableSemaphores) i))))
			     ,(vkthrow `(vkCreateSemaphore
					 ,(g `_device)
					 &semaphoreInfo
					 NULL
					 (ref (aref ,(g `_renderFinishedSemaphores) i))))
			     ,(vkthrow `(vkCreateFence
					 ,(g `_device)
					 &fenceInfo
					 NULL
					 (ref (aref ,(g `_inFlightFences) i)))))))))
   (define-module
	    `(draw_frame
	      ()
	      (do0
	       (include <string.h>)
	       (defun now ()
		 (declare (values double))
		 (let ((tp))
		  (declare (type "struct timespec" tp))
		  ;; https://stackoverflow.com/questions/6749621/how-to-create-a-high-resolution-timer-in-linux-to-measure-program-performance
		  (clock_gettime CLOCK_REALTIME &tp)
		  (return (+ (cast double tp.tv_sec)
			     (* 1d-9 tp.tv_nsec)))))
	       
	       (defun updateUniformBuffer (currentImage)
			 (declare (type uint32_t currentImage)
				  (values void)
				  )
			 (let ((startTime))
			   (declare (type "static double" startTime)
				    )
			   (when (== 0d0 startTime)
			     (setf startTime (now)))
			  (let (
				(currentTime (now))
				(time (- currentTime startTime)
				  )
				)
			    (declare 
				     (type double currentTime)
				     (type double time)
				     )
			    ;; rotate model around z axis
			    (let ((zAxis (cast vec3 (curly 0s0 0s0 1s0)))
				  (eye (cast vec3 (curly 2s0 2s0 2s0)))
				  (center (cast vec3 (curly 0s0 0s0 0s0)))
				  (angularRate (glm_rad 9s0))
				  (rotationAngle (cast float (* time angularRate)))
				  (identity)
				  (model)
				  (look)
				  (projection))
			      (declare	;(type zAxis angularRate)
			       (type mat4 identity model look projection))
			      (do0
			       (glm_mat4_identity identity)
			       (glm_rotate_z identity rotationAngle model))
			      #-nolog-frame ,(vkprint "rotate" `(rotationAngle time startTime currentTime))
			      (do0
			       (glm_lookat ;; eye center up
				eye center zAxis look))

			      (do0
			       (glm_perspective ;; fovy aspect near far
				(glm_rad 45s0)
				(/ ,(g `_swapChainExtent.width)
				   (* 1s0 ,(g `_swapChainExtent.height)))
				.1s0
				10s0
				projection))
			      ,(vk
				`(UniformBufferObject
				  ubo
					;:model model
				  ;; look from above in 45 deg angle
					;:view look
				  ;; use current extent for correct aspect
					;:proj projection
				  )))
			    ;; glm was designed for opengl and has
			    ;; inverted y clip coordinate
			    (glm_mat4_copy model ubo.model)
			    (glm_mat4_copy look ubo.view)
			    (glm_mat4_copy projection ubo.proj)
			    (setf (aref ubo.proj 1 1)
				  (- (aref ubo.proj 1 1)))
			    (let ((data 0))
			      (declare (type void* data))
			      #-nolog-frame ,(vkprint "start map memory" `((aref ,(g `_uniformBuffersMemory)
								   currentImage)))
			      (vkMapMemory ,(g `_device)
					   (aref ,(g `_uniformBuffersMemory)
						 currentImage)
					   0
					   (sizeof ubo)
					   0
					   &data)
			      #-nolog-frame ,(vkprint "mapped memory" `(data (sizeof ubo)))
			      (memcpy data &ubo (sizeof ubo))
			      #-nolog-frame ,(vkprint "unmap memory" `((aref ,(g `_uniformBuffersMemory)
							       currentImage)))
			      (vkUnmapMemory ,(g `_device)
					     (aref ,(g `_uniformBuffersMemory)
						   currentImage)))
			    ;; note: a more efficient way to pass
			    ;; frequently changing values to shaders are
			    ;; push constants
			    ))
			 )
	       (defun recreateSwapChain ()
			  
			  
			  (let ((width 0)
				(height 0))
			    (declare (type int width height))
			    (while (or (== 0 width)
				       (== 0 height))
			      
			      (glfwGetFramebufferSize ,(g `_window)
						      &width
						      &height)
			      ,(vkprint "get frame buffer size" `(width height))
			      (glfwWaitEvents)))

			  ,(vkprint "wait idle")
			  (vkDeviceWaitIdle ,(g `_device)) ;; wait for resources to be not in use anymore
			  (cleanupSwapChain)
			  (createSwapChain)
			  (createImageViews)
			 
			  (createRenderPass)
			  (createGraphicsPipeline)
			  (createColorResources)
			  (createDepthResources)
			  (createFramebuffers)
			  (createUniformBuffers)
			  (createDescriptorPool)
			  (createDescriptorSets)
			  (createCommandBuffers)
			  ,(vkprint "swap chain has been recreated.")
			  )
	       (defun drawFrame ()
		 #-nolog-frame ,(vkprint "wait for fences" `((aref ,(g `_inFlightFences) ,(g `_currentFrame))
					       ,(g `_currentFrame)))
		 (do0
		  (vkWaitForFences ,(g `_device) 1 (ref (aref ,(g `_inFlightFences) ,(g `_currentFrame)))  VK_TRUE UINT64_MAX)
		  )
		 #-nolog-frame ,(vkprint "acquire next image" `(,(g `_swapChain) (aref ,(g `_imageAvailableSemaphores) ,(g `_currentFrame))))
		 (let ((imageIndex 0)
		       (result (vkAcquireNextImageKHR
				,(g `_device)
				,(g `_swapChain)
				UINT64_MAX ;; disable timeout for image 
				(aref ,(g `_imageAvailableSemaphores) ,(g `_currentFrame))
				VK_NULL_HANDLE
				&imageIndex)))
		   (declare (type uint32_t imageIndex))
		   #-nolog-frame ,(vkprint "next image is" `(imageIndex))
		   (when (== VK_ERROR_OUT_OF_DATE_KHR result)
		     (recreateSwapChain)
		     (return))
		   (unless (or (== VK_SUCCESS result)
			       (== VK_SUBOPTIMAL_KHR result))
		     ,(vkprint "failed to acquire swap chain image."))
		   (let ((waitSemaphores[] (curly (aref ,(g `_imageAvailableSemaphores) ,(g `_currentFrame))))
			 (signalSemaphores[] (curly (aref ,(g `_renderFinishedSemaphores) ,(g `_currentFrame))))
			 (waitStages[] (curly VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)))
		     (declare (type VkSemaphore waitSemaphores[]
				    signalSemaphores[])
			      (type VkPipelineStageFlags waitStages[]))
		     #-nolog-frame ,(vkprint "updateUniformBuffer" `(imageIndex))
		     (updateUniformBuffer imageIndex)
		     #-nolog-frame ,(vkprint "submit info")
		     ,(vk
		       `(VkSubmitInfo submitInfo
				      :sType VK_STRUCTURE_TYPE_SUBMIT_INFO
				      :waitSemaphoreCount 1
				      :pWaitSemaphores waitSemaphores
				      ;; pipeline has to wait for image before writing color buffer
				      :pWaitDstStageMask waitStages
				      :commandBufferCount 1
				      :pCommandBuffers (ref (aref ,(g `_commandBuffers) imageIndex))
				      :signalSemaphoreCount 1
				      :pSignalSemaphores signalSemaphores))
		     (vkResetFences ,(g `_device) 1 (ref (aref ,(g `_inFlightFences) ,(g `_currentFrame))))
		     ,(vkthrow
		       `(vkQueueSubmit
			 ,(g `_graphicsQueue)
			 1
			 &submitInfo
					;VK_NULL_HANDLE ;; fence
			 (aref ,(g `_inFlightFences) ,(g `_currentFrame))
			 ))
		     
		     ;; submit result for presentation
		     (let ((swapChains[] (curly ,(g `_swapChain)))
			   )
		       (declare (type VkSwapchainKHR swapChains[]))
		       ,(vk
			 `(VkPresentInfoKHR
			   presentInfo
			   :sType VK_STRUCTURE_TYPE_PRESENT_INFO_KHR
			   ;; wait for signal before presentation
			   :waitSemaphoreCount 1
			   :pWaitSemaphores signalSemaphores
			   :swapchainCount 1
			   :pSwapchains swapChains
			   :pImageIndices &imageIndex 
			   ;; we could check if presentation was successful
			   :pResults NULL))
		       (progn
			 (let ((result2 (vkQueuePresentKHR ,(g `_presentQueue) &presentInfo)))
			   (if (or (== VK_SUBOPTIMAL_KHR result2)
				   (== VK_ERROR_OUT_OF_DATE_KHR result2)
				   ,(g `_framebufferResized))
			       (do0
				(setf ,(g `_framebufferResized) false)
				(recreateSwapChain))
			       (unless (== VK_SUCCESS result2)
				 ,(vkprint "failed to present swap chain image.")))))
		       
					;(vkQueueWaitIdle ,_presentQueue) 
		       )
		     
		     ))
		 (do0
		  #-nolog-frame ,(vkprint "next frame from" `(,(g `_currentFrame)))
		  (setf ,(g `_currentFrame)
			(%
			 (+ 1 ,(g `_currentFrame))
			 _MAX_FRAMES_IN_FLIGHT))
		  #-nolog-frame ,(vkprint "next frame is" `(,(g `_currentFrame))))))))
   (define-module
	    `(cleanup
	      ()
	      (do0
	       (defun cleanupSwapChain ()
		 ,(vkprint "cleanupSwapChain")
		 (do0
		  (do0
		   ;; msaa
		   (vkDestroyImageView ,(g `_device)
				       ,(g `_colorImageView)
				       NULL)
		   (vkDestroyImage ,(g `_device)
				   ,(g `_colorImage)
				   NULL)
		   (vkFreeMemory ,(g `_device)
				 ,(g `_colorImageMemory)
				 NULL))
		  (do0
		   ;; depth
		   

		   ,(vkprint "cleanup depth"
			     `( ,(g `_depthImageView)
			       ,(g `_depthImage)
			       ,(g `_depthImageMemory)))
		   (vkDestroyImageView ,(g `_device)
				       ,(g `_depthImageView)
				       NULL)
		   (vkDestroyImage ,(g `_device)
				   ,(g `_depthImage)
				   NULL)
		   (vkFreeMemory ,(g `_device)
				 ,(g `_depthImageMemory)
				 NULL)
		   )

		  
		  (foreach (b ,(g `_swapChainFramebuffers))
			   ,(vkprint "framebuffer" `(b))
			   (vkDestroyFramebuffer ,(g `_device) b NULL))
		  (vkFreeCommandBuffers ,(g `_device)
					,(g `_commandPool)
					(length ,(g `_commandBuffers))
					,(g `_commandBuffers))
		  ,(vkprint "pipeline" `( ,(g `_graphicsPipeline)
					 ,(g `_pipelineLayout)
					 ,(g `_renderPass)))
		  (vkDestroyPipeline ,(g `_device) ,(g `_graphicsPipeline) NULL)
		  (vkDestroyPipelineLayout
		   ,(g `_device)
		   ,(g `_pipelineLayout)
		   NULL)
		  (vkDestroyRenderPass
		   ,(g `_device)
		   ,(g `_renderPass)
		   NULL)
		  (foreach (view ,(g `_swapChainImageViews))
			   ,(vkprint "image-view" `(view))
			   (vkDestroyImageView
			    ,(g `_device)
			    view
			    NULL))
		  ,(vkprint "swapchain" `( ,(g `_swapChain)))
		  (vkDestroySwapchainKHR ,(g `_device) ,(g `_swapChain) NULL)
		  ;; each swap chain image has a ubo
		  (dotimes (i (length ,(g `_swapChainImages)))
		    ,(vkprint "ubo" `((aref ,(g `_uniformBuffers) i)
				      (aref ,(g `_uniformBuffersMemory) i)
				      ))
		    (vkDestroyBuffer ,(g `_device)
				     (aref ,(g `_uniformBuffers) i)
				     NULL)
		    (vkFreeMemory ,(g `_device)
				  (aref ,(g `_uniformBuffersMemory) i)
				  NULL))
		  ,(vkprint "descriptor-pool" `( ,(g `_descriptorPool)))
		  (vkDestroyDescriptorPool
		   ,(g `_device)
		   ,(g `_descriptorPool)
		   NULL)
		  ))
	       (defun cleanup ()
		 (declare (values void))
		 
		 (do0
		  (cleanupSwapChain)
		  ,(vkprint "cleanup")
		  
		  (do0 ;; tex
		   ,(vkprint "tex"
			     `( ,(g `_textureSampler)
			       ,(g `_textureImageView)
			       ,(g `_textureImage)
			       ,(g `_textureImageMemory)
			       ,(g `_descriptorSetLayout)))
		   (vkDestroySampler ,(g `_device)
				     ,(g `_textureSampler)
				     NULL)
		   (vkDestroyImageView ,(g `_device)
				       ,(g `_textureImageView)
				       NULL)
		   (vkDestroyImage ,(g `_device)
				   ,(g `_textureImage) NULL)
		   (vkFreeMemory ,(g `_device)
				 ,(g `_textureImageMemory) NULL))
		  (vkDestroyDescriptorSetLayout
		   ,(g `_device)
		   ,(g `_descriptorSetLayout)
		   NULL)
		  ,(vkprint "buffers"
			    `(,(g `_vertexBuffer)
			      ,(g `_vertexBufferMemory)
			      ,(g `_indexBuffer)
			      ,(g `_indexBufferMemory)
			      ))
		  (do0 (vkDestroyBuffer ,(g `_device) ,(g `_vertexBuffer) NULL)
		       (vkFreeMemory ,(g `_device) ,(g `_vertexBufferMemory) NULL))
		  (do0 (vkDestroyBuffer ,(g `_device) ,(g `_indexBuffer) NULL)
		       (vkFreeMemory ,(g `_device) ,(g `_indexBufferMemory) NULL))
		  (dotimes (i _MAX_FRAMES_IN_FLIGHT)
		    (do0
		     ,(vkprint "sync"
			       `((aref ,(g `_renderFinishedSemaphores) i)
				 (aref ,(g `_imageAvailableSemaphores) i)
				 (aref ,(g `_inFlightFences) i)))
		     (vkDestroySemaphore ,(g `_device)
					 (aref ,(g `_renderFinishedSemaphores) i)
					 NULL)
		     (vkDestroySemaphore ,(g `_device)
					 (aref ,(g `_imageAvailableSemaphores) i)
					 NULL)
		     (vkDestroyFence ,(g `_device)
				     (aref ,(g `_inFlightFences) i)
				     NULL)))
		  ,(vkprint "cmd-pool"
			    `(,(g `_commandPool)))
		  (vkDestroyCommandPool ,(g `_device) ,(g `_commandPool) NULL)
		  
		  
		  )
		 ,(vkprint "rest"
			   `( ,(g `_device) ,(g `_instance) ,(g `_window)))
		 (vkDestroyDevice ,(g `_device) NULL)
		 
		 #+surface
		 (vkDestroySurfaceKHR ,(g `_instance) ,(g `_surface) NULL)
		 (vkDestroyInstance ,(g `_instance) NULL)
		 (glfwDestroyWindow ,(g `_window))
		 (glfwTerminate)
		 (cleanupModel)
		 ))))

   
   

  (let* ((vertex-code
	  `(do0
	    "#version 450"
	    "#extension GL_ARB_separate_shader_objects : enable"
	    " "
	    "layout(binding = 0) uniform UniformBufferObject { mat4 model; mat4 view; mat4 proj; } ubo;"
	    "layout(location = 0) in vec3 inPosition;" ;; if inPosition was dvec3 than next location would need to be 2
	    "layout(location = 1) in vec3 inColor;"
	    "layout(location = 2) in vec2 inTexCoord;"
	    "layout(location = 0) out vec3 fragColor;"
	    "layout(location = 1) out vec2 fragTexCoord;"
	    
	    
	    (defun main ()
	      (declare (values void))
	      (setf gl_Position
		    (* ubo.proj
		       ubo.view
		       ubo.model
		       (vec4 inPosition
			     1.))
		    fragColor inColor
		    fragTexCoord inTexCoord))
	    "// vertex shader end "))
	 (frag-code
	  `(do0
	    "#version 450"
	    "#extension GL_ARB_separate_shader_objects : enable"
	    " "
	    "layout(location = 0) out vec4 outColor;"
	    "layout(location = 0) in vec3 fragColor;"

	    "layout(location = 1) in vec2 fragTexCoord;"
	    "layout(binding = 1) uniform sampler2D texSampler;"
	    
	    
	    (defun main ()
	      (declare (values void))
	      (setf outColor
		    (texture texSampler fragTexCoord
			     )))))
	 (code
	  `(do0
	    "// https://vulkan-tutorial.com/en/Drawing_a_triangle/Setup/Base_code"
	    "// https://vulkan-tutorial.com/en/Drawing_a_triangle/Setup/Validation_layers"
	    "// https://gpuopen.com/understanding-vulkan-objects/"
	    "/* g++ -std=c++17 run_01_base.cpp  `pkg-config --static --libs glfw3` -lvulkan -o run_01_base -pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel -Wstrict-overflow=5 -Wswitch-default -Wundef -march=native -O2 -g  -ftime-report */"
	    " "
	    
	    )))
					;(write-source *code-file* code)
    ;; we need an empty proto2.h. it has to be written before all c files so that make proto will work
    (write-source (asdf:system-relative-pathname 'cl-cpp-generator2 "example/05_vulkan_generic_c/source/proto2.h")
		  `(do0)  (user-homedir-pathname) t)

    (loop for e in (reverse *module*) and i from 0 do
	 (destructuring-bind (&key name code) e
	   (write-source (asdf:system-relative-pathname
			  'cl-cpp-generator2
			  (format nil
				  "example/05_vulkan_generic_c/source/vulkan_~2,'0d_~a.c"
				  i name))
			 code)))
    (write-source (asdf:system-relative-pathname
		   'cl-cpp-generator2
		   "example/05_vulkan_generic_c/source/utils.h"
		   )

		  `(do0
		    "#ifndef UTILS_H"
		    " "
		    "#define UTILS_H"
		    " "
		    
		    (do0
		    (include <stdio.h>)
		    " "
		    (include <stdbool.h>)
		    ;;"#define _POSIX_C_SOURCE 199309L"
		    " "
		    ;;(include <unistd.h>)
		    (include <time.h>)

		    " "
		    (include <cglm/cglm.h>)
		    " "
		    ,@(loop for e in *utils-code* collect
			 e)
		    "#define length(a) (sizeof((a))/sizeof(*(a)))"
					;"#define max(a,b)  ({ __typeof__ (a) _a = (a);  __typeof__ (b) _b = (b);  _a > _b ? _a : _b; })"
					;"#define min(a,b)  ({ __typeof__ (a) _a = (a);  __typeof__ (b) _b = (b);  _a < _b ? _a : _b; })"
		    "#define max(a,b) ({ __auto_type _a = (a);  __auto_type _b = (b); _a > _b ? _a : _b; })"
		    "#define min(a,b) ({ __auto_type _a = (a);  __auto_type _b = (b); _a < _b ? _a : _b; })"
		    "#define printf_dec_format(x) _Generic((x), default: \"%p\", char: \"%c\", signed char: \"%hhd\", unsigned char: \"%hhu\", signed short: \"%hd\", unsigned short: \"%hu\", signed int: \"%d\", unsigned int: \"%u\", long int: \"%ld\", unsigned long int: \"%lu\", long long int: \"%lld\", float: \"%f\", double: \"%f\", long double: \"%Lf\", char*: \"%s\", const char*: \"%s\", unsigned long long int: \"%llu\",void*: \"%p\",bool:\"%d\")"
		    ,(format nil "#define type_string(x) _Generic((x), ~{~a: \"~a\"~^,~})"
			     (loop for e in `(default
						 
						 ,@(loop for h in
							`(bool
							  ,@(loop for f in `(char short int "long int" "long long int") appending
								 `(,f ,(format nil "unsigned ~a" f)))
							  float double "long double"
							  "char*"
							  "void*"
							  )
						      appending
							`(,h ,(format nil "const ~a" h)))
						 						 
						 )
				appending
				  `(,e ,e)))
		    

		    
		    
		    " "
		    
		    )
		    " "
		    "#endif"
		    " ")
		  )
    (write-source *vertex-file* vertex-code)
    (write-source *frag-file* frag-code)
    (write-source (asdf:system-relative-pathname 'cl-cpp-generator2 "example/05_vulkan_generic_c/source/globals.h")
		  `(do0
		    "#ifndef GLOBALS_H"
		    " "
		    "#define GLOBALS_H"
		    " "
		    
		    
		    ,(emit-globals)
		    " "
		    "#endif"
		    " "))
    
    
    (sb-ext:run-program "/usr/bin/glslangValidator" `("-V" ,(format nil "~a" *frag-file*)
							   "-o"
							   ,(format nil "~a/frag.spv"
								    (directory-namestring *vertex-file*))))
    (sb-ext:run-program "/usr/bin/glslangValidator" `("-V" ,(format nil "~a" *vertex-file*)
							   "-o"
							   ,(format nil "~a/vert.spv"
								    (directory-namestring *vertex-file*))))
    ;; we need to force clang-format to always have the return type in the same line as the function: PenaltyReturnTypeOnItsOwnLine
					;(sb-ext:run-program "/bin/sh" `("gen_proto.sh"))
    (sb-ext:run-program "/usr/bin/make" `("-C" "source" "-j4" "proto2.h"))))
 

