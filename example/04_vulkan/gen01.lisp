(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre"))

(in-package :cl-cpp-generator2)

;; if nolog is off, then validation layers will be used to check for mistakes
;; if surface is on, then a window surface is created; otherwise only off-screen render
(setf *features* (union *features* '(:surface ;:nolog
				     )))
(setf *features* (set-difference *features* '(:nolog)))



(progn
  (defun with-single-time-commands (args)
    (destructuring-bind ((buffer) &rest body) args
     `(let ((,buffer
	     (beginSingleTimeCommands)))
	,@body
	(endSingleTimeCommands ,buffer))))
  (defun vkthrow (cmd)
    `(unless (== VK_SUCCESS
		 ,cmd)
       (throw ("std::runtime_error"
	       (string ,(substitute #\Space #\Newline (format nil "failed to ~a" cmd)))))))
  (defun vkprint (msg
		  rest)
    `(<< "std::cout"
	 (string ,(format nil "~a: " msg))
	 ,@(loop for e in rest appending
		`((string ,(format nil " ~a=" e))
		  ,e))
	 "std::endl"))
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
		(<< "std::cout"
		    ,@(if instance
			  `((string ,(format nil "~a ~a ~a=" verb subject instance))
			    ,instance)
			 `((string ,(format nil "~a ~a" verb subject))))
		    "std::endl"))))))
    
    )
  
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
		  `(setf (dot ,var ,keyword) ,value))))))
  (defparameter *code-file* (asdf:system-relative-pathname 'cl-cpp-generator2 "example/04_vulkan/source/run_01_base.cpp"))
  (defparameter *vertex-file* (asdf:system-relative-pathname 'cl-cpp-generator2 "example/04_vulkan/source/run_01_base.vert"))
  (defparameter *frag-file* (asdf:system-relative-pathname 'cl-cpp-generator2 "example/04_vulkan/source/run_01_base.frag"))
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
	    (do0 "#define GLFW_INCLUDE_VULKAN"
		 (include <GLFW/glfw3.h>)
		 " ")
	    #+nil
	    (include <vulkan/vulkan.h>)
	    
	    (do0
	     "#define GLM_FORCE_RADIANS"
	     "#define GLM_FORCE_DEPTH_ZERO_TO_ONE"
	     ;; by default glm uses -1 .. 1 depth buffer range
	     " "
	     (include <glm/vec4.hpp>
		      <glm/mat4x4.hpp>
		      <glm/glm.hpp>
		      <glm/gtc/matrix_transform.hpp>
		      <chrono>)
	     " "
	     )
	    
	    " "
	    (include <iostream>
		     <stdexcept>
		     <functional>
		     <cstdlib>
		     <cstring>
		     <optional>
		     <set>)

	    

	    #+surface
	    (include
	     ;; UINT32_MAX:
	     <cstdint> 
	     <algorithm>
	     )

	    (do0
	     "#define STB_IMAGE_IMPLEMENTATION"
	     (include "stb_image.h")
	     " ")
	    (do0
	     "#define TINYOBJLOADER_IMPLEMENTATION"
	     (include "tiny_obj_loader.h")
	     " ")
	    
	    (do0
	     "// code to load binary shader from file"
	     (include <fstream>)

	     (do0
	      "typedef struct SwapChainSupportDetails SwapChainSupportDetails;"
	      "typedef struct QueueFamilyIndices QueueFamilyIndices;"
	      "std::vector<char> readFile(const std::string&);"
	      "SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice, VkSurfaceKHR);"
	      "VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>&);"
	      "VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>&);"
	      "VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR&, GLFWwindow*);"
	      "QueueFamilyIndices findQueueFamilies(VkPhysicalDevice, VkSurfaceKHR);"
	      )
	     
	     
	     (defun readFile (filename)
	       (declare (type "const std::string&" filename)
			(values "std::vector<char>"))
	       (let ((file ("std::ifstream" filename (logior "std::ios::ate"
							     "std::ios::binary"))))
		 (unless (file.is_open)
		   (throw ("std::runtime_error"
			   (string "failed to open file."))))
		 (let ((fileSize (file.tellg))
		       (buffer ("std::vector<char>" fileSize)))
		   (file.seekg 0)
		   (file.read (buffer.data)
			      fileSize)
		   (file.close)
		   (return buffer)))))
	    (defstruct0 UniformBufferObject
		(model "glm::mat4")
	      (view "glm::mat4")
	      (proj "glm::mat4"))
	    #+nil
	    (defstruct0 UniformBufferObject
		(model "alignas(16) glm::mat4")
	      (view "alignas(16) glm::mat4")
	      (proj "alignas(16) glm::mat4"))
	    
	    (defstruct0 Vertex
		(pos "glm::vec3")
	      (color "glm::vec3")
	      (texCoord "glm::vec2")
	      ;; here static means the function has no receiver object
	      ;; outside of the class static would limit scope to only
	      ;; this file (which i don't necessarily want)
	      ("getBindingDescription()" "static VkVertexInputBindingDescription")
	      ("getAttributeDescriptions()"
	       "static std::array<VkVertexInputAttributeDescription,3>"))
	    (defun "Vertex::getBindingDescription" ()
	      (declare (values "VkVertexInputBindingDescription"))
	      ,(vk
		`(VkVertexInputBindingDescription
		  bindingDescription
		  :binding 0
		  :stride (sizeof Vertex)
		  ;; move to next data after each vertex
		  :inputRate VK_VERTEX_INPUT_RATE_VERTEX))
	      (return bindingDescription))
	    
	    (let ((g_vertices #+nil (curly
			       (curly (curly  -.5s0 -.5s0 0s0) (curly 1s0 0s0 0s0) (curly 1s0 0s0))
			       (curly (curly  .5s0  -.5s0 0s0) (curly 0s0 1s0 0s0) (curly 0s0 0s0))
			       (curly (curly .5s0  .5s0 0s0) (curly 0s0 0s0 1s0) (curly 0s0 1s0))
			       (curly (curly -.5s0  .5s0 0s0) (curly 1s0 1s0 1s0) (curly 1s0 1s0))

			       (curly (curly  -.5s0 -.5s0 -.5s0) (curly 1s0 0s0 0s0) (curly 1s0 0s0))
			       (curly (curly  .5s0  -.5s0 -.5s0) (curly 0s0 1s0 0s0) (curly 0s0 0s0))
			       (curly (curly .5s0  .5s00 -.5s0) (curly 0s0 0s0 1s0) (curly 0s0 1s0))
			       (curly (curly -.5s0  .5s0 -.5s0) (curly 1s0 1s0 1s0) (curly 1s0 1s0))))
		  (g_indices #+nil (curly 0 1 2 2 3 0
				    4 5 6 6 7 4)))
	      (declare (type "std::vector<Vertex>" g_vertices)
		       (type "std::vector<uint32_t>" g_indices)))
	    (defun "Vertex::getAttributeDescriptions" ()
	      (declare (values "std::array<VkVertexInputAttributeDescription,3>"))
	      (let ((attributeDescriptions (curly)))
		(declare (type "std::array<VkVertexInputAttributeDescription,3>"
			       attributeDescriptions))
		,(set-members `((aref attributeDescriptions 0)
				:binding 0
				:location 0
				:format VK_FORMAT_R32G32B32_SFLOAT
				:offset (offsetof Vertex pos)))
		,(set-members `((aref attributeDescriptions 1)
				:binding 0
				:location 1
				:format VK_FORMAT_R32G32B32_SFLOAT
				:offset (offsetof Vertex color)))
		,(set-members `((aref attributeDescriptions 2)
				:binding 0
				:location 2
				:format VK_FORMAT_R32G32_SFLOAT
				:offset (offsetof Vertex texCoord)))
		(return attributeDescriptions)))
	    
	    (defstruct0 QueueFamilyIndices 
		(graphicsFamily "std::optional<uint32_t>")
	      #+surface (presentFamily "std::optional<uint32_t>")
	      ("isComplete()" bool))
	    (defun "QueueFamilyIndices::isComplete" ()
	      (declare (values bool))
	      (return (and
		       (graphicsFamily.has_value)
		       #+surface (presentFamily.has_value))))

	    #+surface
	    (do0
	     (defstruct0 SwapChainSupportDetails
		 (capabilities VkSurfaceCapabilitiesKHR)
	       (formats "std::vector<VkSurfaceFormatKHR>")
	       (presentModes "std::vector<VkPresentModeKHR>")
	       )
	     (defun querySwapChainSupport (device surface)
	       (declare (values SwapChainSupportDetails)
			(type VkPhysicalDevice device)
			(type VkSurfaceKHR surface))
	       (let ((details))
		 (declare (type SwapChainSupportDetails details))
		 (vkGetPhysicalDeviceSurfaceCapabilitiesKHR
		  device
		  surface
		  &details.capabilities)
		 

		 (let ((formatCount 0))
		   (declare (type uint32_t formatCount))
		   (vkGetPhysicalDeviceSurfaceFormatsKHR device surface &formatCount
							 nullptr)
		   (unless (== 0 formatCount)
		     (details.formats.resize formatCount)
		     (vkGetPhysicalDeviceSurfaceFormatsKHR
		      device surface &formatCount
		      (details.formats.data))))

		 (let ((presentModeCount 0))
		   (declare (type uint32_t presentModeCount
				  ))
		   (vkGetPhysicalDeviceSurfacePresentModesKHR
		    device surface &presentModeCount
		    nullptr)
		   (unless (== 0 presentModeCount)
		     (details.presentModes.resize presentModeCount)
		     (vkGetPhysicalDeviceSurfacePresentModesKHR
		      device surface &presentModeCount
		      (details.presentModes.data))))
		 
		 (return details)))

	     (defun chooseSwapSurfaceFormat (availableFormats)
	       (declare (values VkSurfaceFormatKHR)
			(type "const std::vector<VkSurfaceFormatKHR>&"
			      availableFormats))
	       (foreach (format availableFormats)
			(when (and (== VK_FORMAT_B8G8R8A8_UNORM
				       format.format)
				   (== VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
				       format.colorSpace))
			  (return format)))
	       (return (aref availableFormats 0)))
	     (defun chooseSwapPresentMode (modes)
	       (declare (values VkPresentModeKHR)
			(type "const std::vector<VkPresentModeKHR>&"
			      modes))
	       "// prefer triple buffer (if available)"
	       (foreach (mode modes)
			(when (== VK_PRESENT_MODE_MAILBOX_KHR mode)
			  (return mode)))
	       (return VK_PRESENT_MODE_FIFO_KHR))
	     (defun chooseSwapExtent (capabilities _window)
	       (declare (values VkExtent2D)
			(type "const VkSurfaceCapabilitiesKHR&"
			      capabilities)
			(type GLFWwindow* _window))
	       (if (!= UINT32_MAX capabilities.currentExtent.width)
		   (do0
		    (return capabilities.currentExtent))
		   (do0
		    (let ((width 0)
			  (height 0)
			  )
		      (declare (type int width height))
		      (glfwGetFramebufferSize _window &width &height)
		      (let ((actualExtent (curly (static_cast<uint32_t> width)
						 (static_cast<uint32_t> height)))
			    )
			(declare (type VkExtent2D actualExtent))

			,@(loop for e in `(width height) collect
			       `(setf (dot actualExtent ,e)
				      ("std::max" (dot capabilities.minImageExtent ,e)
						  ("std::min"
						   (dot capabilities.maxImageExtent ,e)
						   (dot actualExtent ,e)))))
			
			(return actualExtent)))))))

	    #+surface
	    (do0
	     (defun checkDeviceExtensionSupport (device _deviceExtensions)
	       (declare (values bool)
			(type VkPhysicalDevice device)
			(type "const std::vector<const char*>" _deviceExtensions))
	       (let ((extensionCount 0))
		 (declare (type uint32_t extensionCount))
		 (vkEnumerateDeviceExtensionProperties
		  device nullptr &extensionCount nullptr)
		 (let (((availableExtensions extensionCount)))
		   (declare (type
			     "std::vector<VkExtensionProperties>"
			     (availableExtensions extensionCount)
			     ))
		   (vkEnumerateDeviceExtensionProperties
		    device
		    nullptr
		    &extensionCount
		    (availableExtensions.data))
		   (let (((requiredExtensions
			   (_deviceExtensions.begin)
			   (_deviceExtensions.end))))
		     (declare (type
			       "std::set<std::string>"
			       (requiredExtensions
				(_deviceExtensions.begin)
				(_deviceExtensions.end))))
		     (foreach (extension availableExtensions)
			      (requiredExtensions.erase
			       extension.extensionName))
		     (return (requiredExtensions.empty))))))
	     
	     (defun findQueueFamilies (device _surface )
	       (declare (type VkPhysicalDevice device)
			(type VkSurfaceKHR _surface)
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
			  (setf indices.presentFamily i))
			(when (indices.isComplete)
			  break))
		      (incf i))))
		 (return indices)))

	     (defun isDeviceSuitable ( device _surface  _deviceExtensions)
	       (declare (values bool)
			(type VkPhysicalDevice device)
			(type VkSurfaceKHR _surface)
			(type "const std::vector<const char*>" _deviceExtensions))
	       #+surface
	       (let ((extensionsSupported (checkDeviceExtensionSupport device  _deviceExtensions))
		     (swapChainAdequate false))
		 (declare (type bool swapChainAdequate))
		 (when extensionsSupported
		   (let ((swapChainSupport (querySwapChainSupport device _surface)))
		     (setf swapChainAdequate
			   (and (not (swapChainSupport.formats.empty))
				(not (swapChainSupport.presentModes.empty)))))))
	       (let ((indices (findQueueFamilies device _surface))
		     (supportedFeatures))
		 (declare (type QueueFamilyIndices indices)
			  (type VkPhysicalDeviceFeatures
				supportedFeatures))
		 (vkGetPhysicalDeviceFeatures device
					      &supportedFeatures)
		 (return (and (indices.isComplete)
			      supportedFeatures.samplerAnisotropy
			      #+surface (and 
					 extensionsSupported
					 swapChainAdequate)))
		 #+nil (return (indices.isComplete)))))
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
		    #-nolog (_validationLayers (curly (string "VK_LAYER_KHRONOS_validation")))
		    (_physicalDevice VK_NULL_HANDLE)
		    (_device)
		    (_graphicsQueue)
		    )
		(declare (type GLFWwindow* _window)
			 (type VkInstance _instance)
			 #-nolog (type "const bool" _enableValidationLayers)
			 #-nolog (type "const std::vector<const char*>" _validationLayers)
			 (type VkPhysicalDevice _physicalDevice)
			 (type VkDevice _device)
			 (type VkQueue _graphicsQueue)
			 
			 )

		(let ((_textureImage)
		      (_textureImageMemory)
		      (_textureImageView)
		      (_textureSampler))
		  (declare (type
			    VkImage
			    _textureImage)
			   (type
			    VkDeviceMemory
			    _textureImageMemory)
			   (type VkImageView
				 _textureImageView)
			   (type VkSampler _textureSampler)))
		(let ((_depthImage)
		      (_depthImageMemory)
		      (_depthImageView))
		  (declare (type VkImage _depthImage)
			   (type VkDeviceMemory _depthImageMemory)
			   (type VkImageView _depthImageView))
		 (let #-surface ()
		      #+surface ((_deviceExtensions (curly VK_KHR_SWAPCHAIN_EXTENSION_NAME))
				 (_presentQueue)
				 (_surface)
				 (_swapChain)
				 (_swapChainImages)
				 (_swapChainImageFormat)
				 (_swapChainExtent)
				 (_swapChainImageViews)
				 (_renderPass)
				 (_descriptorSetLayout)
				 (_pipelineLayout)
				 (_graphicsPipeline)
				 (_swapChainFramebuffers)
				 (_commandPool)
				 (_commandBuffers)
				 (_imageAvailableSemaphores)
				 (_renderFinishedSemaphores)
				 (_inFlightFences)
				 (_MAX_FRAMES_IN_FLIGHT 2)
				 (_currentFrame 0)
				 (_framebufferResized false)
				 (_vertexBuffer) (_vertexBufferMemory)
				 (_indexBuffer) (_indexBufferMemory)
				 (_uniformBuffers)
				 (_uniformBuffersMemory)
				 (_descriptorPool)
				 (_descriptorSets)
				 )
		      #+surface (declare
				 (type VkQueue 
				       _presentQueue)
				
				 (type VkSurfaceKHR _surface)
				 (type "const std::vector<const char*>"
				       _deviceExtensions)
				 (type VkSwapchainKHR _swapChain)
				 (type "std::vector<VkImage>" _swapChainImages)
				 (type VkFormat _swapChainImageFormat)
				 (type VkExtent2D _swapChainExtent)
				 (type "std::vector<VkImageView>" _swapChainImageViews)
				 (type VkDescriptorSetLayout _descriptorSetLayout)
				 (type VkPipelineLayout _pipelineLayout)
				 (type VkRenderPass _renderPass)
				 (type VkPipeline _graphicsPipeline)
				 (type "std::vector<VkFramebuffer>" _swapChainFramebuffers)
				 (type VkCommandPool _commandPool)
				 (type "std::vector<VkCommandBuffer>"
				       _commandBuffers)
				 (type "std::vector<VkSemaphore>" _imageAvailableSemaphores
				       _renderFinishedSemaphores)
				 (type "const int" _MAX_FRAMES_IN_FLIGHT)
				 (type size_t _currentFrame)
				 (type "std::vector<VkFence>" _inFlightFences)
				 (type bool _framebufferResized)
				 (type VkBuffer _vertexBuffer
				       _indexBuffer)
				 (type VkDeviceMemory
				       _vertexBufferMemory
				       _indexBufferMemory)
				 (type "std::vector<VkBuffer>"
				       _uniformBuffers)
				 (type "std::vector<VkDeviceMemory>"
				       _uniformBuffersMemory)
				 (type VkDescriptorPool
				       _descriptorPool)
				 (type "std::vector<VkDescriptorSet>"
				       _descriptorSets))
		     

		     
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
		      (defun framebufferResizeCallback (window width height)
			(declare (values "static void")
				 ;; static because glfw doesnt know how to call a member function with a this pointer
				 (type GLFWwindow* window)
				 (type int width height))
			(let ((app (reinterpret_cast<HelloTriangleApplication*> (glfwGetWindowUserPointer window))))
			  (setf app->_framebufferResized true)))
		      (defun initWindow ()
			(declare (values void))
			(glfwInit)
			(glfwWindowHint GLFW_CLIENT_API GLFW_NO_API)
			(glfwWindowHint GLFW_RESIZABLE GLFW_FALSE)
			(setf _window (glfwCreateWindow 800 600
							(string "vulkan window")
							nullptr
							nullptr))
			;; store this pointer to the instance for use in the callback
			(glfwSetWindowUserPointer _window this)
			(glfwSetFramebufferSizeCallback _window
							framebufferResizeCallback))
		      (defun createInstance ()
			(declare (values void))
			"// initialize member _instance"
			#-nolog ( ;when (and _enableValidationLayers  (not (checkValidationLayerSupport)))
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
			  ,(vkcall
			    `(create
			      instance
			      (:pApplicationInfo &appInfo
						 :enabledExtensionCount glfwExtensionCount
						 :ppEnabledExtensionNames glfwExtensions
						 :enabledLayerCount
						 #+nolog 0
						 #-nolog ("static_cast<uint32_t>"
							  (_validationLayers.size))
						 :ppEnabledLayerNames
						 #+nolog nullptr
						 #-nolog (_validationLayers.data))
			      (&info
			       nullptr
			       &_instance)
			      _instance
			      )
			    :throw t)))
		      (defun createBuffer (size usage properties )
			;; https://www.fluentcpp.com/2018/06/19/3-simple-c17-features-that-will-make-your-code-simpler/
			(declare (values "std::tuple<VkBuffer,VkDeviceMemory>")
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
			      (_device
			       &info
			       nullptr
			       &buffer)
			      buffer
			      )
			    :throw t)
			 
			  (let ((memReq))
			    (declare (type VkMemoryRequirements memReq))
			    (vkGetBufferMemoryRequirements _device
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
				(_device
				 &info
				 nullptr
				 &bufferMemory)
				bufferMemory)
			      :throw t)
			  
			    (vkBindBufferMemory _device
						buffer
						bufferMemory
						0)
			    (return ("std::make_tuple"
				     buffer
				     bufferMemory)))))
		      (defun createVertexBuffer ()
			(declare (values void))
			(let ((bufferSize (* (sizeof (aref g_vertices 0))
					     (g_vertices.size)))
			      ((bracket stagingBuffer
					stagingBufferMemory)
			       (createBuffer
				bufferSize
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT
				(logior
				 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
				 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))))
			 			
			  (let ((data))
			    (declare (type void* data))
			    (vkMapMemory _device stagingBufferMemory
					 0	  ;; offset
					 bufferSize ;; size
					 0	    ;; flags
					 &data)
			    (memcpy data
				    (g_vertices.data)
				    bufferSize)
			    ;; without coherent bit, the changed memory
			    ;; might not immediatly be visible.
			    ;; alternatively: vkFlushMappedMemoryRanges
			    ;; or vkInvalidateMappedMemoryRanges; the
			    ;; memory transfer is defined to be
			    ;; complete as of the next call to
			    ;; vkQueueSubmit
			    (vkUnmapMemory _device stagingBufferMemory)))

			(let  (((bracket
				 vertexBuffer
				 vertexBufferMemory)
				(createBuffer
				 bufferSize
				 (logior
				  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
				  ;; can be a data transfer destination
				  VK_BUFFER_USAGE_TRANSFER_DST_BIT)
				 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)))
			  (setf _vertexBuffer vertexBuffer
				_vertexBufferMemory vertexBufferMemory)
			  (copyBuffer stagingBuffer
				      _vertexBuffer
				      bufferSize))
		       
			(do0
			 (vkDestroyBuffer _device stagingBuffer nullptr)
			 (vkFreeMemory _device stagingBufferMemory nullptr)))
		      (defun createIndexBuffer ()
			(declare (values void))
			(let ((bufferSize (* (sizeof (aref g_indices 0))
					     (g_indices.size)))
			      ((bracket stagingBuffer
					stagingBufferMemory)
			       (createBuffer
				bufferSize
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT
				(logior
				 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
				 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))))
			 			
			  (let ((data))
			    (declare (type void* data))
			    (vkMapMemory _device stagingBufferMemory
					 0	  ;; offset
					 bufferSize ;; size
					 0	    ;; flags
					 &data)
			    (memcpy data
				    (g_indices.data)
				    bufferSize)
			  
			    (vkUnmapMemory _device stagingBufferMemory)))

			(let  (((bracket
				 indexBuffer
				 indexBufferMemory)
				(createBuffer
				 bufferSize
				 (logior
				  VK_BUFFER_USAGE_INDEX_BUFFER_BIT
				  ;; can be a data transfer destination
				  VK_BUFFER_USAGE_TRANSFER_DST_BIT)
				 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)))
			  (setf _indexBuffer indexBuffer
				_indexBufferMemory indexBufferMemory)
			  (copyBuffer stagingBuffer
				      _indexBuffer
				      bufferSize))
		       
			(do0
			 (vkDestroyBuffer _device stagingBuffer nullptr)
			 (vkFreeMemory _device stagingBufferMemory nullptr)))
		      (defun
			  beginSingleTimeCommands ()
			(declare (values VkCommandBuffer))
			(let ((commandBuffer))
			  (declare (type VkCommandBuffer commandBuffer))
			  ,(vkcall
			    `(allocate
			      command-buffer
			      (:level VK_COMMAND_BUFFER_LEVEL_PRIMARY
				      :commandPool _commandPool
				      :commandBufferCount 1
				      )
			      (_device &info &commandBuffer)
			      
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
			(vkQueueSubmit _graphicsQueue
				       1
				       &submitInfo
				       VK_NULL_HANDLE)
			(vkQueueWaitIdle _graphicsQueue)
			(vkFreeCommandBuffers
			 _device
			 _commandPool
			 1
			 &commandBuffer)
			(<<
			 "std::cout"
			 (string "endSingleTimeCommands ")
			 commandBuffer
			 "std::endl"))
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

		      (defun findMemoryType (typeFilter properties)
			(declare (values uint32_t)
				 (type uint32_t typeFilter)
				 (type VkMemoryPropertyFlags properties))
			(let ((ps))
			  (declare (type VkPhysicalDeviceMemoryProperties ps))
			  (vkGetPhysicalDeviceMemoryProperties _physicalDevice
							       &ps)
			  (dotimes (i ps.memoryTypeCount)
			    (when (and (logand (<< 1 i)
					       typeFilter)
				       (== properties
					   (logand properties
						   (dot (aref ps.memoryTypes i)
							propertyFlags))))
			      (return i)))
			  (throw ("std::runtime_error"
				  (string "failed to find suitable memory type.")))))
		      (defun initVulkan ()
			(declare (values void))
			(createInstance)
			#+surface
			(do0 "// create window surface because it can influence physical device selection"
			     (createSurface))
			(pickPhysicalDevice)
			(createLogicalDevice)
			#+surface
			(do0
			 (createSwapChain)
			 (createImageViews)
			 (createRenderPass)
			 (createDescriptorSetLayout)
			 (createGraphicsPipeline)
			 
			 (createCommandPool)
			 ;; create texture image needs command pools
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
			 (createSyncObjects)))

		      (do0
		       (defun loadModel ()
			 (declare (values void))
			 (let ((attrib)
			       (shapes)
			       (materials)
			       (warning)
			       (err))
			   (declare (type "tinyobj::attrib_t" attrib)
				    (type "std::vector<tinyobj::shape_t>"
					  shapes)
				    (type "std::vector<tinyobj::material_t>"
					  materials)
				    (type "std::string"
					  warning err))
			   (unless ("tinyobj::LoadObj"
				    &attrib
				    &shapes
				    &materials
				    &warning
				    &err
				    (string "chalet.obj"))
			     (throw ("std::runtime_error"
				     (+ warning err))))
			   (foreach
			    (shape shapes)
			    (foreach
			     (index shape.mesh.indices)
			     ,(vk
			       `(Vertex
				 vertex
				 :pos (curly
				       ,@(loop for i below 3 collect
					      `(aref attrib.vertices
						     (+ ,i (* 3 index.vertex_index)))))
				 :texCoord (curly
					    (aref attrib.texcoords
						  (+ 0 (* 2 index.texcoord_index)))
					    (- 1s0 (aref attrib.texcoords
						   (+ 1 (* 2 index.texcoord_index)))))
				 :color (curly 1s0 1s0 1s0)))
			     (g_vertices.push_back vertex)
			     (g_indices.push_back (g_indices.size)))))))
		      (do0
		       (defun findSupportedFormat (candidates
						   tiling
						   features)
			 (declare (values VkFormat)
				  (type "const std::vector<VkFormat>&" candidates)
				  (type VkImageTiling tiling)
				  (type VkFormatFeatureFlags features))
			 (foreach (format candidates)
				  (let ((props))
				    (declare (type VkFormatProperties props))
				    (vkGetPhysicalDeviceFormatProperties _physicalDevice
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
			 (throw ("std::runtime_error"
				 (string "failed to find supported format!")))
			 
			 )
		       (defun findDepthFormat ()
			 (declare (values VkFormat))
			 (return (findSupportedFormat
				  (curly
				   VK_FORMAT_D32_SFLOAT
				   VK_FORMAT_D32_SFLOAT_S8_UINT
				   VK_FORMAT_D24_UNORM_S8_UINT)
				  VK_IMAGE_TILING_OPTIMAL
				  VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)))
		       (defun hasStencilComponent (format)
			 (declare (values bool)
				  (type VkFormat format))
			 (return (or
				  (== VK_FORMAT_D32_SFLOAT_S8_UINT format)
				  (== VK_FORMAT_D24_UNORM_S8_UINT format))))
		       (defun createDepthResources ()
			 (declare (values void))
			 (let ((depthFormat (findDepthFormat))
			       ((bracket depthImage
					 depthImageMemory)
				(createImage _swapChainExtent.width
					     _swapChainExtent.height
					     depthFormat
					     VK_IMAGE_TILING_OPTIMAL
					     VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
					     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
					     
					     )))
			   (setf _depthImage depthImage
				 _depthImageMemory depthImageMemory
				 _depthImageView
				 (createImageView _depthImage
						  depthFormat
						  VK_IMAGE_ASPECT_DEPTH_BIT))
			   (transitionImageLayout
			    _depthImage
			    depthFormat
			    VK_IMAGE_LAYOUT_UNDEFINED
			    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL))))
		      
		      (do0
		      
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
				    :imageOffset (curly 0 0 0)
				    :imageExtent (curly width height 1)))
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
		       (defun transitionImageLayout (image
						     format
						     oldLayout
						     newLayout)
			 (declare (values void)
				  (type VkImage image)
				  (type VkFormat format)
				  (type VkImageLayout oldLayout newLayout)
				  )
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
			       :subresourceRange.levelCount 1
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
				      (do0
				       (throw
					   ("std::invalid_argument"
					    (string "unsupported layout transition."))))))))
			   (vkCmdPipelineBarrier
			    commandBuffer
			    ;; https://www.khronos.org/registry/vulkan/specs/1.0/html/vkspec.html#synchronization-access-types-supported
			    srcStage ;; stage that should happen before barrier
			    dstStage ;; stage that will wait
			    0	     ;; per-region
			    0 nullptr ;; memory barrier
			    0 nullptr ;; buffer memory barrier
			    1 &barrier ;; image memory barrier
			    )
			  
			   (endSingleTimeCommands commandBuffer)))
		       (defun createImage (width height
					   format tiling
					   usage
					   properties)
			 (declare (values
				   "std::tuple<VkImage,VkDeviceMemory>")
				  (type uint32_t width height)
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
				:mipLevels 1
				:arrayLayers 1
				:format format
				;; if you need direct access, use linear tiling for row major
				:tiling tiling
				:initialLayout VK_IMAGE_LAYOUT_UNDEFINED
				:usage usage
			      
				:sharingMode
				VK_SHARING_MODE_EXCLUSIVE
				:samples
				VK_SAMPLE_COUNT_1_BIT
				:flags 0)
			       (_device
				&info
				nullptr
				&image)
			       image
			       )
			     :throw t))
			 (let ((memReq))
			   (declare (type VkMemoryRequirements memReq))
			   (vkGetImageMemoryRequirements
			    _device
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
			       (_device
				&info
				nullptr
				&imageMemory)
			       imageMemory
			       )
			     :throw t)
			   (vkBindImageMemory _device
					      image
					      imageMemory
					      0)
			   (return ("std::make_tuple"
				    image
				    imageMemory)))
			
			 )
		       (defun createTextureImage ()
			 (declare (values void))
			 "// uses command buffers "
			 (let ((texWidth 0)
			       (texHeight 0)
			       (texChannels 0)
			       (pixels
				(stbi_load
				 (string "chalet.jpg")
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
			     (throw ("std::runtime_error"
				     (string "failed to load texture image."))))
			   (let (((bracket stagingBuffer
					   stagingBufferMemory)
				  (createBuffer
				   imageSize
				   VK_BUFFER_USAGE_TRANSFER_SRC_BIT
				   (logior
				    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
				    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
				    )))
				 (data nullptr))
			     (declare (type void* data))
			     (vkMapMemory _device
					  stagingBufferMemory
					  0
					  imageSize
					  0
					  &data)
			     (memcpy data pixels
				     (static_cast<size_t>
				      imageSize))
			     (vkUnmapMemory _device
					    stagingBufferMemory)
			     (stbi_image_free pixels))

			   (let (((bracket image
					   imageMemory)
				  (createImage
				   texWidth
				   texHeight
				   VK_FORMAT_R8G8B8A8_UNORM
				   VK_IMAGE_TILING_OPTIMAL
				   (logior
				    VK_IMAGE_USAGE_TRANSFER_DST_BIT
				    VK_IMAGE_USAGE_SAMPLED_BIT)
				   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
				   )))
			     (setf _textureImage image
				   _textureImageMemory
				   imageMemory)
			     (transitionImageLayout
			      _textureImage
			      VK_FORMAT_R8G8B8A8_UNORM
			      VK_IMAGE_LAYOUT_UNDEFINED
			      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
			     (copyBufferToImage
			      stagingBuffer
			      _textureImage
			      (static_cast<uint32_t> texWidth)
			      (static_cast<uint32_t> texHeight))
			     (transitionImageLayout
			      _textureImage
			      VK_FORMAT_R8G8B8A8_UNORM
			      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
			      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)

			     (vkDestroyBuffer
			      _device stagingBuffer nullptr)
			     (vkFreeMemory _device
					   stagingBufferMemory
					   nullptr))))
		       (defun createTextureSampler ()
			 (declare (values void))
			 ,(vkcall
			   `(create
			     sampler
			     (:magFilter VK_FILTER_LINEAR
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
					 :minLod 0s0
					 :maxLod 0s0)
			     (_device
			      &info
			      nullptr
			      &_textureSampler)
			     _textureSampler
			     )
			   :throw t))
		       (defun createTextureImageView ()
			 (declare (values void))
			 (setf
			  _textureImageView
			  (createImageView _textureImage
					   VK_FORMAT_R8G8B8A8_UNORM
					   VK_IMAGE_ASPECT_COLOR_BIT))
			 ))
		     
		      #+surface
		      (do0
		       (defun createDescriptorSets ()
			 (declare (values void))
			 (let ((n (static_cast<uint32_t> (_swapChainImages.size)))
			       ((layouts n _descriptorSetLayout)))
			   (declare (type "std::vector<VkDescriptorSetLayout>"
					  (layouts n _descriptorSetLayout)))
			   (_descriptorSets.resize n)
			   ,(vkcall
			     `(allocate
			       descriptor-set
			       (:descriptorPool _descriptorPool
						:descriptorSetCount n
						:pSetLayouts (layouts.data))
			       (_device
				&info
				(_descriptorSets.data))
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
				 :buffer (aref _uniformBuffers i)
				 :offset 0
				 :range (sizeof UniformBufferObject)))
			    
			     ,(vk
			       `(VkWriteDescriptorSet
				 uboDescriptorWrite
				 :sType VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
				 :dstSet (aref _descriptorSets i)
				 :dstBinding 0
				 :dstArrayElement 0 ;; if multiple
				 ;; descriptors
				 ;; should be
				 ;; updated at once,
				 ;; start here
				 :descriptorType VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
				 :descriptorCount 1
				 :pBufferInfo &bufferInfo ;; buffer data
				 :pImageInfo nullptr ;; image data
				 :pTexelBufferView nullptr ;; buffer views
				 ))
			     ,(vk
			       `(VkDescriptorImageInfo
				 imageInfo
				 :imageLayout VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
				 :imageView _textureImageView
				 :sampler _textureSampler))
			     ,(vk
			       `(VkWriteDescriptorSet
				 samplerDescriptorWrite
				 :sType VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
				 :dstSet (aref _descriptorSets i)
				 :dstBinding 1
				 :dstArrayElement 0
				 :descriptorType VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
				 :descriptorCount 1
				 :pBufferInfo nullptr  ;; buffer data
				 :pImageInfo &imageInfo ;; image data
				 :pTexelBufferView nullptr ;; buffer views
				 ))
			     (let ((descriptorWrites (curly
						      uboDescriptorWrite
						      samplerDescriptorWrite)))
			       (declare (type
					 "std::array<VkWriteDescriptorSet,2>"
					 descriptorWrites))
			       (vkUpdateDescriptorSets
				_device
				(static_cast<uint32_t>
				 (descriptorWrites.size))
				(descriptorWrites.data)
				0
				nullptr ;; copy descriptor sets
				)))
			   ))
		       (defun createDescriptorPool ()
			 (declare (values void))
			 (let ((n (static_cast<uint32_t> (_swapChainImages.size)))
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
			   (let ((poolSizes (curly uboPoolSize
						   samplerPoolSize)))
			     (declare (type
				       "std::array<VkDescriptorPoolSize,2>"
				       poolSizes))
			     ,(vkcall
			       `(create
				 descriptor-pool
				 (:poolSizeCount (static_cast<uint32_t>
						  (poolSizes.size))
						 :pPoolSizes (poolSizes.data)
						 :maxSets n
						 ;; we could allow to free the sets
						 :flags 0)
				 (_device
				  &info
				  nullptr
				  &_descriptorPool)
				 _descriptorPool)
			       :throw t))))
		       (defun createUniformBuffers ()
			 (declare (values void))
			 (let ((bufferSize (sizeof UniformBufferObject))
			       (n (_swapChainImages.size)))
			   (_uniformBuffers.resize n)
			   (_uniformBuffersMemory.resize n)
			   (dotimes (i n)
			     (let (((bracket buf mem)
				    (createBuffer
				     bufferSize
				     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
				     (logior
				      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
				      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
				     )))
			       (setf (aref _uniformBuffers i)
				     buf
				     (aref _uniformBuffersMemory i)
				     mem)))))
		       (defun createDescriptorSetLayout ()
			 (declare (values void))
			 ,(vk
			   `(VkDescriptorSetLayoutBinding
			     samplerLayoutBinding
			     :binding 1
			     :descriptorCount 1
			     :descriptorType VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
			     :pImmutableSamplers nullptr
			     :stageFlags VK_SHADER_STAGE_FRAGMENT_BIT
			     ))
			 ,(vk
			   `(VkDescriptorSetLayoutBinding
			     uboLayoutBinding
			     :binding 0
			     :descriptorType VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
			     :descriptorCount 1
			     :stageFlags VK_SHADER_STAGE_VERTEX_BIT
			     :pImmutableSamplers nullptr))

 			 (let ((bindings (curly
					  uboLayoutBinding
					  samplerLayoutBinding)))
			   (declare (type
				     "std::array<VkDescriptorSetLayoutBinding, 2>"
				     bindings))
			   ,(vkcall
			     `(create
			       descriptor-set-layout
			       (:bindingCount (static_cast<uint32_t>
					       (bindings.size))
					      :pBindings (bindings.data))
			       (_device &info nullptr &_descriptorSetLayout)
			       _descriptorSetLayout
			       )
			     :throw t))
			
			 )
		      
		       (defun recreateSwapChain ()
			 (declare (values void))
			 (<< "std::cout"
			    (string "***** recreateSwapChain")
			    "std::endl")
			 (let ((width 0)
			       (height 0))
			   (declare (type int width height))
			   (while (or (== 0 width)
				      (== 0 height))
			     (glfwGetFramebufferSize _window
						     &width
						     &height)
			     (glfwWaitEvents)))
			
			 (vkDeviceWaitIdle _device) ;; wait for resources to be not in use anymore
			 (cleanupSwapChain)
			 (createSwapChain)
			 (createImageViews)
			 
			 (createRenderPass)
			 (createGraphicsPipeline)
			 (createDepthResources)
			 (createFramebuffers)
			 (createUniformBuffers)
			 (createDescriptorPool)
			 (createDescriptorSets)
			 (createCommandBuffers))
		       (do0
			"// shader stuff"
			(defun createSyncObjects ()
			  (declare (values void))
			  (_imageAvailableSemaphores.resize _MAX_FRAMES_IN_FLIGHT)
			  (_renderFinishedSemaphores.resize _MAX_FRAMES_IN_FLIGHT)
			  (_inFlightFences.resize _MAX_FRAMES_IN_FLIGHT)
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
					_device
					&semaphoreInfo
					nullptr
					(ref (aref _imageAvailableSemaphores i))))
			    ,(vkthrow `(vkCreateSemaphore
					_device
					&semaphoreInfo
					nullptr
					(ref (aref _renderFinishedSemaphores i))))
			    ,(vkthrow `(vkCreateFence
					_device
					&fenceInfo
					nullptr
					(ref (aref _inFlightFences i))))))
			(defun createCommandBuffers ()
			  (declare (values void))
			  (_commandBuffers.resize
			   (_swapChainFramebuffers.size))
			  ,(vkcall
			    `(allocate
			      command-buffer
			      (:commandPool _commandPool
					    :level VK_COMMAND_BUFFER_LEVEL_PRIMARY
					    :commandBufferCount (_commandBuffers.size))
			      (_device
			       &info
			       (_commandBuffers.data))
			      ;_commandBuffers
			      )
			    :throw t
			    :plural t)
			 
			 
			  (dotimes (i (_commandBuffers.size))
			    ,(vkcall
			      `(begin
				command-buffer
				(;; flags can select if exectution is
				 ;; once, inside single renderpass, or
				 ;; resubmittable
				 :flags 0 
				 :pInheritanceInfo nullptr)
				((aref _commandBuffers i) &info)
				(aref _commandBuffers i))
			      :throw t)
			   

			    ,(vk
			      `(VkClearValue
			       clearColor
			       :color (curly 0s0 0s0 0s0 1s0)))
			    ,(vk
			      `(VkClearValue
				clearDepth
				;; depth buffer far=1 by default
			       :depthStencil (curly 1s0 0)))
			    (let ((clearValues ("std::array<VkClearValue,2>" (curly clearColor clearDepth)))
				  )
			      
			      ,(vk
				`(VkRenderPassBeginInfo
				  renderPassInfo
				  :sType VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO
				  :renderPass _renderPass
				  :framebuffer (aref _swapChainFramebuffers i)
				  :renderArea.offset (curly  0 0)
				  :renderArea.extent _swapChainExtent
				  :clearValueCount (clearValues.size)
				  :pClearValues (clearValues.data)
				  )))
			    (vkCmdBeginRenderPass
			     (aref _commandBuffers i)
			     &renderPassInfo
			     ;; dont use secondary command buffers
			     VK_SUBPASS_CONTENTS_INLINE)
			    (vkCmdBindPipeline
			     (aref _commandBuffers i)
			     VK_PIPELINE_BIND_POINT_GRAPHICS _graphicsPipeline)
			    (let ((vertexBuffers[] (curly _vertexBuffer))
				  (offsets[] (curly 0))
				  )
			      (declare (type VkBuffer vertexBuffers[])
				       (type VkDeviceSize offsets[]))
			      (vkCmdBindVertexBuffers
			       (aref _commandBuffers i)
			       0
			       1
			       vertexBuffers
			       offsets
			       ))
			    (vkCmdBindIndexBuffer
			     (aref _commandBuffers i)
			     _indexBuffer
			     0
			     VK_INDEX_TYPE_UINT32)
			    (vkCmdBindDescriptorSets
			     (aref _commandBuffers i)
			     ;; descriptor could also be bound to compute
			     VK_PIPELINE_BIND_POINT_GRAPHICS
			     _pipelineLayout
			     0
			     1
			     (ref (aref _descriptorSets i))
			     0
			     nullptr)
			    ;; draw the triangle
			    (vkCmdDrawIndexed
			     (aref _commandBuffers i)
			     (static_cast<uint32_t> (g_indices.size)) ;; count
			     1 ;; no instance rendering
			     0 ;; offset to first index into buffer
			     0 ;; offset to add to index
			     0 ;; firstInstance
			     )
			    (vkCmdEndRenderPass
			     (aref _commandBuffers i))
			    ,(vkthrow `(vkEndCommandBuffer
					(aref _commandBuffers i)))
			    ))
			(defun createCommandPool ()
			  (declare (values void))
			  (let ((queueFamilyIndices (findQueueFamilies
						     _physicalDevice
						     _surface)))
			    ,(vkcall
			      `(create
				command-pool
				(;; cmds for drawing go to graphics queue
				 :queueFamilyIndex (queueFamilyIndices.graphicsFamily.value)
				 :flags 0)
				(_device
				 &info
				 nullptr
				 &_commandPool)
				_commandPool)
			      :throw t)))
			(defun createFramebuffers ()
			  (declare (values void))
			  (let ((n (_swapChainImageViews.size)))
			    (_swapChainFramebuffers.resize n)
			    (dotimes (i n)
			      ;; color attachment differs for each
			      ;; swap chain image, but depth image can
			      ;; be reused. at any time only one
			      ;; subpass is running
			      (let ((attachments ("std::array<VkImageView,2>"
						  (curly (aref _swapChainImageViews i)
							 _depthImageView))))
				
				,(vkcall
				  `(create
				    framebuffer
				    (:renderPass
				     _renderPass
				     :attachmentCount (static_cast<uint32_t>
						       (attachments.size))
				     :pAttachments (attachments.data)
				     :width _swapChainExtent.width
				     :height _swapChainExtent.height
				     :layers 1)
				    (_device
				     &info
				     nullptr
				     (ref
				      (aref _swapChainFramebuffers i)))
				    (aref _swapChainFramebuffers i))
				  :throw t)))))
			(defun createRenderPass ()
			  (declare (values void))
			  ,(vk
			    `(VkAttachmentDescription
			      colorAttachment
			      :format _swapChainImageFormat
			      :samples VK_SAMPLE_COUNT_1_BIT
			      :loadOp VK_ATTACHMENT_LOAD_OP_CLEAR
			      :storeOp VK_ATTACHMENT_STORE_OP_STORE
			      :stencilLoadOp VK_ATTACHMENT_LOAD_OP_DONT_CARE
			      :stencilStoreOp VK_ATTACHMENT_STORE_OP_DONT_CARE
			      :initialLayout VK_IMAGE_LAYOUT_UNDEFINED
			      ;; image to be presented in swap chain
			      :finalLayout VK_IMAGE_LAYOUT_PRESENT_SRC_KHR))
			  ,(vk
			    `(VkAttachmentDescription
			      depthAttachment
			      :format (findDepthFormat)
			      :samples VK_SAMPLE_COUNT_1_BIT
			      :loadOp VK_ATTACHMENT_LOAD_OP_CLEAR
			      :storeOp VK_ATTACHMENT_STORE_OP_DONT_CARE
			      :stencilLoadOp VK_ATTACHMENT_LOAD_OP_DONT_CARE
			      :stencilStoreOp VK_ATTACHMENT_STORE_OP_DONT_CARE
			      :initialLayout VK_IMAGE_LAYOUT_UNDEFINED
			      ;; image to be presented in swap chain
			      :finalLayout VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL))
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
			    `(VkSubpassDescription
			      subpass
			      :pipelineBindPoint VK_PIPELINE_BIND_POINT_GRAPHICS
			      :colorAttachmentCount 1
			      ;; frag shader references this as outColor
			      :pColorAttachments &colorAttachmentRef
			      :pDepthStencilAttachment &depthAttachmentRef
			      ))
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
			  (let ((attachments ("std::array<VkAttachmentDescription,2>"
					      (curly colorAttachment
						     depthAttachment))))
			    
			    ,(vkcall
				   `(create
				     render-pass
				     (:attachmentCount 
				      (static_cast<uint32_t>
				       (attachments.size))
				      :pAttachments (attachments.data)
				      :subpassCount 1
				      :pSubpasses &subpass
				      ;; wait with writing of the color attachments
				      :dependencyCount 1
				      :pDependencies &dependency)
				     (_device
				      &info
				      nullptr
				      &_renderPass)
				     _renderPass)
				   :throw t)))
			(defun createGraphicsPipeline ()
			  (declare (values void))
			  (let ((vertShaderModule (createShaderModule
						   (readFile (string "vert.spv"))))
				(fragShaderModule (createShaderModule
						   (readFile (string "frag.spv")))))
			    ,@(loop for e in `(frag vert) collect
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
				      :pSpecializationInfo nullptr)))
			    (let (("shaderStages[]"
				   (curly vertShaderStageInfo
					  fragShaderStageInfo))
				  (bindingDescription ("Vertex::getBindingDescription"))
				  (attributeDescriptions ("Vertex::getAttributeDescriptions")))
			      (declare (type VkPipelineShaderStageCreateInfo
					     "shaderStages[]"))
			      ,(vk
				`(VkPipelineVertexInputStateCreateInfo
				  vertexInputInfo
				  :sType VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO
				  :vertexBindingDescriptionCount 1
				  :pVertexBindingDescriptions &bindingDescription
				  :vertexAttributeDescriptionCount (static_cast<uint32_t> (attributeDescriptions.size))
				  :pVertexAttributeDescriptions (attributeDescriptions.data)))
			      ,(vk
				`(VkPipelineInputAssemblyStateCreateInfo
				  inputAssembly
				  :sType VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO
				  :topology VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
				  ;; this would allow to break up lines
				  ;; and strips with 0xfff or 0xffffff
				  :primitiveRestartEnable VK_FALSE))
			      ,(vk
				`(VkViewport
				  viewport
				  :x 0s0
				  :y 0s0
				  :width (* 1s0 _swapChainExtent.width)
				  :height (* 1s0 _swapChainExtent.height)
				  :minDepth 0s0
				  :maxDepth 1s0))
			      ,(vk
				`(VkRect2D
				  scissor
				  :offset (curly 0 0)
				  :extent _swapChainExtent))
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
				  :rasterizationSamples VK_SAMPLE_COUNT_1_BIT
				  :minSampleShading 1s0
				  :pSampleMask nullptr
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
				  :front (curly)
				  :back (curly)))
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
						  :pSetLayouts &_descriptorSetLayout
						  ;; another way of passing dynamic values to shaders
						  :pushConstantRangeCount 0
						  :pPushConstantRanges nullptr
						  )
				 (_device
				  &info
				  nullptr
				  &_pipelineLayout)
				 _pipelineLayout)
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
					     :pDynamicState nullptr
					     :layout _pipelineLayout
					     :renderPass _renderPass
					     :subpass 0
					     ;; similar pipelines can be derived
					     ;; from each other to speed up
					     ;; switching
					     :basePipelineHandle VK_NULL_HANDLE
					     :basePipelineIndex -1)
				(_device
				 VK_NULL_HANDLE ;; pipline cache
				 1
				 &info
				 nullptr
				 &_graphicsPipeline)
				_graphicsPipeline)
			      :plural t
			      :throw t)			   
			    (vkDestroyShaderModule _device
						   fragShaderModule
						   nullptr)
			    (vkDestroyShaderModule _device
						   vertShaderModule
						   nullptr)))
			(defun createShaderModule (code)
			  (declare (values VkShaderModule)
				   (type "const std::vector<char>&" code))
			  ;;std::vector<char> fullfills alignment requirements of uint32_t
			 
			  (let ((shaderModule))
			    (declare (type VkShaderModule shaderModule))
			    ,(vkcall
			      `(create
				shader-module
				(:codeSize (code.size)
					   :pCode ("reinterpret_cast<const uint32_t*>"
						   (code.data)))
				(_device
				 &info
				 nullptr
				 &shaderModule)
				shaderModule)
			      :throw t)
			    (return shaderModule))))
		      
		       (defun createSurface ()
			 (declare (values void))
			 "// initialize _surface member"
			 "// must be destroyed before the instance is destroyed"
			 ,(vkthrow `(glfwCreateWindowSurface
				     _instance _window
				     nullptr &_surface)))
		      
		       (defun createSwapChain ()
			 (declare (values void))
			 (let ((swapChainSupport
				(querySwapChainSupport _physicalDevice _surface))
			       (surfaceFormat
				(chooseSwapSurfaceFormat
				 swapChainSupport.formats))
			       (presentMode
				(chooseSwapPresentMode
				 swapChainSupport.presentModes))
			       (extent
				(chooseSwapExtent
				 swapChainSupport.capabilities
				 _window))
			       (imageCount
				(+ swapChainSupport.capabilities.minImageCount 1))
			       (indices (findQueueFamilies _physicalDevice _surface))
			       ((aref queueFamilyIndices) (curly
							   (indices.graphicsFamily.value)
							   (indices.presentFamily.value)))
			       ;; best performance mode:
			       (imageSharingMode VK_SHARING_MODE_EXCLUSIVE)
			       (queueFamilyIndexCount 0)
			       (pQueueFamilyIndices nullptr))
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
			       (:surface _surface
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
			       (_device
				&info
				nullptr
				&_swapChain
				)
			       _swapChain)
			     :throw t
			     :khr "KHR")			  
			   (do0
			    "// now get the images, note will be destroyed with the swap chain"
			    (vkGetSwapchainImagesKHR _device
						     _swapChain
						     &imageCount
						     nullptr)
			    (_swapChainImages.resize imageCount)
			    (vkGetSwapchainImagesKHR _device
						     _swapChain
						     &imageCount
						     (_swapChainImages.data))
			    (setf _swapChainImageFormat surfaceFormat.format
				  _swapChainExtent extent))))
		       (defun createImageView (image format aspectFlags)
			 (declare (values VkImageView)
				  (type VkImage image)
				  (type VkFormat format)
				  (type VkImageAspectFlags aspectFlags))
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
			       :subresourceRange.levelCount 1
			       :subresourceRange.baseArrayLayer 0
			       :subresourceRange.layerCount 1
			       )
			      (_device
			       &info
			       nullptr
			       &imageView)
			      imageView)
			    :throw t)
			  (return imageView)))
		       (defun createImageViews ()
			 (declare (values void))
			 (_swapChainImageViews.resize
			  (_swapChainImages.size))
			 (dotimes (i (_swapChainImages.size))
			   (setf
			    (aref _swapChainImageViews i)
			    (createImageView
			     (aref _swapChainImages i)
			     _swapChainImageFormat
			     VK_IMAGE_ASPECT_COLOR_BIT)))))
		      (defun createLogicalDevice ()
			(declare (values void))
			"// initialize members _device and _graphicsQueue"
			(let ((indices (findQueueFamilies _physicalDevice _surface))
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
			    (setf deviceFeatures.samplerAnisotropy VK_TRUE)
			    ,(vkcall
			      `(create
				device
				(:pQueueCreateInfos (queueCreateInfos.data)
						    :queueCreateInfoCount (static_cast<uint32_t>
									   (queueCreateInfos.size))
						    :pEnabledFeatures &deviceFeatures
						    :enabledExtensionCount #-surface 0
						    #+surface (static_cast<uint32_t> (_deviceExtensions.size))
						    #+surface :ppEnabledExtensionNames 
						    #+surface (_deviceExtensions.data)
						    :enabledLayerCount
						    #-nolog (static_cast<uint32_t> (_validationLayers.size))
						    #+nolog 0
						    :ppEnabledLayerNames
						    #+nolog nullptr
						    #-nolog (_validationLayers.data))
				(_physicalDevice
				 &info
				 nullptr
				 &_device)
				_physicalDevice)
			      :throw t)
			    (vkGetDeviceQueue _device (indices.graphicsFamily.value)
					      0 &_graphicsQueue)
			    #+surface
			    (vkGetDeviceQueue _device (indices.presentFamily.value)
					      0 &_presentQueue))))
		     
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
				     (when (isDeviceSuitable device _surface _deviceExtensions)
				       (setf _physicalDevice device)
				       break))
			    (when (== VK_NULL_HANDLE
				      _physicalDevice)
			      (throw ("std::runtime_error"
				      (string "failed to find a suitable gpu.")))))))
		      (defun mainLoop ()
			(declare (values void))
			(while (not (glfwWindowShouldClose _window))
			  (glfwPollEvents)
			  #+surface (drawFrame))
			(vkDeviceWaitIdle _device) ;; wait for gpu before cleanup
			)
		      #+surface
		      (defun drawFrame ()
			(declare (values void))
			(do0
			 (vkWaitForFences _device 1 (ref (aref _inFlightFences _currentFrame))  VK_TRUE UINT64_MAX)
			 )
			(let ((imageIndex 0)
			      (result (vkAcquireNextImageKHR
				       _device
				       _swapChain
				       UINT64_MAX ;; disable timeout for image 
				       (aref _imageAvailableSemaphores _currentFrame)
				       VK_NULL_HANDLE
				       &imageIndex)))
			  (declare (type uint32_t imageIndex))
			 
			  (when (== VK_ERROR_OUT_OF_DATE_KHR result)
			    (recreateSwapChain)
			    (return))
			  (unless (or (== VK_SUCCESS result)
				      (== VK_SUBOPTIMAL_KHR result))
			    (throw ("std::runtime_error"
				    (string "failed to acquire swap chain image."))))
			  (let ((waitSemaphores[] (curly (aref _imageAvailableSemaphores _currentFrame)))
				(signalSemaphores[] (curly (aref _renderFinishedSemaphores _currentFrame)))
				(waitStages[] (curly VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)))
			    (declare (type VkSemaphore waitSemaphores[]
					   signalSemaphores[])
				     (type VkPipelineStageFlags waitStages[]))
			    (updateUniformBuffer imageIndex)
			    ,(vk
			      `(VkSubmitInfo submitInfo
					     :sType VK_STRUCTURE_TYPE_SUBMIT_INFO
					     :waitSemaphoreCount 1
					     :pWaitSemaphores waitSemaphores
					     ;; pipeline has to wait for image before writing color buffer
					     :pWaitDstStageMask waitStages
					     :commandBufferCount 1
					     :pCommandBuffers (ref (aref _commandBuffers imageIndex))
					     :signalSemaphoreCount 1
					     :pSignalSemaphores signalSemaphores))
			    (vkResetFences _device 1 (ref (aref _inFlightFences _currentFrame)))
			    ,(vkthrow
			      `(vkQueueSubmit
				_graphicsQueue
				1
				&submitInfo
					;VK_NULL_HANDLE ;; fence
				(aref _inFlightFences _currentFrame)
				))
			   
			    ;; submit result for presentation
			    (let ((swapChains[] (curly _swapChain))
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
				  :pResults nullptr))
			      (progn
				(let ((result2 (vkQueuePresentKHR _presentQueue &presentInfo)))
				  (if (or (== VK_SUBOPTIMAL_KHR result2)
					  (== VK_ERROR_OUT_OF_DATE_KHR result2)
					  _framebufferResized)
				      (do0
				       (setf _framebufferResized false)
				       (recreateSwapChain))
				      (unless (== VK_SUCCESS result2)
					(throw ("std::runtime_error"
						(string "fialed to present swap chain image.")))))))
			     
					;(vkQueueWaitIdle _presentQueue) 
			      )
			   
			    ))
			(setf _currentFrame
			      (%
			       (+ 1 _currentFrame)
			       _MAX_FRAMES_IN_FLIGHT)))
		      (defun updateUniformBuffer (currentImage)
			(declare (type uint32_t currentImage)
				 (values void))
			(let ((startTime

			       ("std::chrono::high_resolution_clock::now"))
			      (currentTime
			       ("std::chrono::high_resolution_clock::now"))
			      (time (dot ("std::chrono::duration<float,std::chrono::seconds::period>" (- currentTime startTime))
					 (count)))
			      )
			  (declare (type "static auto" startTime)
				   (type auto currentTime)
				   (type float time)
				   )
			  ;; rotate model around z axis
			  (let ((zAxis ("glm::vec3" 0s0 0s0 1s0))
					;(identityMatrix ("glm::mat4" 1s0))
				(angularRate ("glm::radians" 90s0))
				(rotationAngle (* time angularRate)))
			    (declare (type "const auto" zAxis angularRate
					   identityMatrix))
			    ,(vk
			      `(UniformBufferObject
				ubo
				:model ("glm::rotate"
					("glm::mat4" 1s0)
					rotationAngle
					zAxis)
				;; look from above in 45 deg angle
				:view ("glm::lookAt"
				       ("glm::vec3" 2s0 2s0 2s0)
				       ("glm::vec3" 0s0 0s0 0s0)
				       zAxis
				       )
				;; use current extent for correct aspect
				:proj ("glm::perspective"
				       ("glm::radians" 45s0)
				       (/ _swapChainExtent.width
					  (* 1s0 _swapChainExtent.height))
				       .1s0
				       10s0)
				)))
			  ;; glm was designed for opengl and has
			  ;; inverted y clip coordinate
			  (setf (aref ubo.proj 1 1)
				(- (aref ubo.proj 1 1)))
			  (let ((data 0))
			    (declare (type void* data))
			    (vkMapMemory _device
					 (aref _uniformBuffersMemory
					       currentImage)
					 0
					 (sizeof ubo)
					 0
					 &data)
			    (memcpy data &ubo (sizeof ubo))
			    (vkUnmapMemory _device
					   (aref _uniformBuffersMemory
						 currentImage)))
			  ;; note: a more efficient way to pass
			  ;; frequently changing values to shaders are
			  ;; push constants
			  )
			)
		      (defun cleanupSwapChain ()
			(declare (values void))
			(<< "std::cout"
			    (string "***** cleanupSwapChain")
			    "std::endl")
			#+surface
			(do0
			 (do0
			  ;; depth
			  

			  ,(vkprint "cleanup depth"
				     `(_depthImageView
				       _depthImage
				       _depthImageMemory))
			  (vkDestroyImageView _device
					      _depthImageView
					      nullptr)
			  (vkDestroyImage _device
					  _depthImage
					  nullptr)
			  (vkFreeMemory _device
					_depthImageMemory
					nullptr)
			  )

			  
			 (foreach (b _swapChainFramebuffers)
				  ,(vkprint "framebuffer" `(b))
				  (vkDestroyFramebuffer _device b nullptr))
			 (vkFreeCommandBuffers _device
					       _commandPool
					       (static_cast<uint32_t>
						(_commandBuffers.size))
					       (_commandBuffers.data))
			 ,(vkprint "pipeline" `(_graphicsPipeline
						_pipelineLayout
						_renderPass))
			 (vkDestroyPipeline _device _graphicsPipeline nullptr)
			 (vkDestroyPipelineLayout
			  _device
			  _pipelineLayout
			  nullptr)
			 (vkDestroyRenderPass
			  _device
			  _renderPass
			  nullptr)
			 (foreach (view _swapChainImageViews)
				  ,(vkprint "image-view" `(view))
				  (vkDestroyImageView
				   _device
				   view
				   nullptr))
			 ,(vkprint "swapchain" `(_swapChain))
			 (vkDestroySwapchainKHR _device _swapChain nullptr)
			 ;; each swap chain image has a ubo
			 (dotimes (i (_swapChainImages.size))
			   ,(vkprint "ubo" `((aref _uniformBuffers i)
					     (aref _uniformBuffersMemory i)
					     ))
			   (vkDestroyBuffer _device
					    (aref _uniformBuffers i)
					    nullptr)
			   (vkFreeMemory _device
					 (aref _uniformBuffersMemory i)
					 nullptr))
			 ,(vkprint "descriptor-pool" `(_descriptorPool))
			 (vkDestroyDescriptorPool
			  _device
			  _descriptorPool
			  nullptr)
			 ))
		      (defun cleanup ()
			(declare (values void))
			
			#+surface
			(do0
			 (cleanupSwapChain)
			 (<< "std::cout"
			    (string "***** cleanup")
			    "std::endl")
			 (do0 ;; tex
			  ,(vkprint "tex"
				    `(_textureSampler
				      _textureImageView
				      _textureImage
				      _textureImageMemory
				      _descriptorSetLayout))
			  (vkDestroySampler _device
					    _textureSampler
					    nullptr)
			  (vkDestroyImageView _device
					      _textureImageView
					      nullptr)
			  (vkDestroyImage _device
					  _textureImage nullptr)
			  (vkFreeMemory _device
					_textureImageMemory nullptr))
			 (vkDestroyDescriptorSetLayout
			  _device
			  _descriptorSetLayout
			  nullptr)
			 ,(vkprint "buffers"
				   `(_vertexBuffer
				     _vertexBufferMemory
				     _indexBuffer
				     _indexBufferMemory
				     ))
			 (do0 (vkDestroyBuffer _device _vertexBuffer nullptr)
			      (vkFreeMemory _device _vertexBufferMemory nullptr))
			 (do0 (vkDestroyBuffer _device _indexBuffer nullptr)
			      (vkFreeMemory _device _indexBufferMemory nullptr))
			 (dotimes (i _MAX_FRAMES_IN_FLIGHT)
			   (do0
			    ,(vkprint "sync"
				      `((aref _renderFinishedSemaphores i)
					(aref _imageAvailableSemaphores i)
					(aref _inFlightFences i)))
			    (vkDestroySemaphore _device
						(aref _renderFinishedSemaphores i)
						nullptr)
			    (vkDestroySemaphore _device
						(aref _imageAvailableSemaphores i)
						nullptr)
			    (vkDestroyFence _device
					    (aref _inFlightFences i)
					    nullptr)))
			 ,(vkprint "cmd-pool"
				      `(_commandPool))
			 (vkDestroyCommandPool _device _commandPool nullptr)
			
			
			 )
			,(vkprint "rest"
				  `(_device _instance _window))
			(vkDestroyDevice _device nullptr)
			#+surface
			(vkDestroySurfaceKHR _instance _surface nullptr)
			(vkDestroyInstance _instance nullptr)
			(glfwDestroyWindow _window)
			(glfwTerminate)
			))))
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
    (write-source *code-file* code)
    (write-source *vertex-file* vertex-code)
    (write-source *frag-file* frag-code)
    (sb-ext:run-program "/usr/bin/glslangValidator" `("-V" ,(format nil "~a" *frag-file*)
							   "-o"
							   ,(format nil "~a/frag.spv"
								    (directory-namestring *vertex-file*))))
    (sb-ext:run-program "/usr/bin/glslangValidator" `("-V" ,(format nil "~a" *vertex-file*)
							   "-o"
							   ,(format nil "~a/vert.spv"
								    (directory-namestring *vertex-file*))))))
 

