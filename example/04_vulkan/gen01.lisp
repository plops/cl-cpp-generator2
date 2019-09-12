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
  (defparameter *vertex-file* (asdf:system-relative-pathname 'cl-cpp-generator2 "example/04_vulkan/source/run_01_base.vert"))
  (defparameter *frag-file* (asdf:system-relative-pathname 'cl-cpp-generator2 "example/04_vulkan/source/run_01_base.frag"))
  (let* ((vertex-code
	  `(do0
	    "#version 450"
	    " "
	    "layout(location = 0) out vec3 fragColor;"
	    
	    (let (("positions[]" ("vec2[]"
				(vec2 .0 -.5)
				(vec2 .5 .5)
				(vec2 -.5 .5)))
		  ("colors[]" ("vec3[]"
			     (vec3 1.  .0 .0)
			     (vec3 .0 1.  .0 )
			     (vec3 .0 .0 1.))))
	      (declare (type vec2 "positions[]")
		       (type vec3 "colors[]"))
	      
	      (defun main ()
		(declare (values void))
		(setf gl_Position
		      (vec4 (aref positions gl_VertexIndex)
			    .0
			    1.)
		      fragColor
		      (aref colors gl_VertexIndex)))
	      "// vertex shader end ")))
	 (frag-code
	  `(do0
	    "#version 450"
	    "#extension GL_ARB_separate_shader_objects : enable"
	    " "
	    "layout(location = 0) out vec4 outColor;"
	    "layout(location = 0) in vec3 fragColor;"
	    
	    (defun main ()
	      (declare (values void))
	      (setf outColor
		    (vec4 fragColor
			  1.)))))
	 (code
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
		     <set>)
	    #+surface
	    (include
	     ;; UINT32_MAX:
	     <cstdint> 
	     <algorithm>
	     )

	    (do0
	     "// code to load binary shader from file"
	     (include <fstream>)
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
	    (defstruct0 QueueFamilyIndices 
		(graphicsFamily "std::optional<uint32_t>")
	      #+surface (presentFamily "std::optional<uint32_t>")
	      #+nil(defun isComplete ()
		     (declare (values bool))
		     (return (graphicsFamily.has_value))))

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
	     (defun chooseSwapExtent (capabilities)
	       (declare (values VkExtent2D)
			(type "const VkSurfaceCapabilitiesKHR&"
			      capabilities))
	       (if (!= UINT32_MAX capabilities.currentExtent.width)
		   (do0
		    (return capabilities.currentExtent))
		   (do0
		    (let ((actualExtent (curly 800 600))
			  )
		      (declare (type VkExtent2D actualExtent))

		      ,@(loop for e in `(width height) collect
			     `(setf (dot actualExtent ,e)
				    ("std::max" (dot capabilities.minImageExtent ,e)
						("std::min"
						 (dot capabilities.maxImageExtent ,e)
						 (dot actualExtent ,e)))))
		      
		      (return actualExtent))))))
	    
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
				(_pipelineLayout
				 )
				(_graphicsPipeline)
				(_swapChainFramebuffers)
				(_commandPool)
				(_commandBuffers)
				(_imageAvailableSemaphores)
				(_renderFinishedSemaphores)
				(_inFlightFences)
				(_MAX_FRAMES_IN_FLIGHT 2)
				(_currentFrame 0))
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
				(type "std::vector<VkFence>" _inFlightFences))
		
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
		       (createLogicalDevice)
		       #+surface
		       (do0
			(createSwapChain)
			(createImageViews)
			(createRenderPass)
			(createGraphicsPipeline)
			(createFramebuffers)
			(createCommandPool)
			(createCommandBuffers)
			(createSyncObjects)))
		     
		     #+surface
		     (do0
		      
		      (defun recreateSwapChain ()
			(declare (values void))
			(vkDeviceWaitIdle _device) ;; wait for resources to be not in use anymore
			(createSwapChain)
			(createImageViews)
			(createRenderPass)
			(createGraphicsPipeline)
			(createFramebuffers)
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
			  (unless (and
				   (== VK_SUCCESS
				       (vkCreateSemaphore
					_device
					&semaphoreInfo
					nullptr
					(ref (aref _imageAvailableSemaphores i))))
				   (== VK_SUCCESS
				       (vkCreateSemaphore
					_device
					&semaphoreInfo
					nullptr
					(ref (aref _renderFinishedSemaphores i))))
				   (== VK_SUCCESS
				       (vkCreateFence
					_device
					&fenceInfo
					nullptr
					(ref (aref _inFlightFences i)))))
			    (throw ("std::runtime_error"
				    (string "failed to create sync objects."))))))
		       (defun createCommandBuffers ()
			 (declare (values void))
			 (_commandBuffers.resize
			  (_swapChainFramebuffers.size))
			 ,(vk
			   `(VkCommandBufferAllocateInfo
			     allocInfo
			     :sType VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO
			     :commandPool _commandPool
			     :level VK_COMMAND_BUFFER_LEVEL_PRIMARY
			     :commandBufferCount (_commandBuffers.size)))
			 (unless (== VK_SUCCESS
				     (vkAllocateCommandBuffers _device
							       &allocInfo
							       (_commandBuffers.data)))
			   (throw ("std::runtime_error"
				   (string "failed to allocate command buffers."))))
			 (dotimes (i (_commandBuffers.size))
			   ,(vk
			     `(VkCommandBufferBeginInfo
			       beginInfo
			       :sType VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
			       ;; flags can select if exectution is
			       ;; once, inside single renderpass, or
			       ;; resubmittable
			       :flags 0 
			       :pInheritanceInfo nullptr
			       ))
			   (unless (== VK_SUCCESS
				       (vkBeginCommandBuffer
					(aref _commandBuffers i)
					&beginInfo))
			     (throw ("std::runtime_error"
				     (string "failed to begin recording command buffer."))))
			   (let ((clearColor (curly 0s0 0s0 0s0 1s0)))
			     (declare (type VkClearValue clearColor))
			    ,(vk
			      `(VkRenderPassBeginInfo
				renderPassInfo
				:sType VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO
				:renderPass _renderPass
				:framebuffer (aref _swapChainFramebuffers i)
				:renderArea.offset (curly  0 0)
				:renderArea.extent _swapChainExtent
				:clearValueCount 1
				:pClearValues &clearColor
				)))
			   (vkCmdBeginRenderPass
			    (aref _commandBuffers i)
			    &renderPassInfo
			    ;; dont use secondary command buffers
			    VK_SUBPASS_CONTENTS_INLINE)
			   (vkCmdBindPipeline
			    (aref _commandBuffers i)
			    VK_PIPELINE_BIND_POINT_GRAPHICS _graphicsPipeline)
			   ;; draw the triangle
			   (vkCmdDraw (aref _commandBuffers i)
				      3 ;; vertex count
				      1 ;; no instance rendering
 				      0 ;; offset to first vertex
				      0 ;; firstInstance
				      )
			   (vkCmdEndRenderPass
			    (aref _commandBuffers i))
			   (unless (== VK_SUCCESS
				       (vkEndCommandBuffer
					(aref _commandBuffers i)))
			     (throw ("std::runtime_error"
				     (string "failed to record command buffer."))))
			   ))
		       (defun createCommandPool ()
			 (declare (values void))
			 (let ((queueFamilyIndices (findQueueFamilies
						    _physicalDevice)))
			  ,(vk
			    `(VkCommandPoolCreateInfo
			      poolInfo
			      :sType VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO
			      ;; cmds for drawing go to graphics queue
			      :queueFamilyIndex (queueFamilyIndices.graphicsFamily.value)
			      :flags 0)))
			 (unless (== VK_SUCCESS
				     (vkCreateCommandPool _device
							  &poolInfo
							  nullptr
							  &_commandPool))
			   (throw ("std::runtime_error"
				   (string "failed to create command pool."))))
			 )
		       (defun createFramebuffers ()
			 (declare (values void))
			 (_swapChainFramebuffers.resize
			  (_swapChainImageViews.size))
			 (dotimes (i (_swapChainImageViews.size))
			   (let (("attachments[]" (curly (aref _swapChainImageViews i))))
			     (declare (type VkImageView "attachments[]"))
			     ,(vk
			       `(VkFramebufferCreateInfo
				 framebufferInfo
				 :sType VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO
				 :renderPass _renderPass
				 :attachmentCount 1
				 :pAttachments attachments
				 :width _swapChainExtent.width
				 :height _swapChainExtent.height
				 :layers 1))
			     (unless (== VK_SUCCESS
					 (vkCreateFramebuffer
					  _device
					  &framebufferInfo
					  nullptr
					  (ref
					   (aref _swapChainFramebuffers i))))
			       (throw ("std::runtime_error"
				       (string "failed to create framebuffer.")))))))
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
			     :finalLayout VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
			     ))
			 ,(vk
			   `(VkAttachmentReference
			     colorAttachmentRef
			     ;; we only have one attachment description
			     :attachment 0
			     ;; choose best layout for use case color buffer
			     :layout VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL))
			 ,(vk
			   `(VkSubpassDescription
			     subpass
			     :pipelineBindPoint VK_PIPELINE_BIND_POINT_GRAPHICS
			     :colorAttachmentCount 1
			     ;; frag shader references this as outColor
			     :pColorAttachments &colorAttachmentRef))
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
			 ,(vk
			   `(VkRenderPassCreateInfo
			     renderPassInfo
			     :sType VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO
			     :attachmentCount 1
			     :pAttachments &colorAttachment
			     :subpassCount 1
			     :pSubpasses &subpass
			     ;; wait with writing of the color attachment
			     :dependencyCount 1
			     :pDependencies &dependency))
			 (unless (== VK_SUCCESS
				     (vkCreateRenderPass
				      _device
				      &renderPassInfo
				      nullptr
				      &_renderPass))
			   (throw ("std::runtime_error"
				   (string "failed to create render pass."))))
			 )
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
					 fragShaderStageInfo)))
			     (declare (type VkPipelineShaderStageCreateInfo
					    "shaderStages[]"))
			     ,(vk
			       `(VkPipelineVertexInputStateCreateInfo
				 vertexInputInfo
				 :sType VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO
				 :vertexBindingDescriptionCount 0
				 :pVertexBindingDescriptions nullptr
				 :vertexAttributeDescriptionCount 0
				 :pVertexAttributeDescriptions nullptr))
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
				 :frontFace VK_FRONT_FACE_CLOCKWISE
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
			    ,(vk
			      `(VkPipelineLayoutCreateInfo
				pipelineLayoutInfo
				:sType VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
				:setLayoutCount 0
				:pSetLayouts nullptr
				;; another way of passing dynamic values to shaders
				:pushConstantRangeCount 0
				:pPushConstantRanges nullptr))
			    (unless (== VK_SUCCESS
					(vkCreatePipelineLayout _device
								&pipelineLayoutInfo
								nullptr
								&_pipelineLayout))
			      (throw ("std::runtime_error"
				      (string "failed to create pipeline layout.")))))

			   ,(vk
			     `(VkGraphicsPipelineCreateInfo
			       pipelineInfo
			       :sType VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO
			       :stageCount 2
			       :pStages shaderStages
			       :pVertexInputState &vertexInputInfo
			       :pInputAssemblyState &inputAssembly
			       :pViewportState &viewPortState
			       :pRasterizationState &rasterizer
			       :pMultisampleState &multisampling
			       :pDepthStencilState nullptr
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
			       :basePipelineIndex -1))
			   (unless (== VK_SUCCESS
				       (vkCreateGraphicsPipelines
					_device
					VK_NULL_HANDLE ;; pipline cache
					1
					&pipelineInfo
					nullptr
					&_graphicsPipeline))
			     (throw ("std::runtime_error"
				     (string "failed to create graphics pipeline."))))
			   
			   (vkDestroyShaderModule _device
						  fragShaderModule
						  nullptr)
			   (vkDestroyShaderModule _device
						  vertShaderModule
						  nullptr))
			 )
		       (defun createShaderModule (code)
			 (declare (values VkShaderModule)
				  (type "const std::vector<char>&" code))
			 ;;std::vector<char> fullfills alignment requirements of uint32_t
			 ,(vk
			   `(VkShaderModuleCreateInfo
			     createInfo
			     :sType VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
			     :codeSize (code.size)
			     :pCode ("reinterpret_cast<const uint32_t*>"
				     (code.data))))
			 (let ((shaderModule))
			   (declare (type VkShaderModule shaderModule))
			   (unless (== VK_SUCCESS
				       (vkCreateShaderModule _device
							     &createInfo
							     nullptr
							     &shaderModule))
			     (throw ("std::runtime_error"
				     (string "failed to create shader module."))))
			   (return shaderModule))))
		      
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
				swapChainSupport.capabilities))
			      (imageCount
			       (+ swapChainSupport.capabilities.minImageCount 1))
			      (indices (findQueueFamilies _physicalDevice))
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
			  ,(vk `(VkSwapchainCreateInfoKHR
				 createInfo
				 :sType VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR
				 :surface _surface
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
				 ))
			  (unless (== VK_SUCCESS
				      (vkCreateSwapchainKHR
				       _device
				       &createInfo
				       nullptr
				       &_swapChain))
			    (throw ("std::runtime_error" (string "failed to create swap chain")))
			    )
			  
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
				 _swapChainExtent extent)
			   )))
		      (defun createImageViews ()
			(declare (values void))
			(_swapChainImageViews.resize
			 (_swapChainImages.size))
			(dotimes (i (_swapChainImages.size))
			  ,(vk
			    `(VkImageViewCreateInfo
			      createInfo
			      :sType VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO
			      :image (aref _swapChainImages i)
			      :viewType VK_IMAGE_VIEW_TYPE_2D
			      :format _swapChainImageFormat
			      ;; here we could move color channels around
			      :components.r VK_COMPONENT_SWIZZLE_IDENTITY
			      :components.g VK_COMPONENT_SWIZZLE_IDENTITY
			      :components.b VK_COMPONENT_SWIZZLE_IDENTITY
			      :components.a VK_COMPONENT_SWIZZLE_IDENTITY
			      ;; color targets without mipmapping or
			      ;; multi layer (stereo)
			      :subresourceRange.aspectMask VK_IMAGE_ASPECT_COLOR_BIT
			      :subresourceRange.baseMipLevel 0
			      :subresourceRange.levelCount 1
			      :subresourceRange.baseArrayLayer 0
			      :subresourceRange.layerCount 1
			      ))
			  (unless
			      (== VK_SUCCESS
				  (vkCreateImageView
				   _device
				   &createInfo
				   nullptr
				   (ref (aref _swapChainImageViews i))))
			    (throw ("std::runtime_error"
				    (string "failed to create image view.")))))))
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
				  :enabledExtensionCount #-surface 0
				  #+surface (static_cast<uint32_t> (_deviceExtensions.size))
				  #+surface :ppEnabledExtensionNames 
				  #+surface (_deviceExtensions.data)
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
		     #+surface
		     (defun checkDeviceExtensionSupport (device)
		       (declare (values bool)
				(type VkPhysicalDevice device))
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
		     (defun isDeviceSuitable ( device)
		       (declare (values bool)
				(type VkPhysicalDevice device))
		       #+surface
		       (let ((extensionsSupported (checkDeviceExtensionSupport device))
			     (swapChainAdequate false))
			 (declare (type bool swapChainAdequate))
			 (when extensionsSupported
			   (let ((swapChainSupport (querySwapChainSupport device _surface)))
			     (setf swapChainAdequate
				   (and (not (swapChainSupport.formats.empty))
					(not (swapChainSupport.presentModes.empty)))))))
		       (let ((indices (findQueueFamilies device)))
			 (declare (type QueueFamilyIndices indices))
			 (return (and (indices.graphicsFamily.has_value)
				      #+surface (and (indices.presentFamily.has_value)
						     extensionsSupported
						     swapChainAdequate)))
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
			 (glfwPollEvents)
			 #+surface (drawFrame))
		       
		       (vkDeviceWaitIdle _device) ;; wait for gpu before cleanup
		       )
		     #+surface
		     (defun drawFrame ()
		       (declare (values void))
		       ;; wait for previous frame, FIXME _currentFrame-1?
		       #+nil (let ((oldFrame (- _currentFrame 1)))
			 (declare (type int oldFrame))
			 (if (== -1 oldFrame)
			     (setf oldFrame (- _MAX_FRAMES_IN_FLIGHT 1)))
			 (do0
			  (vkWaitForFences _device 1 (ref (aref _inFlightFences oldFrame))  VK_TRUE UINT64_MAX)
			  (vkResetFences _device 1 (ref (aref _inFlightFences oldFrame)))))
		       (do0
			  (vkWaitForFences _device 1 (ref (aref _inFlightFences _currentFrame))  VK_TRUE UINT64_MAX)
			  (vkResetFences _device 1 (ref (aref _inFlightFences _currentFrame))))
		       (let ((imageIndex 0))
			 (declare (type uint32_t imageIndex))
			 (vkAcquireNextImageKHR
			  _device
			  _swapChain
			  UINT64_MAX ;; disable timeout for image 
			  (aref _imageAvailableSemaphores _currentFrame)
			  VK_NULL_HANDLE
			  &imageIndex)
			 
			 (let ((waitSemaphores[] (curly (aref _imageAvailableSemaphores _currentFrame)))
			       (signalSemaphores[] (curly (aref _renderFinishedSemaphores _currentFrame)))
			       (waitStages[] (curly VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)))
			   (declare (type VkSemaphore waitSemaphores[]
					  signalSemaphores[])
				    (type VkPipelineStageFlags waitStages[]))
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
			   (unless (== VK_SUCCESS
				       (vkQueueSubmit
					_graphicsQueue
					1
					&submitInfo
					;VK_NULL_HANDLE ;; fence
					(aref _inFlightFences _currentFrame)
					)
				       )
			     (throw ("std::runtime_error"
				     (string "failed to submit draw command buffer."))))
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
			    (vkQueuePresentKHR _presentQueue &presentInfo)
			    
			    ;(vkQueueWaitIdle _presentQueue) 
			    )
			   
			   ))
		       (setf _currentFrame
			     (%
			      (+ 1 _currentFrame)
			      _MAX_FRAMES_IN_FLIGHT)))
		     #+nil
		     (defun cleanup ()
		       (declare (values void))
		       
		       #+surface
		       (do0
			
			(dotimes (i _MAX_FRAMES_IN_FLIGHT)
			  (do0
			   (vkDestroySemaphore _device
					       (aref _renderFinishedSemaphores i)
					       nullptr)
			   (vkDestroySemaphore _device
					       (aref _imageAvailableSemaphores i)
					       nullptr)
			   (vkDestroyFence _device
					   (aref _inFlightFences i)
					   nullptr)))
			(vkDestroyCommandPool _device _commandPool nullptr)
			(foreach (b _swapChainFramebuffers)
				 (vkDestroyFramebuffer _device b nullptr))
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
				 (vkDestroyImageView
				  _device
				  view
				  nullptr))
			(vkDestroySwapchainKHR _device _swapChain nullptr))
		       (vkDestroyDevice _device nullptr)
		       #+surface
		       (vkDestroySurfaceKHR _instance _surface nullptr)
		       (vkDestroyInstance _instance nullptr)
		       (glfwDestroyWindow _window)
		       (glfwTerminate)
		       )
		     (defun cleanupSwapChain ()
			(declare (values void))
			
			#+surface
			(do0
			 
			 (foreach (b _swapChainFramebuffers)
				  (vkDestroyFramebuffer _device b nullptr))
			 (vkFreeCommandBuffers _device
					       _commandPool
					       (static_cast<uint32_t>
						(_commandBuffers.size))
					       (_commandBuffers.data))
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
				  (vkDestroyImageView
				   _device
				   view
				   nullptr))
			 (vkDestroySwapchainKHR _device _swapChain nullptr)))
		     (defun cleanup ()
		       (declare (values void))
		       
		       #+surface
		       (do0
			(cleanupSwapChain)
			(dotimes (i _MAX_FRAMES_IN_FLIGHT)
			  (do0
			   (vkDestroySemaphore _device
					       (aref _renderFinishedSemaphores i)
					       nullptr)
			   (vkDestroySemaphore _device
					       (aref _imageAvailableSemaphores i)
					       nullptr)
			   (vkDestroyFence _device
					   (aref _inFlightFences i)
					   nullptr)))
			(vkDestroyCommandPool _device _commandPool nullptr)
			
			
			)
		       (vkDestroyDevice _device nullptr)
		       #+surface
		       (vkDestroySurfaceKHR _instance _surface nullptr)
		       (vkDestroyInstance _instance nullptr)
		       (glfwDestroyWindow _window)
		       (glfwTerminate)
		       )))
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
 

