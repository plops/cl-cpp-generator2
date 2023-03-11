(eval-when (:compile-toplevel :execute :load-toplevel)
	   (ql:quickload "cl-cpp-generator2")
	   (ql:quickload "cl-ppcre")
	   (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *source-dir* #P"example/119_vulkan_hpp_comp/source00/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  
  ;(load "util.lisp")
  
  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"main.cpp"
		     *source-dir*))
   `(do0

     (comments "sudo pacman -S vulkan-headers vulkan-devel")
     (include<> vulkan/vulkan.hpp
		shaderc/shaderc.hpp
		vector)
     
     "using namespace vk;"
     
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       "(void) argc;"
       "(void) argv;"
       (let ((info (ApplicationInfo (string "hello world")
				    0
				    nullptr
				    0
				    VK_API_VERSION_1_3))
	     (instance (createInstanceUnique (InstanceCreateInfo
					      "{}" &info)))
	     (physicalDevice (-> instance
				 (aref (enumeratePhysicalDevices) 0)))
	     (qProps (physicalDevice.getQueueFamilyProperties))
	     (family 0))
	 (for-range (qProp qProps)
		    (when (and qProp.queueFlags
			       QueueFlagBits--eCompute)
		      break)
		    (incf family))
	 (let ((priority (curly 1s0)))
	   (declare (type (array "constexpr float" 1) priority))
	   (let ((qInfo (DeviceQueueCreateInfo "{}"
					       family
					       1
					       priority))
		 (device (physicalDevice.createDeviceUnique
			  (DeviceCreateInfo "{}"
					    qInfo)))
		 (printShader (string-r ,(emit-c :code `(do0
							 "#version 460"
							 "#extension GL_EXT_debug_printf : require"
							 (defun main ()
							   (debugPrintfEXT
							    (string "hello from thread %d\\n")
							    gl_GlobalInvocationID.x))))))
		 )
	     (declare (type "const std::string" printShader))
	     (let ((compiled (dot (shaderc--Compiler)
				  (CompileGlslToSpv printShader
						    shaderc_compute_shader
						    (string "hello_world.comp"))))
		   (spirv (std--vector<uint32_t> (compiled.cbegin)
						 (compiled.cend)))
		   (shaderModule (device->createShaderModuleUnique
				  (ShaderModuleCreateInfo "{}" spirv)))
		   (stageInfo (PipelineShaderStageCreateInfo
			       "{}" ShaderStageFlagBits--eCompute
			       *shaderModule
			       (string "main")))
		   (pipelineLayout (device->createPipelineLayoutUnique
				   (PipelineLayoutCreateInfo)))
		   (pipelineInfo (ComputePipelineCreateInfo (curly)
							    stageInfo
							    *pipelineLayout))
		   ((bracket status pipeline)
		     (device->createComputePipelineUnique
		      (deref (device->createPipelineCacheUnique (curly)))
		      pipelineInfo)))))))
       (return 0))))
  )


  
