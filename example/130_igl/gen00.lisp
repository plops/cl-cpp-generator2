(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (progn
    (defparameter *source-dir* #P"example/130_igl/source00/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*)))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")


  (defun share (name)
    (format nil "std::shared_ptr<~a>" name))
  (defun uniq (name)
    (format nil "std::unique_ptr<~a>" name))
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     ,@(loop for e in `((GLFW_INCLUDE_NONE)
			(GLFW_EXPOSE_NATIVE_X11)
			(GLFW_EXPOSE_NATIVE_GLX)
			(USE_OPENGL_BACKEND 1)
			(ENABLE_MULTIPLE_COLOR_ATTACHMENTS 0)
			(IGL_FORMAT "fmt::format"))
	     collect
	     (destructuring-bind (name &optional (value "")) e
	       (format nil 
		       "#define ~a ~a" name value)))
     
     (include<>
      GLFW/glfw3.h
      GLFW/glfw3native.h
      cassert
      ;regex
      iostream
      igl/IGL.h
      igl/opengl/glx/Context.h
      igl/opengl/glx/Device.h
      igl/opengl/glx/HWDevice.h
      igl/opengl/glx/PlatformDevice.h
      )

     (include<> fmt/core.h)
     
     "using namespace igl;"
     
     "static const uint32_t kNumColorAttachments = 1;"


     (setf "std::string codeVS"
	   (string-r ,(emit-c
		       :code
		       `(do0 "#version 460"
			     "layout (location=0) out vec3 color;"
			     (let ((pos ("vec2[3]" (vec2 -.6 -.4)
						   (vec2 .6 -.4)
						   (vec2 .0 .6)))
				   (col ("vec3[3]" (vec3 1.0 .0 .0)
						   (vec3 .0 1.0 .0)
						   (vec3 .0 .0 1.0))))
			       (declare (type (array "const vec2" 3) pos)
					(type (array "const vec3" 3) col))
			       (defun main ()
				 (setf gl_Position (vec4 (aref pos gl_VertexID ;Index
							       )
							 .0
							 1.)
				       color (aref col gl_VertexID ;Index
						   ))))
			     )
		       :omit-redundant-parentheses t
		       )))

     (setf "std::string codeFS"
	   (string-r ,(emit-c
		       :code
		       `(do0 "#version 460"
			     "layout (location=0) in vec3 color;"
			     "layout (location=0) out vec4 out_FragColor;"
			     (defun main ()
			       (setf out_FragColor (vec4 color 1.)))
			     )
		       :omit-redundant-parentheses t)))

     ,@(loop for e in `((window GLFWwindow* nullptr)
			(width int 800)
			(height int 600)
			(device ,(uniq `IDevice))
			(commandQueue ,(share 'ICommandQueue))
			(renderPass RenderPassDesc)
			(framebuffer ,(share 'IFramebuffer))
			(renderPipelineState_Triangle ,(share 'IRenderPipelineState)))
	     collect
	     (destructuring-bind (name type &optional value)
		 e
	       (format nil "~a ~a_~@[ = ~a~];" type name value)))

     (defun initWindow (outWindow)
       (declare (type GLFWwindow** outWindow)
		(values "static bool"))
       (unless (glfwInit)
	 (return false))

       ,@(loop for e in `((context-version-major 4)
			  (context-version-minor 1)
			  (opengl-profile GLFW_OPENGL_COMPAT_PROFILE)
			  (visible true)
			  (doublebuffer true)
			  (srgb-capable true)
			  (client-api GLFW_OPENGL_API)
			  (resizable GLFW_TRUE))
	       collect
	       (destructuring-bind (name value) e
		 `(glfwWindowHint ,(cl-change-case:constant-case (format nil "glfw-~a" name))
				  ,value)))
       (let ((*window (glfwCreateWindow width_ height_ (string "OpenGL Triangle") nullptr nullptr)))
	 (unless window
	   (glfwTerminate)
	   (return false))

	 (glfwSetErrorCallback (lambda (err desc)
				 (declare (type int err)
					  (type "const char*" desc))
				 ,(lprint :msg "GLFW Error" :vars `(err desc))))
	 (glfwSetKeyCallback window
			     (lambda (window key a action b)
			       (declare (type GLFWwindow* window)
					(type int key a action b))
			       (when (and (== key GLFW_KEY_ESCAPE)
					  (== action GLFW_PRESS))
				 (glfwSetWindowShouldClose window GLFW_TRUE))))
	 (glfwSetWindowSizeCallback window
				   (lambda (window width height)
			       (declare (type GLFWwindow* window)
					(type int width height))
				     ,(lprint :msg "window resized" :vars `(width height))
				     (setf width_ width
					   height_ height)))
	 (glfwGetWindowSize window &width_ &height_)
	 (when outWindow
	   (setf *outWindow window))
	 (return true))
       )

     (defun initIGL ()
       (let ((ctx ("std::make_unique<igl::opengl::glx::Context>"
		   nullptr
		   (glfwGetX11Display)
		   ("reinterpret_cast<igl::opengl::glx::GLXDrawable>" (glfwGetX11Window window_))
		   ("reinterpret_cast<igl::opengl::glx::GLXContext>" (glfwGetGLXContext window_)))))
	 (setf device_ ("std::make_unique<igl::opengl::glx::Device>" (std--move ctx)))
	 (IGL_ASSERT device_)

	 "CommandQueueDesc desc{CommandQueueType::Graphics};"
	 (setf commandQueue_ (-> device_
				 (createCommandQueue desc nullptr)))
	 (dotimes (i kNumColorAttachments)
	   (when (& i (hex 1))
	     continue)
	   (setf (dot renderPass_ (aref colorAttachments i))
		 "igl::RenderPassDesc::ColorAttachmentDesc{}")
	   ,@(loop for (name value) in `((loadAction "LoadAction::Clear")
					 (storeAction "StoreAction::Store")
					 (clearColor "{1.f,1.f,1.f,1.f}")
					 )
		   collect
		   `(setf (dot renderPass_ (aref colorAttachments i) ,name)
			  ,value)))
	 (setf (dot renderPass_
		    depthAttachment
		    loadAction)
	       "LoadAction::DontCare")

	 ))
     (defun createRenderPipeline ()
       (when renderPipelineState_Triangle_
	 return)
       (IGL_ASSERT framebuffer_)
       (let ((desc (RenderPipelineDesc)))
	 (dot desc
	      targetDesc
	      colorAttachments
	      (resize kNumColorAttachments))
	 (dotimes (i kNumColorAttachments)
	   (when (-> framebuffer_
		     (getColorAttachment i))
	     (setf (dot desc
			targetDesc
			(aref colorAttachments i)
			textureFormat)
		   (-> framebuffer_
		       (getColorAttachment i)
		       (getFormat)))))
	 (when (-> framebuffer_
		   (getDepthAttachment))
	   (setf (dot desc
		      targetDesc
		      depthAttachmentFormat
		      )
		 (-> framebuffer_
		     (getDepthAttachment)
		     (getFormat))))

	 (setf desc.shaderStages (ShaderStagesCreator--fromModuleStringInput
				  *device_
				  (codeVS.c_str)
				  (string "main")
				  (string "")
				  (codeFS.c_str)
				  (string "main")
				  (string "")
				  nullptr)
	       renderPipelineState_Triangle_ (-> device_
						 (createRenderPipeline desc nullptr)))
	 (IGL_ASSERT renderPipelineState_Triangle_)))

     (defun getNativeDrawable ()
       (declare (static)
		(values ,(share 'ITexture)))
       (let ((ret (Result))
	     (drawable (,(share 'ITexture)))
	     (platformDevice (-> device_
				 ("getPlatformDevice<opengl::glx::PlatformDevice>"))))
	 (IGL_ASSERT (!= platformDevice nullptr))
	 (setf drawable (platformDevice->createTextureFromNativeDrawable width_
									 height_
									 &ret))
	 (IGL_ASSERT_MSG (ret.isOk)
			 (ret.message.c_str))
	 (IGL_ASSERT (!= drawable nullptr))
	 (return drawable)))

     (defun createFramebuffer (nativeDrawable)
       (declare (static)
		(type "const std::shared_ptr<ITexture>&" nativeDrawable))
       (let ((framebufferDesc (FramebufferDesc))
	     )
	 (setf (dot framebufferDesc
		    (aref colorAttachments 0)
		    texture)
	       nativeDrawable)
	 (dotimes (i kNumColorAttachments)
	   (when (and i (hex 1))
	     continue)
	   (let ((desc (TextureDesc--new2D (-> nativeDrawable (getFormat))
					   (-> nativeDrawable (dot (getDimensions) width))
					   (-> nativeDrawable (dot (getDimensions) height))
					   (or TextureDesc--TextureUsageBits--Attachment
					       TextureDesc--TextureUsageBits--Sampled)
					   (string "C"))))
	     (setf (dot framebufferDesc
			(aref colorAttachments i)
			texture)
		   (-> device_
		       (createTexture desc nullptr)))))
	 (setf framebuffer_ (-> device_
				(createFramebuffer framebufferDesc nullptr)))
	 (IGL_ASSERT framebuffer_)))

     (defun render (nativeDrawable)
       (declare (static)
		(type "const std::shared_ptr<ITexture>&" nativeDrawable))
       (let ((size (-> framebuffer_
		       (getColorAttachment 0)
		       (getSize))))
	 (if (logior (!= size.width width_)
		     (!= size.height height_))
	     (createFramebuffer nativeDrawable)
	     (framebuffer_->updateDrawable nativeDrawable))
	 (let ((cbDesc (CommandBufferDesc))
	       (buffer (-> commandQueue_
			   (createCommandBuffer cbDesc nullptr)))
	       (viewport (igl--Viewport (curly 0s0 0s0 (static_cast<float> width_)
					       (static_cast<float> height_)
					       0s0 1s0)))
	       (scissor (igl--ScissorRect (curly 0 0
						 (static_cast<uint32_t> width_)
					       (static_cast<uint32_t> height_))))
	       (commands (buffer->createRenderCommandEncoder renderPass_
							     framebuffer_)))
	   ,@(loop for e in `((bindRenderPipelineState renderPipelineState_Triangle_)
			      (bindViewport viewport)
			      (bindScissorRect scissor)
			      (pushDebugGroupLabel (string "render triangle")
						   (igl--Color 1 0 0))
			      (draw PrimitiveType--Triangle 0 3)
			      (popDebugGroupLabel)
			      (endEncoding))
		   collect
		   (destructuring-bind (name &rest args) e
		     `(-> commands (,name ,@args))))
	   (buffer->present nativeDrawable)
	   (commandQueue_->submit *buffer))))
     
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))
       "(void) argc;"
       "(void) argv;"

       (renderPass_.colorAttachments.resize kNumColorAttachments)
       (initWindow &window_)
       (initIGL)
       (createFramebuffer (getNativeDrawable))
       (createRenderPipeline)
       (while (!glfwWindowShouldClose window_)
	      (render (getNativeDrawable))
	      (glfwPollEvents))
       (setf renderPipelineState_Triangle_ nullptr)
       (setf framebuffer_ nullptr
	     )
       (device_.reset nullptr)
       (glfwDestroyWindow window_)
       (glfwTerminate)
       (return 0)))
   :omit-parens t))


