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
      regex
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
				 (setf gl_Position (vec4 (aref pos gl_VertexIndex)
							 .0
							 1.)
				       color (aref gl_VertexIndex))))
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
       (let ((*window (glfwCreateWindow 800 600 (string "OpenGL Triangle") nullptr nullptr)))
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

     (defun initGL ()
       (let ((ctx ("std::make_unique<igl::opengl::glx::Context>"
		   nullptr
		   (glfwGetX11Display)
		   ("reinterpret_cast<igl::opengl::glx::GLXDrawable>" (glfwGetX11Window window_))
		   ("reinterpret_cast<igl::opengl::glx::GLXContext>" (glfwGetGLXContext window_)))))
	 (setf device_ ("std::make_unique<igl::opengl::glx::Device>" (std--move ctx)))
	 (IGL_ASSERT device_)))
     
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))
       "(void) argc;"
       "(void) argv;"
              
       (return 0)))
   :omit-parens t))


