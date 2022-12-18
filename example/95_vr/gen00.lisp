(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-change-case")
  (ql:quickload "cl-ppcre"))

(in-package :cl-cpp-generator2)

(let ((log-preamble
       `(do0
	 (include			;<iostream>
					;<iomanip>
					;<chrono>
					;<thread>
	  <spdlog/spdlog.h>
	  ))))

  (progn
    ;; for classes with templates use write-source and defclass+
    ;; for cpp files without header use write-source
    ;; for class definitions and implementation in separate h and cpp file
    (defparameter *source-dir* #P"example/95_vr/source00/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*))
    (ensure-directories-exist *full-source-dir*)
    (load "util.lisp")

    (defparameter *includes* `(do0
					;(include <spdlog/spdlog.h>)
			       (include
					;<tuple>
					;<mutex>
					;<thread>
				<iostream>
					;<iomanip>
					;<chrono>
					;<cassert>
					;  <memory>
				)

			       #+nil
			       (include
				"App.h"
				"AttribPointer.h"
				"core.h"
				"Cube.h"
				"Egl.h"
				"format.h"
				"format-inl.h"
				"Framebuffer.h"
				"Geometry.h"
				"Program.h"
				"Renderer.h"
				"Vertex.h"


				)

			       (include "VrApi.h"
					"VrApi_Helpers.h"
					"VrApi_Input.h"
					"VrApi_SystemUtils.h"
					"android_native_app_glue.h"
					<EGL/egl.h>
					<EGL/eglext.h>
					<GLES3/gl3.h>
					<android/log.h>
					<android/window.h>
					;<cstdin>
					<vector>
					<cstdlib>
					<unistd.h>)))
    (defparameter *uniforms*
      `((:name "model-matrix" :type mat4 :ptype GLfloat*)
	(:name "view-matrix" :type mat4 :ptype GLfloat*)
	(:name "projection-matrix" :type mat4 :ptype GLfloat*)))

    (let ((name `App))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  (include "Egl.h"
				   "Renderer.h")
			  #+nil(include
				"AttribPointer.h"
				"core.h"
				"Cube.h"
				"Egl.h"
				"format.h"
				"format-inl.h"
				"Framebuffer.h"
				"Geometry.h"
				"Program.h"
				"Renderer.h"
				"Vertex.h")
			  ,*includes*)
       :implementation-preamble `(do0
				  (include ,(format nil "~a.h" name))
				  ,*includes*)
       :code `(do0
	       (defclass ,name ()
		 "public:"
		 "ovrJava* java;"
		 "bool resumed;"
		 "Egl egl;"
		 "Renderer renderer;"
		 "ANativeWindow* window;"
		 "ovrMobile* ovr;"
		 "bool back_button_down_previous_frame;"
		 "uint64_t frame_index;"

		 (defmethod ,name (java)
		   (declare
					;  (explicit)
		    (type ovrJava* java)
		    (construct
		     (java java)
		     (resumed false)
		     (egl (Egl))
		     (renderer
		      (Renderer
		       (vrapi_GetSystemPropertyInt
			java
			VRAPI_SYS_PROP_SUGGESTED_EYE_TEXTURE_WIDTH)
		       (vrapi_GetSystemPropertyInt
			java
			VRAPI_SYS_PROP_SUGGESTED_EYE_TEXTURE_HEIGHT)))
		     (window nullptr)
		     (ovr nullptr)
		     (back_button_down_previous_frame false)
		     (frame_index 0))
		    (values :constructor)))
		 (defmethod update_vr_mode ()
		   (if (and resumed
			    (!= nullptr window))
		       (when (== nullptr ovr)
			 (let ((mode_parms (vrapi_DefaultModeParms java)))
			   (setf mode_parms.Flags
				 (logior mode_parms.Flags
					 VRAPI_MODE_FLAG_NATIVE_WINDOW)
				 )
			   (setf mode_parms.Flags
				 (logand mode_parms.Flags
					 ~VRAPI_MODE_FLAG_RESET_WINDOW_FULLSCREEN)
				 )
			   (setf mode_parms.Display (reinterpret_cast<size_t> egl.display)
				 mode_parms.WindowSurface (reinterpret_cast<size_t> window)
				 mode_parms.ShareContext (reinterpret_cast<size_t> egl.context)
				 )
			   ,(lprint :msg "enter vr mode")
			   (setf ovr (vrapi_EnterVrMode &mode_parms))
			   (when (== nullptr
				     ovr)
			     ,(lprint :msg "error: cant enter vr mode")
			     (std--exit -1))
			   (vrapi_SetClockLevels
			    ovr
			    CPU_LEVEL
			    GPU_LEVEL)))
		       (unless (== nullptr
				   ovr)
			 ,(lprint :msg "leave vr mode")
			 (vrapi_LeaveVrMode ovr)
			 (setf ovr nullptr))))
		 (defmethod handle_input ()
		   (let ((back_button_down_current_frame false)
			 (i 0)
			 (capability (ovrInputCapabilityHeader)))
		     (while (<= 0
				(vrapi_EnumerateInputDevices
				 ovr i &capability))
		       (when (== ovrControllerType_TrackedRemote
				 capability.Type)
			 (let ((input_state (ovrInputStateTrackedRemote)))
			   (setf input_state.Header.ControllerType
				 ovrControllerType_TrackedRemote)
			   (when (== ovrSuccess
				     (vrapi_GetCurrentInputState
				      ovr capability.DeviceID
				      &input_state.Header))
			     (setf back_button_down_current_frame
				   (logior back_button_down_current_frame
					   (logand input_state.Buttons
						   ovrButton_Back)))
			     (setf back_button_down_current_frame
				   (logior back_button_down_current_frame
					   (logand input_state.Buttons
						   ovrButton_B)))
			     (setf back_button_down_current_frame
				   (logior back_button_down_current_frame
					   (logand input_state.Buttons
						   ovrButton_Y))))))
		       (incf i))
		     (when (and back_button_down_previous_frame
				(not back_button_down_current_frame))
		       (vrapi_ShowSystemUI java
					   VRAPI_SYS_UI_CONFIRM_QUIT_MENU))
		     (setf back_button_down_previous_frame
			   back_button_down_current_frame)

		     ))
		 #+nil (defmethod ,(format nil "~~~a" name) ()
			 (declare
			  (values :constructor)))))))


    (let ((name `Renderer))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  (include "Framebuffer.h"
				   "Program.h"
				   "Geometry.h"
				   )
			  ,*includes*)
       :implementation-preamble `(do0
				  (include ,(format nil "~a.h" name))
				  ,*includes*)
       :code `(do0
	       (defclass ,name ()
		 "public:"
		 "std::vector<Framebuffer> framebuffers;"
		 "Program program;"
		 "Geometry geometry;"
		 (defmethod ,name (width height)
		   (declare
					;  (explicit)
		    (type GLsizei width height)
		    (construct
		     (program (Program))
		     (geometry (Geometry))
		     )
		    (values :constructor))
		   (dotimes (i VRAPI_FRAME_LAYER_EYE_MAX)
		     (framebuffers.push_back
		      (Framebuffer width height)))
		   )
		 (defmethod render_frame (tracking)
		   (declare (type ovrTracking2* tracking)
			    (values ovrLayerProjection2))
		   (let ((model_matrix (ovrMatrix4f_CreateTranslation 0s0 0s0 -1s0)))
		     (setf model_matrix (ovrMatrix4f_Transpose &model_matrix))
		     (let ((layer (vrapi_DefaultLayerProjection2)))
		       (setf layer.Header.Flags
			     (logior layer.Header.Flags
				     VRAPI_FRAME_LAYER_FLAG_CHROMATIC_ABERRATION_CORRECTION)
			     layer.HeadPose
			     tracking->HeadPose))
		     (dotimes (i VRAPI_FRAME_LAYER_EYE_MAX)
		       (let (
			     (view_matrix (ovrMatrix4f_Transpose
					   (ref
					    (-> tracking
						(dot (aref Eye i)
						     ViewMatrix)))))

			     (projection_matrix
			      (ovrMatrix4f_Transpose
			       (ref (-> tracking
					(dot (aref Eye i)
					     ProjectionMatrix))))
			       )
			     (*framebuffer (ref (dot framebuffers (at i))))

			     )
			 (setf (dot layer (aref Textures i) ColorSwapChain)
			       framebuffer->color_texture_swap_chain)
			 (setf (dot layer (aref Textures i) SwapChainIndex)
			       framebuffer->swap_chain_index)
			 (comments "this seems to be the heart of the thing. maybe they distort a pre-rendered version to match current head position")
			 (setf (dot layer (aref Textures i) TexCoordsFromTanAngles)
			       (ovrMatrix4f_TanAngleMatrixFromProjection
				(ref (-> tracking
					 (dot (aref Eye i)
					      ProjectionMatrix)))))
			 (glBindFramebuffer
			  GL_DRAW_FRAMEBUFFER
			  (dot framebuffer->framebuffers
			       (at framebuffer->swap_chain_index)))
			 (glEnable GL_CULL_FACE)
			 (glEnable GL_DEPTH_TEST)
			 (glEnable GL_SCISSOR_TEST)
			 (glViewport 0 0
				     framebuffer->width framebuffer->height)
			 (glScissor 0 0
				    framebuffer->width framebuffer->height)
			 (glClearColor 0s0 0s0 0s0 0s0)
			 (glClear (logior GL_COLOR_BUFFER_BIT
					  GL_DEPTH_BUFFER_BIT))
			 (glUseProgram program.program)
			 (let ((count 1)
			       (transpose GL_FALSE))
			   ,@(loop for e in *uniforms*
				   collect
				   (destructuring-bind (&key name type ptype)
				       e
				     `(glUniformMatrix4fv
				       (dot program
					    (aref uniform_locations
						  ,(cl-change-case:constant-case
						    (format nil "uniform-~a" name))))
				       count
				       transpose
				       (,(format nil "reinterpret_cast<const ~a>" ptype)
					 (ref ,(cl-change-case:snake-case
						name)))))))
			 (glBindVertexArray geometry.vertex_array)
			 (glDrawElements GL_TRIANGLES
					 (dot geometry
					      cube
					      indices
					      (size))
					 GL_UNSIGNED_SHORT
					 nullptr)
			 (glBindVertexArray 0)
			 (glUseProgram 0)

			 (do0
			  (glClearColor 0s0 0s0 0s0 1s0)

			  ,@(loop for (x y w h) in `((0 0 1 framebuffer->height)
						     ((- framebuffer->width 1) 0 1 framebuffer->height)
						     (0 0 framebuffer->width 1)
						     (0 (- framebuffer->height 1)
							framebuffer->width 1))
				  collect
				  `(do0
				    (glScissor ,x ,y ,w ,h)
				    (glClear GL_COLOR_BUFFER_BIT)))

			  (let ((ATTACHMENTS ("std::array<const GLenum,1>"
					      (curly GL_DEPTH_ATTACHMENT))))
			    (declare (type "static auto" ATTACHMENTS))
			    (glInvalidateFramebuffer GL_DRAW_FRAMEBUFFER
						     (ATTACHMENTS.size)
						     (ATTACHMENTS.data))
			    (glFlush)
			    (glBindFramebuffer GL_DRAW_FRAMEBUFFER 0)
			    (setf framebuffer->swap_chain_index
				  (% (+ framebuffer->swap_chain_index 1)
				     framebuffer->swap_chain_length)))

			  )

			 )))
		   (return layer))

		 #+nil (defmethod ,(format nil "~~~a" name) ()
			 (declare
			  (construct)
			  (values :constructor)))))))

    (let ((name `Cube))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  ,*includes*
			  (include "Vertex.h"))
       :implementation-preamble `(do0
				  (include ,(format nil "~a.h" name))
				  ,*includes*
				  )
       :code `(do0
	       (defclass ,name ()
		 "public:"
		 "std::vector<Vertex> vertices;"
		 "std::vector<unsigned short> indices;"
		 (defmethod ,name ()
		   (declare
					;  (explicit)
		    (construct
		     (vertices
		      (curly
		       ,@(loop for (e f) in `(((-1 1 -1) (1 0 1))
					      ((1 1 -1)  (0 1 0))
					      ((1 1 1)   (0 0 1))
					      ((-1 1 1)  (1 0 0))
					      ((-1 -1 -1) (0 0 1))
					      ((-1 -1 1) (0 1 0))
					      ((1 -1 1)  (1 0 1))
					      ((1 -1 -1) (1 0 0)))
			       collect
			       `(curly (Vertex ("std::array<float,3>" (curly ,@e))
					       ("std::array<float,3>" (curly ,@f)))))))
		     (indices
		      (curly 0 1 2 2 0 3
			     4 6 5 6 4 7
			     2 6 7 7 1 2
			     0 4 5 5 3 0
			     3 5 6 6 2 3
			     0 1 7 7 4 0)))
		    (values :constructor))))
	       )))

    (let ((name `Vertex))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  (include <array>)
			  ,*includes*)
       :implementation-preamble `(do0
				  ,*includes*
				  (include ,(format nil "~a.h" name))
				  )
       :code `(do0
	       (defclass ,name ()
		 "public:"
		 "std::array<float,4> position;"
		 "std::array<float,4> color;"
		 (defmethod ,name (p c)
		   (declare
		    (type "std::array<float,3>" p c)
		    (construct (position ("std::array<float,4>"
					  (curly
					   (aref p 0)
					   (aref p 1)
					   (aref p 2)
					   0s0)))
			       (color ("std::array<float,4>"
				       (curly
					(aref c 0)
					(aref c 1)
					(aref c 2)
					0s0))))
		    (values :constructor)))
		 #+nil (defmethod ,(format nil "~~~a" name) ()
			 (declare
					;  (explicit)
		    	  (construct)
			  (values :constructor)))))))


    (let ((name `AttribPointer))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  ,*includes*)
       :implementation-preamble `(do0
				  ,*includes*
				  (include ,(format nil "~a.h" name))
				  )
       :code `(do0
	       (defclass ,name ()
		 "public:"
		 "GLint size;"
		 "GLenum type;"
		 "GLboolean normalized;"
		 "GLsizei stride;"
		 "const GLvoid* pointer;"
		 (defmethod ,name (size type normalized stride pointer)

		   (declare
		    (type GLint size)
		    (type GLenum type)
		    (type GLboolean normalized)
		    (type GLsizei stride)
		    (type "const GLvoid*" pointer)
					;  (explicit)
		    (construct (size size)
			       (type type)
			       (normalized normalized)
			       (stride stride)
			       (pointer pointer))
		    (values :constructor)))
		 #+nil (defmethod ,(format nil "~~~a" name) ()
			 (declare
					;  (explicit)

			  (construct
			   )
			  (values :constructor))

			 )
		 )
	       )))

    (let ((name `Geometry))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  ,*includes*
			  (include "Cube.h"))
       :implementation-preamble `(do0
				  ,*includes*
				  (include ,(format nil "~a.h" name))
				  (include "DataExtern.h")
				  )
       :code `(do0
	       (defclass ,name ()
		 "public:"
		 "GLuint vertex_array, vertex_buffer, index_buffer;"
		 "Cube cube;"
		 (defmethod ,name ()
		   (declare
		    (construct)
		    (values :constructor))
		   (glGenVertexArrays 1
				      &vertex_array)
		   (glBindVertexArray vertex_array)
		   (glGenBuffers 1 &vertex_buffer)
		   (glBufferData GL_ARRAY_BUFFER
				 (cube.vertices.size)
				 (cube.vertices.data)
				 GL_STATIC_DRAW)
		   (do0 (let ((i 0))
			  (for-range (attrib ATTRIB_POINTERS)
				     (glEnableVertexAttribArray i)
				     (glVertexAttribPointer
				      i attrib.size attrib.type
				      attrib.normalized attrib.stride
				      attrib.pointer)
				     (incf i)))
			(glGenBuffers 1 &index_buffer)
			(glBindBuffer GL_ELEMENT_ARRAY_BUFFER
				      index_buffer)
			(do0 (glBufferData GL_ELEMENT_ARRAY_BUFFER
					   (dot cube
						indices
						(size))
					   (cube.indices.data)
					   GL_STATIC_DRAW)
			     (glBindVertexArray 0))))
		 (defmethod ,(format nil "~~~a" name) ()
		   (declare (values :constructor))
		   (glDeleteBuffers 1 &index_buffer)
		   (glDeleteBuffers 1 &vertex_buffer)
		   (glDeleteVertexArrays 1 &vertex_array))))))

    (let ((name `Egl))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  ,*includes*
			  (include <array>))
       :implementation-preamble `(do0
				  (include ,(format nil "~a.h" name))
				  ,*includes*)
       :code `(do0
	       (defclass ,name ()
		 "public:"
		 "EGLDisplay display;"
		 "EGLContext context;"
		 "EGLSurface surface;"
		 (defmethod ,name ()
		   (declare
					;  (explicit)
		    (construct
		     (display (eglGetDisplay EGL_DEFAULT_DISPLAY)))
		    (values :constructor))
		   (when (== EGL_NO_DISPLAY
			     display)
		     ,(lprint :msg "can't get egl display"))
		   (when (== EGL_FALSE
			     (eglInitialize display
					    nullptr
					    nullptr))
		     ,(lprint :msg "can't initialize egl display"))
		   (do0
		    ,(lprint :msg "get number of egl configs ..")
		    (let ((numConfigs ((lambda ()
					 (declare (capture "&"))
					 (let ((n (EGLint 0)))
					   (when (== EGL_FALSE
						     (eglGetConfigs display nullptr
								    0 &n))
					     ,(lprint :msg "cant get number of egl configs"))
					   (return n)))))
			  (configs (std--vector<EGLConfig> )))
		      (configs.resize numConfigs)
		      (when (== EGL_FALSE
				(eglGetConfigs display (configs.data) numConfigs &numConfigs))
			,(lprint :msg "cant get egl configs"))
		      ))
		   (do0
		    ,(lprint :msg "choose egl config")
		    (let ((found_config (EGLConfig nullptr)))
		      (for-range
		       (config configs)
		       (let ((renderable_type
			      ((lambda (renderable_type)
				 (declare (type auto renderable_type)
					  (capture "&"))
				 (when (== EGL_FALSE
					   (eglGetConfigAttrib display
							       config
							       EGL_RENDERABLE_TYPE
							       &renderable_type))
				   ,(lprint :msg "cant get EGL config renderable type"))
				 (return renderable_type))
			       (EGLint 0))))
			 (when (or ,@(loop for e in `(EGL_OPENGL_ES3_BIT_KHR
						      )
					   collect
					   `(== 0
						(logand renderable_type
							,e))
					   ))
			   continue)
			 (let ((surface_type ((lambda (i)
						(declare (type auto i)
							 (capture "&"))
						(when (== EGL_FALSE
							  (eglGetConfigAttrib
							   display
							   config
							   EGL_SURFACE_TYPE
							   &i))
						  ,(lprint :msg "cant get surface config type"))
						(return i))
					      (EGLint 0))))
			   (when (or ,@(loop for e in `(
							EGL_PBUFFER_BIT
							EGL_WINDOW_BIT)
					     collect
					     `(== 0
						  (logand surface_type
							  ,e))
					     ))
			     continue)
			   ,(let ((l-attrib `((red-size 8)
					      (green-size 8)
					      (blue-size 8)
					      (alpha-size 8)
					      (depth-size 0)
					      (stencil-size 0)
					      (samples 0))))
			      `(progn
				 (let ((check (lambda (attrib)
						(declare (type auto attrib)
							 (values auto)
							 (capture "&"))
						(let ((value (EGLint 0)))
						  (when (== EGL_FALSE
							    (eglGetConfigAttrib display config attrib &value))
						    ,(lprint :msg "cant get config attrib")))
						(return value))))
				   (when (and ,@(remove-if
						 #'null
						 (loop for (e f) in l-attrib and i from 0
						       collect
						       (let ((attrib (cl-change-case:constant-case (format nil "egl-~a" e))))
							 (unless (eq 0 f)
							   `(<= ,f (check ,attrib)))))))
				     (setf found_config config)
				     break)))))))
		      (when (== nullptr found_config)
			,(lprint :msg "cant choose egl config")
			(std--exit -1))
		      (do0
		       ,(lprint :msg "create egl config")
		       (let ((CONTEXT_ATTRIBS ("std::array<EGLint,3>"
					       (curly EGL_CONTEXT_CLIENT_VERSION
						      3
						      EGL_NONE))))
			 (setf context
			       (eglCreateContext display
						 found_config
						 EGL_NO_CONTEXT
						 (CONTEXT_ATTRIBS.data)))
			 (when (== EGL_NO_CONTEXT
				   context)
			   ,(lprint :msg "can't create egl context")
			   (std--exit -1))

			 ))
		      (do0
		       ,(lprint :msg "create egl surface")
		       ,(let ((l `(EGL_WIDTH 16 EGL_HEIGHT 16 EGL_NONE)))
			  `(let ((SURFACE_ATTRIBS (,(format nil "std::array<EGLint,~a>" (length l))
						    (curly ,@l))))
			     (setf surface
				   (eglCreatePbufferSurface
				    display
				    found_config
				    (SURFACE_ATTRIBS.data)))
			     (when (== EGL_NO_SURFACE
				       surface)
			       ,(lprint :msg "can't create pixel buffer surface")
			       (std--exit -1))

			     )))
		      (do0
		       ,(lprint :msg "make egl context current")
		       (when (== EGL_FALSE
				 (eglMakeCurrent display
						 surface
						 surface
						 context))
			 ,(lprint :msg "can't make egl context current")
			 (std--exit -1)))

		      )))
		 #+nil (defmethod ,(format nil "~~~a" name) ()
			 (declare
			  (values :constructor)))))))

    (let ((name `Framebuffer))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  ,*includes*
			  (include "DataTypes.h")

			  )
       :implementation-preamble `(do0
				  (include ,(format nil "~a.h" name))
				  ,*includes*
				  (include
				   <EGL/egl.h>
				   <EGL/eglext.h>
				   <GLES3/gl3.h>))
       :code `(do0
	       (defclass ,name ()
		 "public:"
		 "int swap_chain_index;"
		 "int swap_chain_length;"
		 "GLsizei width;"
		 "GLsizei height;"
		 "ovrTextureSwapChain* color_texture_swap_chain;"
		 "std::vector<GLuint> depth_renderbuffers;"
		 "std::vector<GLuint> framebuffers;"
		 (defmethod ,name (w h)
		   (declare
		    (type GLsizei w h)
		    (construct
		     (swap_chain_index 0)
		     (width w)
		     (height h)
		     (color_texture_swap_chain
		      (vrapi_CreateTextureSwapChain3
		       VRAPI_TEXTURE_TYPE_2D
		       GL_RGBA8
		       w h 1 3)))
		    (values :constructor))
		   (when (== nullptr
			     color_texture_swap_chain)
		     ,(lprint :msg "cant create color texture swap chain")
		     (std--exit -1))
		   (setf swap_chain_length
			 (vrapi_GetTextureSwapChainLength
			  color_texture_swap_chain))
		   (do0
		    (depth_renderbuffers.resize swap_chain_length)
		    (glGenRenderbuffers swap_chain_length
					(depth_renderbuffers.data)))
		   (do0
		    (framebuffers.resize swap_chain_length)
		    (glGenFramebuffers swap_chain_length
				       (framebuffers.data)))
		   (dotimes (i swap_chain_length)
		     (do0
		      ,(lprint :msg "color texture " :vars `(i ))
		      (let ((color_texture (vrapi_GetTextureSwapChainHandle
					    color_texture_swap_chain
					    i))))
		      (glBindTexture GL_TEXTURE_2D
				     color_texture)
		      (glTexParameteri GL_TEXTURE_2D
				       GL_TEXTURE_MIN_FILTER
				       GL_LINEAR)
		      (glTexParameteri GL_TEXTURE_2D
				       GL_TEXTURE_MAG_FILTER
				       GL_LINEAR)
		      (glTexParameteri GL_TEXTURE_2D
				       GL_TEXTURE_WRAP_S
				       GL_CLAMP_TO_EDGE)
		      (glTexParameteri GL_TEXTURE_2D
				       GL_TEXTURE_WRAP_T
				       GL_CLAMP_TO_EDGE)
		      (glBindTexture GL_TEXTURE_2D
				     0))
		     (do0 ,(lprint :msg "create depth buffer"
				   :vars `(i))
			  (glBindRenderbuffer
			   GL_RENDERBUFFER
			   (dot depth_renderbuffers (at i)))
			  (glRenderbufferStorage GL_RENDERBUFFER
						 GL_DEPTH_COMPONENT24
						 w h)
			  (glBindRenderbuffer GL_RENDERBUFFER 0))
		     (do0
		      ,(lprint :msg "create framebuffer"
			       :vars `(i))
		      (glBindFramebuffer GL_DRAW_FRAMEBUFFER
					 (dot framebuffers (at i)))
		      (glFramebufferTexture2D GL_DRAW_FRAMEBUFFER
					      GL_COLOR_ATTACHMENT0
					      GL_TEXTURE_2D
					      color_texture
					      0)
		      (glFramebufferRenderbuffer GL_DRAW_FRAMEBUFFER
						 GL_DEPTH_ATTACHMENT
						 GL_RENDERBUFFER
						 (depth_renderbuffers.at i))
		      (progn
			(let ((status (glCheckFramebufferStatus
				       GL_DRAW_FRAMEBUFFER)))
			  (unless (== GL_FRAMEBUFFER_COMPLETE
				      status)
			    ,(lprint :msg "cant create framebuffer" :vars `(i))
			    (std--exit -1))))
		      (glBindFramebuffer GL_DRAW_FRAMEBUFFER
					 0))
		     ))
		 (defmethod ,(format nil "~~~a" name) ()
		   (declare
		    (values :constructor))
		   (glDeleteFramebuffers swap_chain_length
					 (framebuffers.data))
		   (glDeleteRenderbuffers swap_chain_length
					  (depth_renderbuffers.data))
		   (vrapi_DestroyTextureSwapChain
		    color_texture_swap_chain)))
	       )))

    (let ((name `Program))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  "#pragma once"
			  ,*includes*
			  (include "DataTypes.h"
				   "DataExtern.h"
				   <array>))

       :implementation-preamble `(do0
				  (do0
				   (do0 "#define FMT_HEADER_ONLY"
					(include "core.h"
						 "format.h"
						 <android/log.h>)))

				  (include ,(format nil "~a.h" name))
				  ,*includes*)
       :code `(do0
	       (defclass ,name ()
		 "public:"
		 "GLuint program;"
		 "std::array<GLint,UNIFORM_END> uniform_locations;"
		 (defmethod compileShader (type str)
		   (declare (type GLenum type)
			    (type "std::string" str)
			    (values GLuint))
		   (let ((shader (glCreateShader type)))
		     (let ((c_str (str.c_str)))
		       (glShaderSource shader
				       1
				       (ref c_str)
				       nullptr))
		     (glCompileShader shader)
		     (progn
		       (let ((status (GLint 0)))
			 (glGetShaderiv shader
					GL_COMPILE_STATUS
					&status)
			 (when (== GL_FALSE
				   status)
			   (let ((length (GLint 0)))
			     (glGetShaderiv shader
					    GL_INFO_LOG_LENGTH
					    &length)
			     (let ((log (std--vector<char> length)))
			       (glGetShaderInfoLog shader
						   length
						   nullptr
						   (log.data))
			       (let ((logstr (std--string (std--begin log)
							  (std--end log))))
				 ,(lprint :msg "cant compile shader"
					  :vars `(logstr))
				 (std--exit -1)))))
			 (return shader)))))
		 (defmethod ,name ()
		   (declare
		    (construct (program (glCreateProgram)))
		    (values :constructor))
		   (let ((FRAGMENT_SHADER
			  (std--string
			   (string-r
			    ,(cl-cpp-generator2::emit-c
			      :code
			      `(do0
				"#version 300 es"
				"in lowp vec3 vColor;"
				"out lowp vec4 outColor;"
				(defun main ()
				  (setf outColor
					(vec4 vColor 1s0))))))))
			 (VERTEX_SHADER
			  (std--string
			   (string-r
			    ,(cl-cpp-generator2::emit-c
			      :code
			      `(do0
				"#version 300 es"
				"in vec3 aPosition;"
				"in vec3 aColor;"
				"uniform mat4 uModelMatrix;"
				"uniform mat4 uViewMatrix;"
				"uniform mat4 uProjectionMatrix;"
				"out vec3 vColor;"
				(defun main ()
				  (setf gl_Position
					(* uProjectionMatrix
					   (* uViewMatrix
					      (* uModelMatrix
						 (vec4 (* aPosition 1s0)
						       1s0))))
					vColor aColor)))))))
			 (vertexShader (compileShader GL_VERTEX_SHADER
						      VERTEX_SHADER))
			 (fragmentShader (compileShader GL_FRAGMENT_SHADER
							FRAGMENT_SHADER)))
		     (glAttachShader program
				     vertexShader)
		     (glAttachShader program
				     fragmentShader)
		     (let ((i 0))
		       (for-range (name ATTRIB_NAMES)
				  (glBindAttribLocation program
							i
							(name.c_str))
				  (incf i)))
		     (glLinkProgram program)
		     (progn
		       (let ((status (GLint 0)))
			 (glGetProgramiv program
					 GL_LINK_STATUS
					 &status)
			 (when (== GL_FALSE
				   status)
			   (let ((length (GLint 0)))
			     (glGetProgramiv program
					     GL_INFO_LOG_LENGTH
					     &length)
			     (let ((log (std--vector<char> length)))
			       (glGetProgramInfoLog program
						    length
						    nullptr
						    (log.data))
			       (let ((logstr (std--string (std--begin log)
							  (std--end log))))
				 ,(lprint :msg "cant compile shader"
					  :vars `(logstr))
				 (std--exit -1)))))
			 ))
		     (dotimes (i UNIFORM_END)
		       (setf (aref uniform_locations i)
			     (glGetUniformLocation program
						   (dot (aref UNIFORM_NAMES i)
							(c_str)))))))
		 (defmethod ,(format nil "~~~a" name) ()
	       	   (declare
		    (values :constructor))
		   (glDeleteProgram program))))))

    (write-source
     (asdf:system-relative-pathname
      'cl-cpp-generator2
      (merge-pathnames #P"DataExtern.h"
		       *source-dir*))
     `(do0
       "#pragma once"
       (include <vector>
		<string>
		"AttribPointer.h")
       "extern const std::vector<std::string> ATTRIB_NAMES;"
       "extern const std::vector<std::string> UNIFORM_NAMES;"
       "extern const std::array<AttribPointer,2> ATTRIB_POINTERS;"
       ))
    (write-source
     (asdf:system-relative-pathname
      'cl-cpp-generator2
      (merge-pathnames #P"DataTypes.h"
		       *source-dir*))
     `(do0
       "#pragma once"
       (include <vector>
		<string>)
       (do0 "#define FMT_HEADER_ONLY"
	    (include "core.h"
		     "format.h"
		     <android/log.h>))

       (space enum attrib
	      (curly
	       ATTRIB_BEGIN
	       (= ATTRIB_POSITION ATTRIB_BEGIN)
	       ATTRIB_COLOR
	       ATTRIB_END))
       (space enum uniform
	      (curly
	       UNIFORM_BEGIN
	       (= UNIFORM_MODEL_MATRIX UNIFORM_BEGIN)
	       UNIFORM_VIEW_MATRIX
	       UNIFORM_PROJECTION_MATRIX
	       UNIFORM_END))
       (space enum level
	      (curly (= CPU_LEVEL 2)
		     (= GPU_LEVEL 3)))))
    (write-source
     (asdf:system-relative-pathname
      'cl-cpp-generator2
      (merge-pathnames #P"main.cpp"
		       *source-dir*))
     `(do0



       ,*includes*
       #+nil (do0
					;(include <spdlog/spdlog.h>)
	      (include
					;<tuple>
					;<mutex>
					;<thread>
	       <iostream>
					;<iomanip>
					;<chrono>
					;<cassert>
					;  <memory>
	       )


					; (include "AttribPointer.h")
	      (include "VrApi.h"
		       "VrApi_Helpers.h"
		       "VrApi_Input.h"
		       "VrApi_SystemUtils.h"
		       "android_native_app_glue.h"

		       <android/log.h>
		       <android/window.h>
					;<cstdin>
		       <cstdlib>
		       <unistd.h>

		       )
	      (include
	       <EGL/egl.h>
	       <EGL/eglext.h>
	       <GLES3/gl3.h>))
       (include "App.h"
		"Vertex.h")

       (do0
	"#define FMT_HEADER_ONLY"

	(include "core.h"))

       "const std::vector<std::string> ATTRIB_NAMES = {\"aPosition\",\"aColor\"};"
       "const std::vector<std::string> UNIFORM_NAMES = {\"uModelMatrix\",\"uViewMatrix\",\"uProjectionMatrix\"};"

       (let ((ATTRIB_POINTERS
	      (curly (AttribPointer 3 GL_FLOAT GL_FALSE (sizeof Vertex)
				    (reinterpret_cast<GLvoid*> (offsetof Vertex position)))
		     (AttribPointer 3 GL_FLOAT GL_FALSE (sizeof Vertex)
				    (reinterpret_cast<GLvoid*> (offsetof Vertex color))))))
	 (declare (type "const std::array<AttribPointer,2>" ATTRIB_POINTERS)))

       (defun app_on_cmd (android_app cmd)
	 (declare (type android_app* android_app)
		  (type int32_t cmd))
	 (let ((app (reinterpret_cast<App*> android_app->userData)))
	   (case cmd
	     ,@(loop for e in `((start)
				(resume (setf app->resumed true))
				(pause (setf app->resumed false))
				(stop)
				(destroy (setf app->window nullptr))
				(init-window (setf app->window
						   android_app->window))
				(term-window (setf app->window nullptr))
				;; some more commands:
				(input-changed)
				(window-resized)
				(window-redraw-needed)
				(content-rect-changed)
				(gained-focus)
				(lost-focus)
				(config-changed)
				(low-memory)
				(save-state)
				)
		     collect
		     (destructuring-bind (name &optional code) e
		       (let* ((clause (cl-change-case:constant-case
				       (format nil "app-cmd-~a" name)) )
			      (res `(,clause
				     ,(lprint :msg (format nil "~a" e)))))
			 (when code
			   (setf res (append res `(,code))))
			 res)))
	     (t ,(lprint :msg "app_on_cmd default")))))

       (defun android_main (android_app)
	 (declare (type android_app* android_app))
	 (ANativeActivity_setWindowFlags
	  android_app->activity
	  AWINDOW_FLAG_KEEP_SCREEN_ON
	  0)
	 (do0
	  ,(lprint :msg "attach current thread"
		   :level "info")
	  (let ((java (ovrJava)))
	    (setf java.Vm android_app->activity->vm)
	    (-> java.Vm			;"(*java.Vm)"
		(AttachCurrentThread	;java.Vm
		 &java.Env
		 nullptr))
	    (setf java.ActivityObject android_app->activity->clazz)))
	 (do0
	  ,(lprint :msg "initialize vr api")
	  (let ((init_params (vrapi_DefaultInitParms
			      &java)))
	    (unless (== VRAPI_INITIALIZE_SUCCESS
			(vrapi_Initialize &init_params))
	      ,(lprint :msg "can't initialize vr api")
	      (std--exit 1))))
	 (do0
	  (let ((app (App
		      &java)))
	    (setf android_app->userData (ref app)
		  android_app->onAppCmd app_on_cmd)
	    (while (not android_app->destroyRequested)
	      (while true
		(let ((events 0)
		      (source nullptr))
		  (declare (type "android_poll_source*"
				 source))
		  (when (< (ALooper_pollAll
			    (or android_app->destroyRequested
				(? (!= nullptr app.ovr)
				   0 -1))
			    nullptr
			    &events
			    (reinterpret_cast<void**> &source))
			   0)
		    break)
		  (unless (== nullptr source)
		    (source->process android_app
				     source))
		  (app.update_vr_mode)
		  ))
	      (app.handle_input)
	      (when (== nullptr
			app.ovr)
		continue)
	      (incf app.frame_index)
	      (let ((display_time
		     (vrapi_GetPredictedDisplayTime
		      app.ovr app.frame_index))
		    (tracking (vrapi_GetPredictedTracking2
			       app.ovr display_time))
		    (layer (app.renderer.render_frame
			    &tracking))
		    (layers ("std::array<ovrLayerHeader2*,1>"
			     (curly
			      &layer.Header)))
		    (frame (curly (= .Flags 0)
				  (= .SwapInterval 1)
				  (= .FrameIndex app.frame_index)
				  (= .DisplayTime display_time)
				  (= .LayerCount 1)
				  (= .Layers (layers.data))
				  )))
		(declare (type ovrSubmitFrameDescription2 frame))
		,(lprint :vars `(app.frame_index
				 display_time))
		(vrapi_SubmitFrame2
		 app.ovr &frame))
	      )
	    ,(lprint :msg "shut down vr api")
	    (vrapi_Shutdown)
	    ,(lprint :msg "detach current thread")
	    (-> java.Vm
		(DetachCurrentThread))

	    )))))

    (with-open-file (s (format nil "~a/CMakeLists.txt" *full-source-dir*)
		       :direction :output
		       :if-exists :supersede
		       :if-does-not-exist :create)
      ;;https://clang.llvm.org/docs/AddressSanitizer.html
      ;; cmake -DCMAKE_BUILD_TYPE=Debug -GNinja ..
      ;;
      (let ((dbg "-ggdb -O0 ")
	    (asan ""
					;"-fno-omit-frame-pointer -fsanitize=address -fsanitize-address-use-after-return=always -fsanitize-address-use-after-scope"
	      )
	    ;; make __FILE__ shorter, so that the log output is more readable
	    ;; note that this can interfere with debugger
	    ;; https://stackoverflow.com/questions/8487986/file-macro-shows-full-path
	    (short-file "" ;"-ffile-prefix-map=/home/martin/stage/cl-cpp-generator2/example/86_glbinding_av/source01/="
	      )
	    (show-err "-Wall -Wextra "	;  -march=arm-v8a
					;" -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self  -Wmissing-declarations -Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wswitch-default -Wundef -Werror -Wno-unused"
					;"-Wlogical-op -Wnoexcept  -Wstrict-null-sentinel  -Wsign-promo-Wstrict-overflow=5  "

	      ))
	(let ((program-name "main"))
	  (macrolet ((out (fmt &rest rest)
		       `(format s ,(format nil "~&~a~%" fmt) ,@rest)))
	    (out "cmake_minimum_required( VERSION 3.0 FATAL_ERROR )")
	    (out "project( ~a LANGUAGES C CXX )" program-name)

            (out "set( CMAKE_C_COMPILER /home/martin/quest2/ndk/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang )")
	    (out "set( CMAKE_CXX_COMPILER /home/martin/quest2/ndk/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++ )")

					;(out "set( CMAKE_CXX_COMPILER g++ )")
	    (out "set( CMAKE_VERBOSE_MAKEFILE ON )")
	    (out "set (CMAKE_CXX_FLAGS_DEBUG \"${CMAKE_CXX_FLAGS_DEBUG} ~a ~a ~a ~a \")" dbg asan show-err short-file)
	    (out "set (CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} ~a ~a ~a ~a \")" dbg asan show-err short-file)
	    (out "set (CMAKE_C_FLAGS \"${CMAKE_C_FLAGS} ~a ~a ~a ~a \")" dbg asan show-err short-file)
	    (out "set (CMAKE_LINKER_FLAGS_DEBUG \"${CMAKE_LINKER_FLAGS_DEBUG} ~a ~a \")" dbg show-err )
					;(out "set( CMAKE_CXX_STANDARD 23 )")



	    (out "set( SRCS ~{~a~^~%~} )"
		 (append
		  (directory (format nil "~a/*.cpp" *full-source-dir*))
		  (directory (format nil "~a/*.c" *full-source-dir*))
		  ))

					;(out "add_executable( mytest ${SRCS} )")
	    (out "add_library (~a SHARED ${SRCS} )" program-name)




	    (out "set_property( TARGET ~a PROPERTY CXX_STANDARD 20 )" program-name)

					;(out "target_link_options( mytest PRIVATE -static-libgcc -static-libstdc++   )")

					;(out "find_package( PkgConfig REQUIRED )")
					;(out "pkg_check_modules( spdlog REQUIRED spdlog )")


	    (out "target_include_directories( ~a PRIVATE
/home/martin/quest2/ndk/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include/
/home/martin/quest2/ovr/VrApi/Include
)" program-name)
	    ;;/home/martin/quest2/ovr/VrApi/Libs/Android/arm64-v8a/Debug
					; /platforms/android-26/arch-arm64/usr/lib
	    (progn
	      (out "add_library( vrapi SHARED IMPORTED )")
	      (out "set_target_properties( vrapi PROPERTIES IMPORTED_LOCATION /home/martin/quest2/ovr/VrApi/Libs/Android/arm64-v8a/Debug/libvrapi.so
 )")
	      )
	    #+nil
	    (progn
	      (out "add_library( c++shared SHARED IMPORTED )")
	      (out "set_target_properties( c++shared PROPERTIES IMPORTED_LOCATION /home/martin/quest2/ndk/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so
 )")
	      )

	    (out "target_link_libraries( ~a PRIVATE ~{~a~^ ~} )"
		 program-name
		 `(android
		   log
		   vrapi
		   EGL
					;c++
		   GLESv3
					;c++shared
		   ))

	    #+nil
	    (out "target_compile_options( ~a PRIVATE ~{~a~^ ~} )"
		 program-name `())

					;(out "target_precompile_headers( ~a PRIVATE fatheader.hpp )" program-name)
	    )))
      )))

