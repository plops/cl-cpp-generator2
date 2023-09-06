(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more)))
  (setf *features* (set-exclusive-or *features* (list :more))))

(progn
  (progn
    (defparameter *source-dir* #P"example/134_grpc_glfw_imgui/source01/src/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*)))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")

  (defun gl-shader-info-log (var-msg)
	       #-more ""
	       #+more
	       (destructuring-bind (&key var msg) var-msg
		`(let ((infoLog (std--vector<char> 512)))
		   (glGetShaderInfoLog ,var (static_cast<GLsizei> (infoLog.size)) nullptr (infoLog.data))
		   (let ((info (std--string (infoLog.begin)
					    (infoLog.end)))))
		   ,(lprint :msg msg
			    :vars `(info))
		   )))
  
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     
     (include<>
      iostream
      string
					;complex
      vector
					;algorithm
      
					;chrono
      thread
      
      filesystem
					;unistd.h
      cstdlib

					;cmath
					;linux/videodev2.h
      future

      )

     (include<>
      glad/gl.h) ;; this needs to come before glfw3.h
     " "
     (include<>

      GLFW/glfw3.h
      
					;glm/glm.hpp
      imgui.h
      imgui_impl_glfw.h
      imgui_impl_opengl3.h

      grpcpp/grpcpp.h
      
      )

     (include
      glgui.grpc.pb.h
      )
     
     (setf "const char *const vertexShaderSrc"
	   (string-r
	    ,(emit-c :code `(do0
			     "#version 450"
			     (space layout (paren (= location 0)) in vec2 aPos)
			     (defun main ()
			       (setf gl_Position (vec4 aPos 1 1))))
		     :omit-redundant-parentheses t)))

     (setf "const char * const fragmentShaderSrc"
	   (string-r
	    ,(emit-c :code `(do0
			     "#version 450"
			     (space layout (paren (= location 0)) out vec4 outColor)
			     (defun main ()
			       (setf outColor (vec4 1 0 0 1))))
		     :omit-redundant-parentheses t)))

     #+more (defun message_callback (source type id severity length message user_param)
	      (declare (type GLenum source type severity)
		       (type GLuint id)
		       (type "[[maybe_unused]] GLsizei" length)
		       (type "GLchar const *" message)
		       (type "[[maybe_unused]] void const *" user_param))
	      ,(lprint :msg "gl"
		       :vars `(source type id severity message)))
					     
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))

       (let ((ch_args))
	 (declare (type "grpc::ChannelArguments" ch_args))
	 (comments "Increase max message size if needed")
	 (ch_args.SetMaxReceiveMessageSize -1)
	 (let ((channel (grpc--CreateCustomChannel (string "localhost:50051")
						   (grpc--InsecureChannelCredentials)
						   ch_args))
	       (stub (glgui--GLGuiService--NewStub channel))
	       ))

	 (let ((get_random_rectangle 
		 (lambda (stub_)
		   (declare (type "std::unique_ptr<glgui::GLGuiService::Stub> const &" stub_)
			    (capture ""))
		   (let ((request (glgui--RectangleRequest))
			 (response (glgui--RectangleResponse))))
		   
		   (let ((context (grpc--ClientContext))))

		   (let ((status (stub_->GetRandomRectangle &context
							    request
							    &response))))
		   
		   (if (status.ok)
		       (do0
			,(lprint :vars `((response.x1))
				 ))
		       (do0
			,(lprint :vars `((status.error_message)))))))))
	 
	 (get_random_rectangle stub)

	 (let ((get_image
		 (lambda (stub_)
		   (declare (type "std::unique_ptr<glgui::GLGuiService::Stub> const &" stub_)
			    (capture ""))
		   (let ((request (glgui--GetImageRequest))
			 (response (glgui--GetImageResponse))))
		   (request.set_width 128)
		   (request.set_height 128)
		   
		   (let ((context (grpc--ClientContext))))
		   (let ((status (stub_->GetImage &context
						  request
						  &response))))

		   (if (status.ok)
		       (do0
			(return response))
		       (do0
			(throw (std--runtime_error (status.error_message)))
			#+nil
			,(lprint :vars `((status.error_message)))))		   
		   ))))
	 
	 
	 )
       
       (do0
	(glfwInit)
	(glfwWindowHint GLFW_CONTEXT_VERSION_MAJOR 4)
	(glfwWindowHint GLFW_CONTEXT_VERSION_MINOR 5)
	(glfwWindowHint GLFW_OPENGL_PROFILE GLFW_OPENGL_CORE_PROFILE)

	(let ((window (glfwCreateWindow 800 600
					(string "v4l")
					nullptr nullptr)))
	  (unless window
	    ,(lprint :msg "Error creating glfw window.")
	    (return -1)))

	(glfwMakeContextCurrent window)
	(glfwSwapInterval 1)

	(unless (gladLoaderLoadGL)
	  ,(lprint :msg "Error initializing glad.")
	  (return -2))

	(do0
	 ,(lprint :msg "Get extensions.")
	 (do0		      ;let ((ext (glGetString GL_EXTENSIONS)))
	  (when (space auto (setf ext (glGetString GL_EXTENSIONS))
		       (!= nullptr ext))
	    (let ((extension_str (std--string ("reinterpret_cast<const char*>" ext))))
	      ,(lprint :msg "extension"
		       :vars `(extension_str)
		       )))))

	(do0
	 (IMGUI_CHECKVERSION)
	 (ImGui--CreateContext)
	 (ImGui_ImplGlfw_InitForOpenGL window true)
	 (ImGui_ImplOpenGL3_Init (string "#version 450 core"))
	 (ImGui--StyleColorsClassic))

	(do0
	 (glEnable GL_CULL_FACE)
	 #+more (glEnable GL_DEBUG_OUTPUT)
	 #+more (glDebugMessageCallback message_callback nullptr))

	(do0
	 #+more ,(lprint :msg "Compile shader.")
	 (let ((success 0)
	       
	       (vertexShader (glCreateShader GL_VERTEX_SHADER)))
	   (glShaderSource vertexShader 1 &vertexShaderSrc nullptr)
	   (glCompileShader vertexShader)

	   (glGetShaderiv vertexShader GL_COMPILE_STATUS &success)
	   (unless success
	     ,(gl-shader-info-log `(:var vertexShader :msg "Vertex shader compilation failed."))
	     (exit -1))
	   )
	 (let ((fragmentShader (glCreateShader GL_FRAGMENT_SHADER)))
	   (glShaderSource fragmentShader 1 &fragmentShaderSrc nullptr)
	   (glCompileShader fragmentShader)

	   (glGetShaderiv fragmentShader GL_COMPILE_STATUS &success)
	   (unless success
	     ,(gl-shader-info-log `(:var fragmentShader :msg "Fragment shader compilation failed."))
	     (exit -1))
	   )
	 (let ((program (glCreateProgram )))
	   (glAttachShader program vertexShader)
	   (glAttachShader program fragmentShader)
	   (glLinkProgram program)
	   (glGetProgramiv program GL_LINK_STATUS &success)
	   (unless success
	     ,(gl-shader-info-log `(:var program :msg "Shader linking failed."))
	     (exit -1))
	   )
	 (glDetachShader program vertexShader)
	 (glDetachShader program fragmentShader)))

       (do0
	(glUseProgram program)
	(glClearColor 1 1 1 1))

              
       (let ((texture (GLuint 0))
	     (texture_w 0)
	     (texture_h 0))
	 (glGenTextures 1 &texture)
	 (glBindTexture GL_TEXTURE_2D texture)
	 
	 (glTexParameteri GL_TEXTURE_2D GL_TEXTURE_MIN_FILTER GL_LINEAR)
	 (glTexParameteri GL_TEXTURE_2D GL_TEXTURE_MAG_FILTER GL_LINEAR))

       (let ((future (std--future<glgui--GetImageResponse>))))
       (let ((update_texture_if_ready
	       (lambda (stub_ future_)
		 (declare (type "std::future<glgui::GetImageResponse>&" future_)
			  (type "std::unique_ptr<glgui::GLGuiService::Stub>&" stub_)
			  (capture &texture_w
				   &texture_h
				   texture
				   get_image
				   ))
		 (if (future_.valid)
		     (when (== (future_.wait_for (std--chrono--seconds 0))
			       std--future_status--ready)
		       (do0		;handler-case
			(let ((response (future_.get)))
			  (glBindTexture GL_TEXTURE_2D texture)
			  (setf texture_w (response.width)
				texture_h (response.height))
			  (glTexImage2D GL_TEXTURE_2D
					0
					GL_RGBA
					texture_w
					texture_h
					0
					GL_RGB
					GL_UNSIGNED_BYTE
					(dot response (data)
					     (c_str)))
					;(comments "Invalidate the future")
			  (setf future_ (std--future<glgui--GetImageResponse>))
			  )
			#+nil ("const std::exception" (&e)
						      ,(lprint :vars `((e.what))))))
		     (setf future_ (std--async
				    std--launch--async
				    get_image
				    (std--ref stub_))))
		 ))))
       
       (while (!glfwWindowShouldClose window)
	      (glfwPollEvents)
					;(glDrawElements GL_TRIANGLES 6 GL_UNSIGNED_INT nullptr)
	      (ImGui_ImplOpenGL3_NewFrame)
	      (ImGui_ImplGlfw_NewFrame)
	      (ImGui--NewFrame)

	      
	       
	      (update_texture_if_ready stub future)
	      (do0
	       (glBindTexture GL_TEXTURE_2D texture)
	       (ImGui--Begin (string "texture"))
	       (ImGui--Image (reinterpret_cast<void*> (static_cast<intptr_t> texture))
			     (ImVec2 (static_cast<float> texture_w)
				     (static_cast<float> texture_h)))
	       (ImGui--End))
	       
	      (space static bool (setf showDemo true))
	      (ImGui--ShowDemoWindow &showDemo)
	      (ImGui--Render)
	      (ImGui_ImplOpenGL3_RenderDrawData (ImGui--GetDrawData))
	      (glfwSwapBuffers window)
	      (glClear GL_COLOR_BUFFER_BIT)
	      (std--this_thread--sleep_for (std--chrono--milliseconds 16))
	      )
       
       
       (do0
	(ImGui_ImplOpenGL3_Shutdown)
	(ImGui_ImplGlfw_Shutdown)
	(ImGui--DestroyContext)
	(glfwDestroyWindow window)
	(glfwTerminate))
       (return 0)))
   :omit-parens t
   :format t
   :tidy nil))

