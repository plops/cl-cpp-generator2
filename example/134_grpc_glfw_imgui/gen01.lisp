(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more)))
  (setf *features* (set-exclusive-or *features* (list ; :more
						 ))))

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

      )

     (include<>
      glad/gl.h) ;; this needs to come before glfw3.h
     " "
     (include<>

      GLFW/glfw3.h
      
      ;glm/glm.hpp
      imgui.h
      imgui_impl_glfw.h
      imgui_impl_opengl3.h)
     
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

       (do0
	(glfwInit)
	(glfwWindowHint GLFW_CONTEXT_VERSION_MAJOR 4)
	(glfwWindowHint GLFW_CONTEXT_VERSION_MINOR 5)
	(glfwWindowHint GLFW_OPENGL_PROFILE GLFW_OPENGL_CORE_PROFILE)

	(let ((window (glfwCreateWindow 800 600
					(string "v4l")
					nullptr nullptr)))
	  (unless window
	    ,(lprint :msg "Error creating glfw window")
	    (return -1)))

	(glfwMakeContextCurrent window)
	(glfwSwapInterval 1)

	(unless (gladLoaderLoadGL)
	  ,(lprint :msg "Error initializing glad")
	  (return -2))

	(do0
	 ,(lprint :msg "get extensions")
	 (do0 ;let ((ext (glGetString GL_EXTENSIONS)))
	   (when (space auto (setf ext (glGetString GL_EXTENSIONS))
			(!= nullptr ext))
	     (let ((extstr (std--string ("reinterpret_cast<const char*>" ext))))
	       ,(lprint :msg "extensions"
			:vars `(extstr)
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
	 #+more ,(lprint :msg "Compile shader")
	 (let ((success 0)
	       
	       (vertexShader (glCreateShader GL_VERTEX_SHADER)))
	   (glShaderSource vertexShader 1 &vertexShaderSrc nullptr)
	   (glCompileShader vertexShader)

	   (glGetShaderiv vertexShader GL_COMPILE_STATUS &success)
	   (unless success
	     #+more (let ((n 512) (infoLog (std--vector<char> n)))
	       (glGetShaderInfoLog vertexShader n nullptr (infoLog.data) )
	       ,(lprint :msg "vertex shader compilation failed"
			:vars `((std--string (infoLog.begin)
					     (infoLog.end))))
	       )
	     (exit -1))
	   )
	 (let ((fragmentShader (glCreateShader GL_FRAGMENT_SHADER)))
	   (glShaderSource fragmentShader 1 &fragmentShaderSrc nullptr)
	   (glCompileShader fragmentShader)

	   (glGetShaderiv fragmentShader GL_COMPILE_STATUS &success)
	   (unless success
	     #+more (let ((n 512) (infoLog (std--vector<char> n)))
	       (glGetShaderInfoLog fragmentShader n nullptr (infoLog.data) )
	       ,(lprint :msg "fragment shader compilation failed"
			:vars `((std--string (infoLog.begin)
					     (infoLog.end))))
	       )
	     (exit -1))
	   )
	 (let ((program (glCreateProgram )))
	   (glAttachShader program vertexShader)
	   (glAttachShader program fragmentShader)
	   (glLinkProgram program)
	   (glGetProgramiv program GL_LINK_STATUS &success)
	   (unless success
	     #+more (let ((n 512) (infoLog (std--vector<char> n)))
	       (glGetShaderInfoLog program n nullptr (infoLog.data) )
	       ,(lprint :msg "shader linking failed"
			:vars `((std--string (infoLog.begin)
					     (infoLog.end))))
	       )
	     (exit -1))
	   )
	 (glDetachShader program vertexShader)
	 (glDetachShader program fragmentShader))
	
	
	)

       (do0
	(glUseProgram program)
	(glClearColor 1 1 1 1))

       
       
       #+nil (let ((texture (GLuint 0)))
	 (glGenTextures 1 &texture)
	 (glBindTexture GL_TEXTURE_2D texture)
	 
	 (glTexParameteri GL_TEXTURE_2D GL_TEXTURE_MIN_FILTER GL_LINEAR)
	 (glTexParameteri GL_TEXTURE_2D GL_TEXTURE_MAG_FILTER GL_LINEAR))
       
       (while (!glfwWindowShouldClose window)
	      (glfwPollEvents)
					;(glDrawElements GL_TRIANGLES 6 GL_UNSIGNED_INT nullptr)
	      (ImGui_ImplOpenGL3_NewFrame)
	      (ImGui_ImplGlfw_NewFrame)
	      (ImGui--NewFrame)
	      #+nil (cap.getFrame
	       (lambda (data size)
		 (declare (type void* data)
			  (type size_t size)
			  (capture "&"))
		 (glBindTexture GL_TEXTURE_2D texture)
		 (glTexImage2D GL_TEXTURE_2D
			       0
			       GL_RGBA
			       w h
			       0
			       GL_RG
			       GL_UNSIGNED_BYTE
			       data
			       )
		 (ImGui--Begin (string "camera feed"))
		 (ImGui--Image (reinterpret_cast<void*> (static_cast<intptr_t> texture))
			       (ImVec2 w h))
		 (ImGui--End)
		 
		 ))

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
   :tidy t))

