(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

;; https://github.com/ocornut/imgui/blob/master/examples/example_sdl2_opengl3/main.cpp
(progn
 (setf *features* (set-difference *features* (list :more
						   :glad)))
 (setf *features* (set-exclusive-or *features* (list :more
					;:glad
						     ))))

(progn
  (progn
    (defparameter *source-dir* #P"example/136_sdl_gear_designer/source01/src/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*)))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")
  
  (let* ((name `Gear)
	 (members `((radius :type double :param t)
		 )))
    
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
		       )
     :implementation-preamble
     `(do0
       (include<> 
	;fstream
	iostream
	;vector
	;string
	;cstring
	stdexcept))
     :code `(do0
	     (defclass ,name ()
	       "public:"
	       (defmethod ,name (,@(remove-if #'null
				    (loop for e in members
					  collect
					  (destructuring-bind (name &key type param (initform 0)) e
					    (let ((nname (intern
							  (string-upcase
							   (cl-change-case:snake-case (format nil "~a" name))))))
					      (when param
						nname))))))
		 (declare
		  ,@(remove-if #'null
			       (loop for e in members
				     collect
				     (destructuring-bind (name &key type param (initform 0)) e
				       (let ((nname (intern
						     (string-upcase
						      (cl-change-case:snake-case (format nil "~a" name))))))
					 (when param
					   
					   `(type ,type ,nname))))))
		  (construct
		   ,@(remove-if #'null
				(loop for e in members
				      collect
				      (destructuring-bind (name &key type param (initform 0)) e
					(let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					      (nname_ (format nil "~a_"
							      (cl-change-case:snake-case (format nil "~a" name)))))
					  (cond
					    (param
					     `(,nname_ ,nname))
					    (initform
					     `(,nname_ ,initform)))))))
		   )
		  (explicit)	    
		  (values :constructor))
		 
		 #+nil (throw (std--runtime_error (+ (string "opening video device failed")
					       #+more (std--string (std--strerror errno))))))

	       "private:"
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (name &key type param (initform 0)) e
				    (let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					  (nname_ (format nil "~a_" (cl-change-case:snake-case (format nil "~a" name)))))
				      `(space ,type ,nname_)))))))))

  
  
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
   (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     
     (include<>
      iostream
      ;string
					;complex
      ;vector
					;algorithm
      
					;chrono
      ;thread
      
      ;filesystem
      ;unistd.h
      ;cstdlib

      ;cmath
      

      )

     #+glad
     (include<>
      glad/gl.h)
     
     (include<>

      ;GLFW/glfw3.h
      
      ;glm/glm.hpp
      imgui.h 
      imgui_impl_sdl2.h
      imgui_impl_opengl3.h
      )
     (include<>
      SDL.h
      SDL_opengl.h
      )
     #+nil
     (do0
      (setf "const char *vertexShaderSrc"
	    (string-r
	     ,(emit-c :code `(do0
			      "#version 450"
			      (space layout (paren (= location 0)) in vec2 aPos)
			      (defun main ()
				(setf gl_Position (vec4 aPos 1 1))))
		      :omit-redundant-parentheses t)))

      (setf "const char *fragmentShaderSrc"
	    (string-r
	     ,(emit-c :code `(do0
			      "#version 450"
			      (space layout (paren (= location 0)) out vec4 outColor)
			      (defun main ()
				(setf outColor (vec4 1 0 0 1))))
		      :omit-redundant-parentheses t))))

     #+nil (do0
      #+more
      (defun message_callback (source type id severity length message user_param)
	(declare (type GLenum source type severity)
		 (type GLuint id)
		 (type GLsizei length)
		 (type "GLchar const *" message)
		 (type "void const *" user_param))
	,(lprint :msg "gl"
		 :vars `(source type id severity message))))
					     
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
       		(type char** argv))

       (when (!= 0 (SDL_Init (or SDL_INIT_VIDEO
				     SDL_INIT_TIMER
				     ;SDL_INIT_GAMECONTROLLER
				     ) ))
	 ,(lprint :msg "Error"
		  :vars `((SDL_GetError)))
	 (return -1))
       (let ((glsl_version (string "#version 130")))
	 ,@(loop for e in `((:key context-flags :value 0)
			    (:key context-profile-mask :value SDL_GL_CONTEXT_PROFILE_CORE)
			    (:key context-major-version :value 3)
			    (:key context-minor-version :value 0)
			    (:key doublebuffer :value 1)
			    (:key depth-size :value 24)
			    (:key stencil-size :value 8)
			    ;(:key :value)
			    )
		 collect
		 (destructuring-bind (&key key value) e
		   `(SDL_GL_SetAttribute
		     ,(cl-change-case:constant-case (format nil "sdl-gl-~a" key))
		     ,value)))
	 (SDL_SetHint SDL_HINT_IME_SHOW_UI (string "1")))

       (let ((*window (SDL_CreateWindow
		       (string "imgui_sdl2_bullet_gears_designer")
		       SDL_WINDOWPOS_CENTERED
		       SDL_WINDOWPOS_CENTERED
		       1280
		       720
		       (or SDL_WINDOW_OPENGL
			       SDL_WINDOW_RESIZABLE
			       SDL_WINDOW_ALLOW_HIGHDPI
			       )
		       ))
	     
	     )
	 (unless window
	    (throw (std--runtime_error (string "Error creating GL window"))))
	 (let ((gl_context (SDL_GL_CreateContext window))))
	 (SDL_GL_MakeCurrent window gl_context)
	 (SDL_GL_SetSwapInterval 1))

       #+glad
       (do0
	(unless (gladLoaderLoadGL)
	  (throw (std--runtime_error (string "Error initializing glad")))
	  (do0
	   ,(lprint :msg "get extensions")
	   (let ((ext (glGetString GL_EXTENSIONS)))
	     (unless (== nullptr ext)
	       (let ((extstr (std--string ("reinterpret_cast<const char*>" ext))))
		 ,(lprint :msg "extensions"
			  :vars `(extstr)
			  )))))
	  ))

       (do0
	 (IMGUI_CHECKVERSION)
	 (ImGui--CreateContext)
	 (let ((*io (ref (ImGui--GetIO)))))
	 (setf io->ConfigFlags (or io->ConfigFlags
				      ImGuiConfigFlags_NavEnableKeyboard))
	 (ImGui--StyleColorsDark)
	 (ImGui_ImplSDL2_InitForOpenGL window gl_context)
	 (ImGui_ImplOpenGL3_Init glsl_version)
	 )

       (do0 (glEnable GL_CULL_FACE)
	     ; #+more (glEnable GL_DEBUG_OUTPUT)
	      ;#+more (glDebugMessageCallback message_callback nullptr)
	      )
       
       #+nil 
       (do0
	

	(do0
	 )

	(do0
	 #+more ,(lprint :msg "Compile shader")
	 (let ((success 0)
	       
	       (vertexShader (glCreateShader GL_VERTEX_SHADER)))
	   (glShaderSource vertexShader 1 &vertexShaderSrc 0)
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
	   (glShaderSource fragmentShader 1 &fragmentShaderSrc 0)
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

       #+nil 
       (do0
	#+more
	(do0 ,(lprint :msg "Create vertex array and buffers")
	     (comments "TBD")) 

	(glUseProgram program)
	(glClearColor 1 1 1 1)

	
	)

       (let ((done false)))
       (while !done
	      (let ((event (SDL_Event))))
	      (while (SDL_PollEvent &event)
		     (ImGui_ImplSDL2_ProcessEvent &event)
		     (when (== 
			    SDL_QUIT
			    event.type)
		       (setf done true))
		     (when (logand (== SDL_WINDOWEVENT event.type)
				   (== SDL_WINDOWEVENT_CLOSE event.window.event)
				   (== event.window.windowID (SDL_GetWindowID window)))
		       (setf done true)))

	      (do0
	       (ImGui_ImplOpenGL3_NewFrame)
	       (ImGui_ImplSDL2_NewFrame)
	       (ImGui--NewFrame))

	      (do0
	       (let ((show_demo true))
		 (declare (type "static bool" show_demo))
		 (when show_demo
		   (ImGui--ShowDemoWindow &show_demo))))
	      (do0
	       (ImGui--Render)
	       (glViewport 0 0 (static_cast<int> io->DisplaySize.x)
			   (static_cast<int> io->DisplaySize.y))
	       (glClearColor 0s0 0s0 0s0 1s0)
	       (glClear GL_COLOR_BUFFER_BIT)
	       (ImGui_ImplOpenGL3_RenderDrawData (ImGui--GetDrawData))
	       (SDL_GL_SwapWindow window)))
       
       #+nil 
       (handler-case
	   (let ((cap (V4L2Capture  (string "/dev/video0")
				    3)))
	     (let ((w ;320 ;
		      1280
		      )
		   (h ;180 ;
		      720
		      ))
	      (cap.setupFormat w h
			       ;V4L2_PIX_FMT_RGB24
			       V4L2_PIX_FMT_YUYV
			       ;V4L2_PIX_FMT_YUV420
			       ))
	     (cap.startCapturing)

	     (do0
	      (let ((texture (GLuint 0)))
		(glGenTextures 1 &texture)
		(glBindTexture GL_TEXTURE_2D texture)
		
		(glTexParameteri GL_TEXTURE_2D GL_TEXTURE_MIN_FILTER GL_LINEAR)
		(glTexParameteri GL_TEXTURE_2D GL_TEXTURE_MAG_FILTER GL_LINEAR)))
	     
	     (while (!glfwWindowShouldClose window)
		    (glfwPollEvents)
					;(glDrawElements GL_TRIANGLES 6 GL_UNSIGNED_INT nullptr)
		    (ImGui_ImplOpenGL3_NewFrame)
		    (ImGui_ImplGlfw_NewFrame)
		    (ImGui--NewFrame)
		    (cap.getFrame
		     (lambda (data size)
		       (declare (type void* data)
				(type size_t size)
				(capture "&"))
		       #+nil (do0 (glPixelStorei GL_UNPACK_ALIGNMENT 1)
			    (glPixelStorei GL_UNPACK_ROW_LENGTH (/ w 2))
			    (glPixelStorei GL_UNPACK_SKIP_PIXELS 0)
			    (glPixelStorei GL_UNPACK_SKIP_ROWS 0))
		       
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

		    (space static bool (setf showDemo false))
		    (ImGui--ShowDemoWindow &showDemo)
		    (ImGui--Render)
		    (ImGui_ImplOpenGL3_RenderDrawData (ImGui--GetDrawData))
		    (glfwSwapBuffers window)
		    (glClear GL_COLOR_BUFFER_BIT)
		    (std--this_thread--sleep_for (std--chrono--milliseconds 16))
		    )
	     
	     
	     #+nil (cap.getFrame (+ (string "/dev/shm/frame_")
			      (std--to_string i)
			      (string ".ppm")))
	     (cap.stopCapturing)
	     
	     )
	 ("const std::runtime_error&" (e)
	   #+more ,(lprint :msg "error"
		    :vars `((e.what)))
	   (return 1)))

       
       (do0
	(ImGui_ImplOpenGL3_Shutdown)
	(ImGui_ImplSDL2_Shutdown)
	(ImGui--DestroyContext)
	(SDL_GL_DeleteContext gl_context)
	(SDL_Quit))
       (return 0)))
   :omit-parens t
   :format t
   :tidy t))

