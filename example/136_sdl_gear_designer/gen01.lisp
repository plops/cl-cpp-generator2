(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  ;; https://github.com/ocornut/imgui/blob/master/examples/example_sdl2_opengl3/main.cpp
  ;; sudo emerge -av box2d
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

  
  
  (let* ((name `Physics)
	 (members `(
		    (timeStep :type float :initform ,(/ 1s0 60s0))
		    (velocityIterations :type int :initform 6)
		    (positionIterations :type int :initform 2)
		    (gravity :type b2Vec2 :initform (b2Vec2 0s0 -10s0))
		    (world :type b2World :initform (b2World gravity_))
		    (body :type b2Body* :initform nullptr )
		)))
    
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include<> box2d/box2d.h)
			)
     :implementation-preamble
     `(do0
       (include<> 
					;fstream
	iostream
					;vector
					;string
					;cstring
	stdexcept)
       (include<> box2d/box2d.h))
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
		  (values :constructor)
		  )
		 
		 (comments "https://github.com/erincatto/box2d/blob/main/unit-test/hello_world.cpp")
		 
		 


		 		 
		 (do0
		  
		  (let (
			(groundBodyDef (b2BodyDef)))
		    
		    (groundBodyDef.position.Set 0s0 -10s0)
		    (let ((groundBody (world_.CreateBody &groundBodyDef))))
		    (let ((groundBox (b2PolygonShape)))
		      (groundBox.SetAsBox 50s0 10s0)
		      )
		    (groundBody->CreateFixture &groundBox 0s0)
		    (let ((bodyDef (b2BodyDef)))
		      (setf bodyDef.type b2_dynamicBody)
		      (bodyDef.position.Set 0s0 4s0)
		      
		      )
		    (setf body_ (world_.CreateBody &bodyDef))
		    
		    (let ((dynamicBox (b2PolygonShape)))
		      (dynamicBox.SetAsBox 1s0 1s0))
		    (let ((fixtureDef (b2FixtureDef)))
		      (setf fixtureDef.shape &dynamicBox
			    fixtureDef.density 1s0
			    fixtureDef.friction .3s0))
		    (body_->CreateFixture &fixtureDef)
		    
		    ))
		 
		 
		 #+nil (throw (std--runtime_error (+ (string "opening video device failed")
						     #+more (std--string (std--strerror errno))))))
	       (defmethod Step ()
		 (declare (values "std::tuple<float,float,float>"))
		 (do0 (world_.Step time_step_ velocity_iterations_ position_iterations_)
		      (let (( position (body_->GetPosition))))
		      (let (( angle (body_->GetAngle))))
		      ,(lprint :vars `(position.x position.y angle))
		      (return (std--make_tuple position.x
					       position.y
					       angle)))
		 )

	       "private:"
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (name &key type param (initform 0)) e
				    (let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					  (nname_ (format nil "~a_" (cl-change-case:snake-case (format nil "~a" name)))))
				      `(space ,type ,nname_)))))))))
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
      memory
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
      
      unordered_map
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
					;(include<> box2d/box2d.h)
     (include "Physics.h")
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
     (defclass+ GuiException "public std::runtime_error"
       "public:"
       "using runtime_error::runtime_error;"
					;"explicit"
       #+nil
       (defmethod GuiException (msg)
	 (declare (explicit)
		  (type "const std::string&" msg)
		  (values :constructor)
		  (construct (std--runtime_error msg)))))

     (let ((slider_factory
	     (lambda ()
	       (declare (capture ""))
	       
	       (let ((values ("std::unordered_map<std::string, float>")))
		 (declare (type "static auto" values))
		 )
	       (let ((make_slider
		       (lambda (label)
			 (declare (type "const std::string&" label)
				  ;(capture "&values")
				  )
			 (if (!values.contains label)
			     (do0
			      ,(lprint :msg "make_slider init"
				       :vars `(label))
			      (setf (aref values label) 100s0))
			     #+nil (do0
				    ,(lprint :msg "make_slider repeated call"
					     :vars `(label))))

			 
			 (return (lambda ()
				   (declare (capture 
						     label
						    ; "&values"
						     ))
				   (return (aref values label))))))))
	       (let ((draw_all_sliders
		       (lambda ()
			 ;(declare (capture "&values"))
			 #+nil ,(lprint :msg "draw_all_sliders"
					:vars `((values.size)))
			 (ImGui--Begin (string "all-sliders"))
			 (for-range ((bracket key value) values)
				    (ImGui--SliderFloat (key.c_str)
							(ref (aref values key))
							100s0
							300s0))
			 (ImGui--End)))))
	       (return (std--make_tuple make_slider
					draw_all_sliders))))))
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
       		(type char** argv))
       ,(lprint :msg "main entry point" :vars `(argc (aref argv 0)))
       (let ((gl_context nullptr))
	 (declare (type void* gl_context)))

       (let ((set_gl_attributes (lambda ()
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
				  (SDL_SetHint SDL_HINT_IME_SHOW_UI (string "1"))
				  
				  ))))
       
       (let ((init_gl
	       (lambda (&gl_context_)
		 (when (!= 0 (SDL_Init (or SDL_INIT_VIDEO
					   SDL_INIT_TIMER
					;SDL_INIT_GAMECONTROLLER
					   ) ))
		   
		   ,(lprint :msg "Error"
			    :vars `((SDL_GetError)))
		   (throw (GuiException (string "Error in SDL_Init."))))
		 

		 (set_gl_attributes)
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
		     (throw (GuiException (string "Error creating GL window"))))
		   (setf gl_context_ (SDL_GL_CreateContext window))
		   (SDL_GL_MakeCurrent window gl_context_)
		   (SDL_GL_SetSwapInterval 1))
		 (do0 (glEnable GL_CULL_FACE)
					; #+more (glEnable GL_DEBUG_OUTPUT)
					;#+more (glDebugMessageCallback message_callback nullptr)
		      )

		 
		 (do0
		  #+glad
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
		 
		 
		 


		 #+nil
		 (do0
		  

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
		 
		 (return window)))))
       
       

       
       (let ((init_imgui (lambda (window_ gl_context_)
			   (do0
			    ,(lprint :msg "init_imgui")
			    (IMGUI_CHECKVERSION)
			    (ImGui--CreateContext)
			    (let ((*io (ref (ImGui--GetIO)))))
			    (setf io->ConfigFlags (or io->ConfigFlags
						      ImGuiConfigFlags_NavEnableKeyboard))
			    (ImGui--StyleColorsDark)
			    (ImGui_ImplSDL2_InitForOpenGL window_ gl_context_)
			    (let ((glsl_version (string "#version 130"))))
			    (ImGui_ImplOpenGL3_Init glsl_version)
			    )))))
       

       

       #+nil 
       (let ((texture (GLuint 0)))
	 (glGenTextures 1 &texture)
	 (glBindTexture GL_TEXTURE_2D texture)
	 
	 (glTexParameteri GL_TEXTURE_2D GL_TEXTURE_MIN_FILTER GL_LINEAR)
	 (glTexParameteri GL_TEXTURE_2D GL_TEXTURE_MAG_FILTER GL_LINEAR))


       (let ((handle_events
	       (lambda (*window_ *done_)
					;(declare (type auto& done_))
		 (do0 (let ((event (SDL_Event))))
		      (while (SDL_PollEvent &event)
			     (ImGui_ImplSDL2_ProcessEvent &event)
			     (when (== 
				    SDL_QUIT
				    event.type)
			       (setf *done_ true))
			     (when (logand (== SDL_WINDOWEVENT event.type)
					   (== SDL_WINDOWEVENT_CLOSE event.window.event)
					   (== event.window.windowID (SDL_GetWindowID window_)))
			       (setf *done_ true)))))
	       )))
       (let ((new_frame (lambda ()
			  (do0
			   (ImGui_ImplOpenGL3_NewFrame)
			   (ImGui_ImplSDL2_NewFrame)
			   (ImGui--NewFrame))))))

       (let ((demo_window
	       (lambda ()
		 (do0
		  (let ((show_demo true))
		    (declare (type "static bool" show_demo))
		    (when show_demo
		      (ImGui--ShowDemoWindow &show_demo)))))
	       )))

       (let ((swap (lambda (*window)
		     (do0
		      (ImGui--Render)
		      (let ((const*io (ref (ImGui--GetIO)))))
		      (glViewport 0 0 (static_cast<int> io->DisplaySize.x)
				  (static_cast<int> io->DisplaySize.y))
		      (glClearColor 0s0 0s0 0s0 1s0)
		      (glClear GL_COLOR_BUFFER_BIT)
		      (ImGui_ImplOpenGL3_RenderDrawData (ImGui--GetDrawData))
		      (SDL_GL_SwapWindow window))))))
       (let ((destroy_gl (lambda (gl_context_)
			   (do0
			    ,(lprint :msg "destroy_gl")
			    (ImGui_ImplOpenGL3_Shutdown)
			    (ImGui_ImplSDL2_Shutdown)
			    (ImGui--DestroyContext)
			    (SDL_GL_DeleteContext gl_context_)
			    (SDL_Quit))))))
       #+nil 
       (let ((widget_slider (lambda ()
			      (let ((value 100s0))
				(declare (type "static float" value)))
			      (ImGui--Begin (string "slider"))
			      (when (ImGui--SliderFloat
				     (string "slider")
				     &value
				     100s0 300s0))
			      (ImGui--End)
			      (return value)))))
       (handler-case
	   (do0
	    (let ((*window (init_gl gl_context))))
	    (init_imgui window gl_context)
	    
	    (let ((physics (std--make_unique<Physics>))))
	    
	    (let (((bracket make_slider
			    draw_all_sliders)
		    (slider_factory))))
	    
	    
	    (let ((done false)))
	    ,(lprint :msg "start gui loop")
	    #+nil
	    (do0
	     (let (( slider1 (make_slider (string "circle_rad")))))
	     (let (( slider2 (make_slider (string "scale"))))))

	    (while
	     !done
	     (handle_events window &done)
	     (new_frame )
	     
	     (draw_all_sliders)
	     
	     (let (((bracket px py angle) (physics->Step))
		   )
	       (let ((draw
		       (ImGui--GetBackgroundDrawList))
		     (rad 100s0)
		     (ppx (* 100 (+ 400 px)))
		     (ppy (* 100 (+ 400  py)))
		     (sx (sin angle))
		     (sy (cos angle)))
		 (draw->AddLine (ImVec2 ppx ppy)
				(ImVec2 (+ ppx (* rad sx))
					(+ ppy (* rad sy)))
				(ImGui--GetColorU32 ImGuiCol_Text)
				4s0)
		 (let ((scale
			 ((make_slider (string "scale")))
					;30s0
					;(slider2)
			 ))
		   )
		 (let ((circle_rad ;(slider1)
			 ((make_slider (string "circle_rad")))
					; (widget_slider)
				   )
		       (circum (* 2 std--numbers--pi_v<float> ; ,(coerce  pi 'single-float)
				  circle_rad
				  )
			       )
		       (num_segments (std--max 7 (static_cast<int> (ceil (/ circum 5s0)))))))
		 (draw->AddCircleFilled
		  (ImVec2 (+ 300 (* scale px)) (+ 300 (* scale py)))
		  circle_rad
		  (ImGui--GetColorU32 ImGuiCol_Separator)
		  num_segments
		  )))
	     
	     (demo_window)
	     (swap window)
	     ))
	 ("const std::runtime_error&" (e)
	   #+more ,(lprint :msg "error"
			   :vars `((e.what)))
	   (destroy_gl gl_context)
	   (return 1)))
       (destroy_gl gl_context)
       (return 0)))
   :omit-parens t
   :format t
   :tidy nil))

