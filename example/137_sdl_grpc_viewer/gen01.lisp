(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more)))
  (setf *features* (set-exclusive-or *features* (list :more))))

(let ((l-proto `((:name clear-color
			:request ((red float)
				  (green float)
				  (blue float)
				  (alpha float))
			:reply ((success bool)
				(elapsed_ms float))))))
  (defparameter *source-dir* #P"example/137_sdl_grpc_viewer/source01/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames "../proto/gl.proto"
		     *source-dir*))
   `(do0
     (setf syntax (string "proto3"))
     "package glproto;"
     
     (space-n service View
	      "{"
	      ,@(loop for e in l-proto
		      collect
		      (destructuring-bind (&key name request reply) e
			(let ((function-name (cl-change-case:pascal-case (format nil "~a" name)))
			      (reply-name (cl-change-case:pascal-case (format nil "~a-reply" name)))
			      (request-name (cl-change-case:pascal-case (format nil "~a-request" name))))
			  `(space rpc (,function-name ,request-name) (returns ,reply-name) "{}"))))
	      "}")
     ,@(loop for e in l-proto
		      appending
		      (destructuring-bind (&key name request reply) e
			(let (; (function-name (cl-change-case:pascal-case (format nil "~a" name)))
			      (reply-name (cl-change-case:pascal-case (format nil "~a-reply" name)))
			      (request-name (cl-change-case:pascal-case (format nil "~a-request" name))))
			  `(
			    (space-n message ,request-name
				     "{"
				     ,@(loop for (var type) in request
					     and i from 1
					     collect
					     `(space ,type (setf ,var ,i)))
				     "}")
			    (space-n message ,reply-name
				     "{"
				     ,@(loop for (var type) in reply
					     and i from 1
					     collect
					     `(space ,type (setf ,var ,i)))
				     "}")))))
     )
   :omit-parens t
   :format t
   :tidy nil)
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

      vector
					;algorithm
      
					;chrono
					;thread
      
					;filesystem
					;unistd.h
					;cstdlib


      cmath
      complex
      unordered_map
      format
      )
     
     (include<>
      imgui.h 
      imgui_impl_sdl2.h
      imgui_impl_opengl3.h
      )
     (include<>
      SDL.h
      SDL_opengl.h
      )

     (defclass+ GuiException "public std::runtime_error"
       "public:"
       "using runtime_error::runtime_error;")

     (let ((slider_factory
	     (lambda ()
	       (declare (capture ""))
	       
	       (let ((values ("std::unordered_map<std::string, float>")))
		 (declare (type "static auto" values))
		 )
	       (let ((make_slider
		       (lambda (label)
			 (declare (type "const std::string&" label))
			 (unless (values.contains label)
			   (do0
			    ,(lprint :msg "make_slider init"
				     :vars `(label))
			    (setf (aref values label) 100s0)))

			 
			 (return (lambda ()
				   (declare (capture 
					     label
					; "&values"
					     ))
				   (return (aref values label))))))))
	       (let ((draw_all_sliders
		       (lambda ()
			 (ImGui--Begin (string "all-sliders"))
			 (for-range ((bracket key value) values)
				    (ImGui--SliderFloat (key.c_str)
							(ref (aref values key))
							10s0
							600s0))
			 (ImGui--End)))))
	       (let ((lookup_slider
		       (lambda (label)
			 (when (values.contains label)
			   (return (aref values label)))
			 (throw (std--runtime_error (std--format  (string "label '{}' undefined.")
								  label)))
			 (return 0s0)))))
	       (return (std--make_tuple make_slider
					draw_all_sliders
					lookup_slider))))))
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
						     (:key stencil-size :value 8))
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
		 (do0 (glEnable GL_CULL_FACE))

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
			       (setf *done_ true))))))))
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
      
       (handler-case
	   (do0
	    (let ((*window (init_gl gl_context))))
	    (init_imgui window gl_context)
	    
	    
	    (let (((bracket make_slider
			    draw_all_sliders
			    lookup_slider)
		    (slider_factory))))
	    
	    
	    (let ((done false)))
	    ,(lprint :msg "start gui loop")
	  
	    (while
	     !done
	     (handle_events window &done)
	     (new_frame )
	     
	     (draw_all_sliders)
	     	     
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

