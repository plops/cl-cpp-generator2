(eval-when (:compile-toplevel :execute :load-toplevel)
     (ql:quickload "cl-cpp-generator2")
     (ql:quickload "cl-ppcre"))

(in-package :cl-cpp-generator2)



(setf *features* (union *features* `()))

(setf *features* (set-difference *features*
				 '()))




(progn

  (defparameter *source-dir* #P"example/17_glfw_lua/source/")
  
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  
  (progn
    ;; collect code that will be emitted in utils.h
    (defparameter *utils-code* nil)
    (defun emit-utils (&key code)
      (push code *utils-code*)
      " ")
    (defparameter *global-code* nil)
    (defun emit-global (&key code)
      (push code *global-code*)
      " "))
  (progn
  
    (defparameter *module-global-parameters* nil)
    (defparameter *module* nil)
    (defun logprint (msg &optional rest)
      `(do0
	" "
	#-nolog
	(do0
					;("std::setprecision" 3)
	 (<< "std::cout"
	     ;;"std::endl"
	     ("std::setw" 10)
	     (dot ("std::chrono::high_resolution_clock::now")
		  (time_since_epoch)
		  (count))
					;,(g `_start_time)
	     
	     (string " ")
	     ("std::this_thread::get_id")
	     (string " ")
	     __FILE__
	     (string ":")
	     __LINE__
	     (string " ")
	     __func__
	     (string " ")
	     (string ,msg)
	     (string " ")
	     ,@(loop for e in rest appending
		    `(("std::setw" 8)
					;("std::width" 8)
		      (string ,(format nil " ~a='" (emit-c :code e)))
		      ,e
		      (string "'")))
	     "std::endl"
	     "std::flush"))))
    (defun guard (code &key (debug t))
      `(do0
	#+lock-debug ,(if debug
			  (logprint (format nil "hold guard on ~a" (cl-cpp-generator2::emit-c :code code))
				    `())
			  "// no debug")
	#+eou ,(if debug
		   `(if (dot ,code ("std::mutex::try_lock"))
			(do0
			 (dot ,code (unlock)))
			(do0
			 ,(logprint (format nil "have to wait on ~a" (cl-cpp-generator2::emit-c :code code))
				    `())))
		   "// no debug")
	"// no debug"
	,(format nil
		 "std::lock_guard<std::mutex> guard(~a);"
		 (cl-cpp-generator2::emit-c :code code))))
    (defun lock (code &key (debug t))
      `(do0
	#+lock-debug ,(if debug
			  (logprint (format nil "hold lock on ~a" (cl-cpp-generator2::emit-c :code code))
				    `())
			  "// no debug")

	#+nil (if (dot ,code ("std::mutex::try_lock"))
		  (do0
		   (dot ,code (unlock)))
		  (do0
		   ,(logprint (format nil "have to wait on ~a" (cl-cpp-generator2::emit-c :code code))
			      `())))
	
	,(format nil
		 "std::unique_lock<std::mutex> lk(~a);"
		 
		 (cl-cpp-generator2::emit-c :code code))
	))

    
    (defun emit-globals (&key init)
      (let ((l `((_start_time ,(emit-c :code `(typeof (dot ("std::chrono::high_resolution_clock::now")
							   (time_since_epoch)
							   (count)))))
		 ,@(loop for e in *module-global-parameters* collect
			(destructuring-bind (&key name type default)
			    e
			  `(,name ,type))))))
	(if init
	    `(curly
	      ,@(remove-if
		 #'null
		 (loop for e in l collect
		      (destructuring-bind (name type &optional value) e
			(when value
			  `(= ,(format nil ".~a" (elt (cl-ppcre:split "\\[" (format nil "~a" name)) 0)) ,value))))))
	    `(do0
	      (include <chrono>)
	      (defstruct0 State
		  ,@(loop for e in l collect
 			 (destructuring-bind (name type &optional value) e
			   `(,name ,type))))))))
    (defun define-module (args)
      "each module will be written into a c file with module-name. the global-parameters the module will write to will be specified with their type in global-parameters. a file global.h will be written that contains the parameters that were defined in all modules. global parameters that are accessed read-only or have already been specified in another module need not occur in this list (but can). the prototypes of functions that are specified in a module are collected in functions.h. i think i can (ab)use gcc's warnings -Wmissing-declarations to generate this header. i split the code this way to reduce the amount of code that needs to be recompiled during iterative/interactive development. if the module-name contains vulkan, include vulkan headers. if it contains glfw, include glfw headers."
      (destructuring-bind (module-name global-parameters module-code) args
	(let ((header ()))
	  #+nil (format t "generate ~a~%" module-name)
	  (push `(do0
		  " "
		  (include "utils.h")
		  " "
		  (include "globals.h")
		  " "
		  (include "proto2.h")
		  " ")
		header)
	  (unless (cl-ppcre:scan "main" (string-downcase (format nil "~a" module-name)))
	    (push `(do0 "extern State state;")
		  header))
	  (push `(:name ,module-name :code (do0 ,@(reverse header) ,module-code))
		*module*))
	(loop for par in global-parameters do
	     (destructuring-bind (parameter-name
				  &key (direction 'in)
				  (type 'int)
				  (default nil)) par
	       (push `(:name ,parameter-name :type ,type :default ,default)
		     *module-global-parameters*))))))
  (defun g (arg)
    `(dot state ,arg))
  
  (define-module
      `(main ((_main_version :type "std::string")
	      (_code_repository :type "std::string")
	      (_code_generation_time :type "std::string")
	      (_cursor_xpos :type double)
	      (_cursor_ypos :type double))
	     (do0
	      (include <iostream>
		       <chrono>
		       <cstdio>
		       <cassert>
					;<unordered_map>
		       <string>
		       <fstream>
		       )

	      "using namespace std::chrono_literals;"
	      (let ((state ,(emit-globals :init t)))
		(declare (type "State" state)))


	      (do0
	      
	       (defun mainLoop ()
		 ,(logprint "mainLoop" `())
		 (while (not (glfwWindowShouldClose ,(g `_window)))
		   (glfwPollEvents)
		   (glfwGetCursorPos ,(g `_window)
				     (ref ,(g `_cursor_xpos))
				     (ref ,(g `_cursor_ypos)))
		   (drawFrame)
		   (drawGui)
		   (glfwSwapBuffers ,(g `_window))
		   )
		 ,(logprint "exit mainLoop" `()))
	       (defun run ()
		 ,(logprint "start run" `())
		 
		 (initWindow)
		 (initGui)
		 (initDraw)
		 (initLua)
		 
		 (mainLoop)
		 ,(logprint "finish run" `())))
	      
	      (defun main ()
		(declare (values int))
		(setf ,(g `_main_version)
		      (string ,(let ((str (with-output-to-string (s)
					    (sb-ext:run-program "/usr/bin/git" (list "rev-parse" "HEAD") :output s))))
				 (subseq str 0 (1- (length str))))))

		 (setf
               
               
                  ,(g `_code_repository) (string ,(format nil "http://10.1.10.5:30080/martin/py_wavelength_tune/"))
		  
                  ,(g `_code_generation_time) 
                  (string ,(multiple-value-bind
                                 (second minute hour date month year day-of-week dst-p tz)
                               (get-decoded-time)
                             (declare (ignorable dst-p))
                      (format nil "~2,'0d:~2,'0d:~2,'0d of ~a, ~d-~2,'0d-~2,'0d (GMT~@d)"
                              hour
                              minute
                              second
                              (nth day-of-week *day-names*)
                              year
                              month
                              date
                              (- tz)))))

		(setf ,(g `_start_time) (dot ("std::chrono::high_resolution_clock::now")
					     (time_since_epoch)
					     (count)))
		,(logprint "start main" `(,(g `_main_version)
					  ,(g `_code_repository)
					  ,(g `_code_generation_time)))
		
		(do0
		 (run)
		 ,(logprint "start cleanups" `())
		 
		(cleanupLua)
		 (cleanupDraw)
		 (cleanupGui)
		 (cleanupWindow)
		)
		,(logprint "end main" `())
		(return 0)))))

  
  
  
  (define-module
      `(glfw_window
	((_window :direction 'out :type GLFWwindow* )
	 (_framebufferResized :direction 'out :type bool))
	(do0
	 (defun keyCallback (window key scancode action mods)
	   (declare (type GLFWwindow* window)
		    (type int key scancode action mods))
	   (when (and (or (== key GLFW_KEY_ESCAPE)
			  (== key GLFW_KEY_Q))
		      (== action GLFW_PRESS))
	     (glfwSetWindowShouldClose window GLFW_TRUE))
	   )
	 (defun errorCallback (err description)
	   (declare (type int err)
		    (type "const char*" description))
	   ,(logprint "error" `(err description)))
	 (defun framebufferResizeCallback (window width height)
	   (declare (values "static void")
		    ;; static because glfw doesnt know how to call a member function with a this pointer
		    (type GLFWwindow* window)
		    (type int width height))
	   ,(logprint "resize" `(width height))
	   (let ((app ("(State*)" (glfwGetWindowUserPointer window))))
	     (setf app->_framebufferResized true)))
	 (defun initWindow ()
	   (declare (values void))
	   (when (glfwInit)
	     (do0
	      
	      (glfwSetErrorCallback errorCallback)
	      
	      (glfwWindowHint GLFW_CONTEXT_VERSION_MAJOR 2)
	      (glfwWindowHint GLFW_CONTEXT_VERSION_MINOR 0)
	      
	      (glfwWindowHint GLFW_RESIZABLE GLFW_TRUE)
	      (let ((label)
		    )
		(declare (type "std::stringstream" label))
		(<< label
		    (string "glfw lua example [")
		    ,(g `_code_generation_time)
		    (string "] git:")
		    "std::fixed"
		    ("std::setprecision" 3)
		    ,(g `_main_version)
		    ))
	      
	      (setf ,(g `_window) (glfwCreateWindow 930 930
						    (dot label (str) (c_str))
						   
						    NULL
						    NULL))
	      ,(logprint "initWindow" `(,(g `_window)
					 (glfwGetVersionString)))
	      ;; store this pointer to the instance for use in the callback
	      (glfwSetKeyCallback ,(g `_window) keyCallback)
	      (glfwSetWindowUserPointer ,(g `_window) (ref state))
	      (glfwSetFramebufferSizeCallback ,(g `_window)
					      framebufferResizeCallback)
	      (glfwMakeContextCurrent ,(g `_window))
	      (glfwSwapInterval 1)
	      )))
	 (defun cleanupWindow ()
	   (declare (values void))
	   (glfwDestroyWindow ,(g `_window))
	   (glfwTerminate)
	   ))))
  

  (define-module
      `(draw ((_fontTex :direction 'out :type GLuint)
	      (_draw_mutex :type "std::mutex")
	      (_draw_display_log :type bool)
	      ,@(loop for e in `(offset_x offset_y scale_x scale_y alpha marker_x)
		   collect
		     `(,(format nil "_draw_~a" e) :type float))
	      (_screen_offset :type glm--vec2)
	      (_screen_start_pan :type glm--vec2)
	      (_screen_scale :type float)
	      (_screen_grid :type float)
	      (_snapped_world_cursor :type glm--vec2)
	      (_temp_shape :type Line*)
	      (_shapes :type std--list<Line*>)
	      (_selected_node :type Node*))
	     (do0
	      ;,(emit-global :code `(include <glm/vec2.hpp>))
	      (include <algorithm>)

	      ,(emit-global
	       :code
	       `(do0
		 (include <list>)
		"// shapes for 2d cad"
		"struct Shape;"
		(space struct Node
		       (progn
			 (let ((parent)
			       (pos))
			   (declare (type Shape* parent)
				    (type glm--vec2 pos)))
			 ))
		(space struct Shape
		       (progn
			 (let ((nodes)
			       (max_nodes 0)
			       (world_scale)
			       (world_offset)
			       (color))
			   (declare (type std--vector<Node> nodes)
				    (type glm--vec4 color)
				    (type int max_nodes)
				    (type "static float" world_scale)
				    (type "static glm::vec2" world_offset)))
			 "virtual void draw() = 0;"
			 ;;"void draw_nodes();"

			 (defun hit_node (p)
			   (declare (type glm--vec2& p)
				    (values Node*))
			   "// p is in world space"
			   (for-range ((n :type auto&) nodes)
				      
				      (when (< (glm--distance
						p
						n.pos)
					       .01s0)
					(return &n))
				      (return nullptr)))
			 
			 (defun draw_nodes ()
			   (for-range (n nodes)
				      (let ((sx 0)
					    (sy 0))
					(world_to_screen n.pos sx sy)
					(glColor4f 1s0 .3s0 .3s0 1s0)
					(draw_circle sx sy 2))))
			 
			 (defun world_to_screen (v screeni screenj)
			   (declare (type "const glm::vec2 &" v)
				    (type "int&" screeni screenj))
			   (setf screeni (static_cast<int> (* (- (aref v 0)
								 (aref world_offset 0))
							      world_scale))
				 screenj (static_cast<int> (* (- (aref v 1)
								 (aref world_offset 1))
							      world_scale))))
			 (defun get_next_node (p)
			   (declare (type "const glm::vec2 &" p)
				    (values Node*))
			   (when (== (nodes.size)  max_nodes)
			     ;; shape is complete
			     (return nullptr))
			   (let ((n))
			     (declare (type Node n))
			     (setf n.parent this
				   n.pos p
				   )
			     (nodes.push_back n)
			     ;; we need to be careful with this pointer
			     ;; instance needs to reserve memory for the nodes vector
			     (return (ref (aref nodes (- (nodes.size)
							 1))))))))
		(space struct Line ":public" Shape
		     (progn
		       (space "Line()"
			      (progn
				(setf max_nodes 2)
				(nodes.reserve max_nodes)
				(setf color (glm--vec4 1s0 1s0 0s0 1s0))))
		       (defun draw ()
			 (let ((sx 0)
			       (sy 0)
			       (ex 0)
			       (ey 0))
			   (world_to_screen (dot (aref nodes 0) pos)
					    sx sy)
			   (world_to_screen (dot (aref nodes 1) pos)
					    ex ey)
			   (glColor4f ,@(loop for i below 4 collect
					     `(aref color ,i)))
			   (glBegin GL_LINES)
			   (glVertex2i sx sy)
			   (glVertex2i ex ey)
			   (glEnd)))))))

	      
	      
	      (do0
	       "// initialize static varibles"
	       (setf "float Shape::world_scale" 1s0
		     "glm::vec2 Shape::world_offset" (curly 0 0)))
	      
	      (defun uploadTex (image w h)
		(declare (type "const void*" image)
			 (type int w h))
		(glGenTextures 1 (ref ,(g `_fontTex)))
		(glBindTexture GL_TEXTURE_2D ,(g `_fontTex))
		(glTexParameteri GL_TEXTURE_2D GL_TEXTURE_MIN_FILTER GL_LINEAR)
		(glTexParameteri GL_TEXTURE_2D GL_TEXTURE_MAG_FILTER GL_LINEAR)
		(glTexImage2D GL_TEXTURE_2D 0 GL_RGBA w h 0 GL_RGBA GL_UNSIGNED_BYTE image))

	      (defun screen_width ()
		(declare (values int))
		(let ((width 0)
		       (height 0))
		     (declare (type int width height))
		     (glfwGetFramebufferSize ,(g `_window)
					     &width
					     &height)
		     (return width)))
	      (defun screen_height ()
		(declare (values int))
		(let ((width 0)
		       (height 0))
		     (declare (type int width height))
		     (glfwGetFramebufferSize ,(g `_window)
					     &width
					     &height)
		     (return height)))

	      (defun get_mouse_position ()
		(declare (values "glm::vec2"))
		(let ((x 0d0)
		      (y 0d0))
		  (glfwGetCursorPos ,(g `_window)
				    &x &y)
		  (return (glm--vec2 (curly (static_cast<float> x)
					    (static_cast<float> y))))))
	      (defun draw_circle (sx sy rad)
		(declare (type float sx sy rad))
		(glBegin GL_LINE_STRIP)
		,@(let ((n 13))
		    (loop for i below n collect
			 `(progn
			    (let ((arg ,(/ (* 1s0 i) (+ n 1)))
				 )
			     (glVertex2f (+ sx (* rad (sinf (* 2 M_PI arg))))
					 (+ sy (* rad (cosf (* 2 M_PI arg)))))))))
		(glEnd))
	      (defun initDraw ()
		(setf ,(g `_temp_shape) nullptr
		      ,(g `_selected_node) nullptr)
		(progn
		  ,(guard (g `_draw_mutex))
		  (setf ,(g `_draw_display_log) true)
		  ,@(loop for (e f) in `((offset_x -.03)
					 (offset_y -.44)
					 (scale_x .22s0)
					 (scale_y .23s0)
					 (alpha .19s0)
					 (marker_x 100s0))
		       collect
			 `(setf ,(g (format nil "_draw_~a" e)) ,f)))
					;(glEnable GL_TEXTURE_2D)
		#+nil (glEnable GL_DEPTH_TEST)
		(do0 (glHint GL_LINE_SMOOTH GL_NICEST)
		     (do0 (glEnable GL_BLEND)
			  (glBlendFunc GL_SRC_ALPHA
				       GL_ONE_MINUS_SRC_ALPHA)))
		(glClearColor 0 0 0 1)
		(setf ,(g `_framebufferResized) true)
		(setf ,(g `_screen_offset) (curly 0s0 0s0)
		      ,(g `_screen_start_pan) (curly 0s0 0s0)
		      ,(g `_screen_scale) 10s0
		      ,(g `_screen_grid) 1s0
		      
		      )

		(do0
		 "// default offset to middle of screen"
		 (setf ,(g `_screen_offset) (curly (/ (static_cast<float> (/ (screen_width) -2))
						      ,(g `_screen_scale))
						   (/ (static_cast<float> (/ (screen_height) -2))
						      ,(g `_screen_scale)))
		       )
                 )
		,(logprint "screen"
			   `(,@(loop for e in `((aref _screen_offset 0)
						(aref _screen_offset 1)
						(aref _screen_start_pan 0)
						(aref _screen_start_pan 1)
						_screen_scale
							       _screen_grid) collect
				    (g e))))
		)

	      (defun world_to_screen (v screeni screenj)
		(declare (type "const glm::vec2 &" v)
			 (type "int&" screeni screenj))
		(setf screeni (static_cast<int> (* (- (aref v 0)
						      (aref ,(g `_screen_offset) 0))
						   ,(g `_screen_scale)))
		      screenj (static_cast<int> (* (- (aref v 1)
						      (aref ,(g `_screen_offset) 1))
						   ,(g `_screen_scale)))))

	      (defun screen_to_world (screeni screenj v)
		(declare (type "glm::vec2 &" v)
			 (type "int" screeni screenj))
		
		(setf (aref v 0) (+ (/ (static_cast<float> screeni)
				       ,(g `_screen_scale))
				    (aref ,(g `_screen_offset) 0))
		      (aref v 1) (+ (/ (static_cast<float> screenj)
				       ,(g `_screen_scale))
				    (aref ,(g `_screen_offset) 1))
		      ))
	      
	      (defun cleanupDraw ()
		(glDeleteTextures 1 (ref ,(g `_fontTex))))
	      (defun drawFrame ()

		(when ,(g `_framebufferResized)
		  (do0
		   (setf ,(g `_framebufferResized) false)
		   (let ((width 0)
			 (height 0))
		     (declare (type int width height))
		     (while (or (== 0 width)
				(== 0 height))
		       (glfwGetFramebufferSize ,(g `_window)
					       &width
					       &height)
                       
		       (glViewport 0 0 width height)
		       
		       (do0 (glMatrixMode GL_PROJECTION)
			    (glPushMatrix)
			    (glLoadIdentity)
			    (glOrtho 0s0 width height 0s0 -1s0 1s0)
					;(glOrtho -1s0 1s0 -1s0 1s0 -1s0 1s0)
			    )

		       (do0
		 "// default offset to middle of screen"
		 (setf ,(g `_screen_offset) (curly (/ (static_cast<float> (/ (screen_width) -2))
						      ,(g `_screen_scale))
						   (/ (static_cast<float> (/ (screen_height) -2))
						      ,(g `_screen_scale)))
		       )
                 )
		       
		       (do0 (glMatrixMode GL_MODELVIEW)
			    (glPushMatrix)
			    (glLoadIdentity))))))

		
		(glClear (logior GL_COLOR_BUFFER_BIT
				 GL_DEPTH_BUFFER_BIT))

		(do0
		 (let ((mouse_state (glfwGetMouseButton ,(g `_window)
							GLFW_MOUSE_BUTTON_MIDDLE))
		       (old_mouse_state GLFW_RELEASE)
		       (mouse_pos (get_mouse_position)))
		   (declare (type "static int" old_mouse_state))
		   
		   (when (and (== mouse_state GLFW_PRESS) ;; new press
			      (== old_mouse_state GLFW_RELEASE))
		     ;,(logprint "left mouse is pressed")
		     (setf ,(g `_screen_start_pan) mouse_pos))

		   (when (and (== mouse_state GLFW_PRESS) ;; button is being held
			      (== old_mouse_state GLFW_PRESS))
		     ;;,(logprint "left mouse is held")
		     (decf ,(g `_screen_offset)
			   (/ (- mouse_pos ,(g `_screen_start_pan))
			      ,(g `_screen_scale)))
		     (setf ,(g `_screen_start_pan) mouse_pos))

		   (do0
		    ;; zoom
		    (let ((mouse_before_zoom (glm--vec2))
			  (zoom_speed .05s0))
		      (screen_to_world (static_cast<int> (aref mouse_pos 0))
				       (static_cast<int> (aref mouse_pos 1))
				       mouse_before_zoom)

		      (progn
		       (let ((key_state (glfwGetKey ,(g `_window)
						    GLFW_KEY_PERIOD)))
			 (when (== key_state GLFW_PRESS)
			   ;; zoom out with .
			   (setf ,(g `_screen_scale)
				 (* (- 1s0 zoom_speed)  ,(g `_screen_scale))))))
		      (progn
		       (let ((key_state (glfwGetKey ,(g `_window)
						    GLFW_KEY_COMMA)))
			 (when (== key_state GLFW_PRESS)
			   ;; zoom in with ,
			   (setf ,(g `_screen_scale)
				 (* (+ 1s0 zoom_speed) ,(g `_screen_scale))))))
		      (let ((mouse_after_zoom (glm--vec2)))
			(screen_to_world (static_cast<int> (aref mouse_pos 0))
				       (static_cast<int> (aref mouse_pos 1))
				       mouse_after_zoom)
			(incf ,(g `_screen_offset)
			      (- mouse_before_zoom
				 mouse_after_zoom))

			(do0
			 "// compute snapped world cursor"
			 ,@(loop for i below 2 collect
				`(setf (aref ,(g `_snapped_world_cursor ) ,i)
				       (floorf (* (+ .5s0 (aref mouse_after_zoom ,i))
						  ,(g `_screen_grid))))))
			))
		    )

		   (progn
		  "// draw line"
		  (let (;(selected_node nullptr)
			;(line nullptr)
			)
		    (declare (type "static Line*" line)
			     (type "static Node*" selected_node))
		   (let ((key_state (glfwGetKey ,(g `_window)
						GLFW_KEY_L)))
		     (when (== key_state GLFW_PRESS)
		       ;,(logprint "start line" `(key_state))
		       (setf ,(g `_temp_shape) (new (Line))
			     ,(g `_selected_node) (-> ,(g `_temp_shape)
						  (get_next_node ,(g `_snapped_world_cursor)))
			     ,(g `_selected_node) (-> ,(g `_temp_shape)
						  (get_next_node ,(g `_snapped_world_cursor))))
		       ))

		   (progn
		     ;; move node
		    (let ((key_state (glfwGetKey ,(g `_window)
						 GLFW_KEY_M)))
		      (when (== key_state GLFW_PRESS)
					
			(setf 
			      ,(g `_selected_node) nullptr
			      )
			(for-range ((shape :type auto&) ,(g `_shapes))
				   (setf ,(g `_selected_node)
					 (-> shape (hit_node ,(g `_snapped_world_cursor))))
				   (unless (== nullptr
					       ,(g `_selected_node))
				     break))
			)))

		   
		   (unless (== ,(g `_selected_node) nullptr)
		     (setf (-> ,(g `_selected_node) pos) ,(g `_snapped_world_cursor)))
		   (let ((left_mouse_button_state (glfwGetMouseButton ,(g `_window)
							GLFW_MOUSE_BUTTON_LEFT))
			 (old_left_mouse_button_state GLFW_RELEASE))
		     (declare (type "static int" old_left_mouse_button_state))
		     ;,(logprint "mouse" `(left_mouse_button_state old_left_mouse_button_state))
		     (when (and (== old_left_mouse_button_state GLFW_PRESS)
				(== left_mouse_button_state GLFW_RELEASE))
		       (unless (== nullptr ,(g `_temp_shape))
			(setf ,(g `_selected_node) (-> ,(g `_temp_shape)
						       (get_next_node ,(g `_snapped_world_cursor))))
			(when (== nullptr ,(g `_selected_node))
			  "//  shape is complete"
			  (setf (-> ,(g `_temp_shape) color)  (glm--vec4 1s0 1s0 1s0 1s0))
			  (dot ,(g `_shapes)
			       (push_back ,(g `_temp_shape)))
			  )))
		     ;; video Practical Polymorphism C++ 29:27
		     (setf old_left_mouse_button_state left_mouse_button_state))))
		   
		   (setf old_mouse_state mouse_state)))


		(do0
		 ;; get visible world
		 (let ((world_top_left (glm--vec2))
		       (world_bottom_right (glm--vec2)))
		   (screen_to_world 0 0 world_top_left)
		   (screen_to_world (screen_width)
				    (screen_height)
				    world_bottom_right)
		   ,@(loop for (edge op) in `((world_top_left floor)
					      (world_bottom_right ceil))
			append
			  (loop for i below 2 collect
			       `(setf (aref ,edge ,i)
				      (,op (aref ,edge ,i)))))
		   (do0
		    ;; world axes
		    (let ((sx 0)
			  (sy 0)
			  (ex 0)
			  (ey 0))
		      (do0 ;; y axis
		       (world_to_screen (curly 0 (aref world_top_left 1)) sx sy)
		      (world_to_screen (curly 0 (aref world_bottom_right 1)) ex ey)
		      
		      (glColor4f .8 .3 .3 1)
		      (glEnable GL_LINE_STIPPLE)
		       (glLineStipple 1 (hex #xF0F0))
		       (glBegin GL_LINES)
		       (glVertex2f sx sy)
		       (glVertex2f ex ey)
		       (glEnd)
		       ;(glLineStipple 1 (hex #xFFFF))
		       )
		      (do0 ;; x axis
		       (world_to_screen (curly (aref world_top_left 0) 0) sx sy)
		       (world_to_screen (curly (aref world_bottom_right 0) 0) ex ey)
		      
		       (glColor4f .8 .3 .3 1)
		       ;(glLineStipple 1 (hex #xF0F0))
		       (glBegin GL_LINES)
		       (glVertex2f sx sy)
		       (glVertex2f ex ey)
		       (glEnd)
		       
		       
		       )

		      (do0
		 ;; grid axes
		       (glColor4f .3 .3 .3 1)
		       (glLineStipple 1 (hex #xAAAA))
		 (glBegin GL_LINES)
		 
		 ,@(loop for i from -100 upto 100 collect
			;; parallel to x axis
			`(do0 (world_to_screen (curly (aref world_top_left 0) ,i) sx sy)
			      (world_to_screen (curly (aref world_bottom_right 0) ,i) ex ey)
			      (glVertex2f sx sy)
			      (glVertex2f ex ey)))
		 ,@(loop for i from -100 upto 100 collect
			;; parallel to y axis
			`(do0 (world_to_screen (curly ,i (aref world_top_left 1)) sx sy)
			      (world_to_screen (curly ,i (aref world_bottom_right 1)) ex ey)
		      

			      (glVertex2f sx sy)
			      (glVertex2f ex ey)))
		 
		 (glEnd)
		 (do0 (glLineStipple 1 (hex #xFFFF))
		      (glDisable GL_LINE_STIPPLE))

		 (do0
		  "// draw the geometric objects"
		  (setf "Shape::world_scale" ,(g `_screen_scale)
			"Shape::world_offset" ,(g `_screen_offset))
		  (for-range (shape ,(g `_shapes))
			     (do0
		     	      (-> shape (draw))
			      (-> shape (draw_nodes))))
		  (unless (== nullptr ,(g `_temp_shape))
		    (do0
		     
		     (-> ,(g `_temp_shape) (draw))
		     (-> ,(g `_temp_shape) (draw_nodes)))))

		 (do0
		  "// draw snapped cursor circle"
		  (world_to_screen ,(g `_snapped_world_cursor) sx sy)
		  (glColor3f 1s0 1s0 0s0)
		  (draw_circle sx sy 3))
		 )
		      )
		    )
		   )
	       )


		
		

		

		(let ((width 0)
		      (height 0))
		  (declare (type int width height))
		  (glfwGetFramebufferSize ,(g `_window)
					       &width
					       &height)
		  (do0 ;; mouse cursor
		   (glColor4f 1 1 1 1)
		   #+nil (do0
		    (glBegin GL_LINES)
		    (let ((x (* 2 (- (/ ,(g `_cursor_xpos)
					width)
				     .5)))
			  (y (* -2 (- (/ ,(g `_cursor_ypos)
					 height)
				      .5))))
		      (glVertex2d x -1)
		      (glVertex2d x 1)
		      (glVertex2d -1 y)
		      (glVertex2d 1 y))
		    (glEnd))
		   (do0
		    (glBegin GL_LINES)
		    (let ((x ,(g `_cursor_xpos))
			  (y ,(g `_cursor_ypos)))
		      (glVertex2d x 0)
		      (glVertex2d x width)
		      (glVertex2d 0 y)
		      (glVertex2d height y))
		    (glEnd))))))))

  
  (define-module
      `(gui ((_gui_mutex :type "std::mutex")
	     (_gui_request_diff_reset :type bool))
	    (do0
	     "// https://youtu.be/nVaQuNXueFw?t=317"
	     "// https://blog.conan.io/2019/06/26/An-introduction-to-the-Dear-ImGui-library.html"
	     (include "imgui/imgui.h"
		      "imgui/examples/imgui_impl_glfw.h"
		      "imgui/examples/imgui_impl_opengl2.h")
	     (include <algorithm>
		      <string>)
	     (include <iostream>
		      <fstream>)
	     (defun initGui ()
	       ,(logprint "initGui" '())
	       (IMGUI_CHECKVERSION)
	       ("ImGui::CreateContext")
	       
	       (ImGui_ImplGlfw_InitForOpenGL ,(g `_window)
					     true)
	       (ImGui_ImplOpenGL2_Init)
	       ("ImGui::StyleColorsDark"))
	     (defun cleanupGui ()
	       (ImGui_ImplOpenGL2_Shutdown)
	       (ImGui_ImplGlfw_Shutdown)
	       ("ImGui::DestroyContext"))
	     (defun get_FixedDeque (data idx)
	       (declare (type "void*" data)
			(type int idx )
			(values float))
	       (let ((data1 (reinterpret_cast<FixedDeque<120>*> data)))
		 (return (aref (aref data1 0) idx))))
	     
	     (defun drawGui ()
	       #+nil (<< "std::cout"
			 (string "g")
			 "std::flush")
	       
	       (ImGui_ImplOpenGL2_NewFrame)
	       (ImGui_ImplGlfw_NewFrame)
	       ("ImGui::NewFrame")
	       
	       (do0
		(ImGui--Begin (string "snapped_cursor"))
		(ImGui--Text (string "x: %04d y: %04d")
			     (static_cast<int> (aref ,(g `_snapped_world_cursor) 0))
			     (static_cast<int> (aref ,(g `_snapped_world_cursor) 1)))
		(ImGui--End))	       
	       
	       (let ((b true))
		 ("ImGui::ShowDemoWindow" &b))
	       ("ImGui::Render")
	       (ImGui_ImplOpenGL2_RenderDrawData
		("ImGui::GetDrawData"))
	       ))))

  (define-module
      `(lua ((_lua_state :type lua_State*))
	    (do0
	     "// Embedding Lua in C++ #1  https://www.youtube.com/watch?v=4l5HdmPoynw"
	      ;; https://stackoverflow.com/questions/35422215/problems-linking-to-lua-5-2-4-even-when-compiling-sources-inline
	      ;; every lua.h inclusion must be extern C!
	      (space "extern \"C\""
		     (progn
		       (include "lua/lua.h"
				"lua/lauxlib.h"
				"lua/lualib.h")))

	      
	      (defun checkLua (L res)
		(declare (type int res)
			 (type lua_State* L)
			 (values bool))
		(unless (== res LUA_OK)
		  (do0 ,(logprint "lua_error" `(res (lua_tostring L -1))))
		  (return false))
		(return true))

	      (defun lua_HostFunction (L)
		(declare 
			 (type lua_State* L)
			 (values int))
		(let ((a (static_cast<float> (lua_tonumber L 1))
			)
		      (b (static_cast<float> (lua_tonumber L 2))))
		  ,(logprint "HostFunction" `(a b))
		  (let ((c (* a b)))
		    (lua_pushnumber L c)))
		;; number of args going back to c++
		(return 1))
	      
	      (defun initLua ()
		(setf ,(g `_lua_state) (luaL_newstate))
		
		(let ((cmd (string "a = 7+11+math.sin(23.7)"))
		      (L ,(g `_lua_state))
		      )
		  (declare (type std--string cmd)
			   (type lua_State* L))
		  (luaL_openlibs L)
		  (lua_register L (string "HostFunction"
					  )
				lua_HostFunction)
		  (let ((res ;(luaL_dostring L (cmd.c_str))
			 (luaL_dofile L (string "init.lua"))
			 ))
		    (if (== res LUA_OK)
			(do0
			 (lua_getglobal L (string "a"))
			 (when (lua_isnumber L -1)
			   ,(logprint "lua_ok" `((static_cast<float> (lua_tonumber L -1)))))

			 (do0
			  (lua_getglobal L (string "DoAThing"))
			  (when (lua_isfunction L -1)
			    (lua_pushnumber L 3.5s0)
			    (lua_pushnumber L 7.1s0)
			    (when (checkLua L (lua_pcall L 2 1 0))
			      ,(logprint "lua" `((static_cast<float> (lua_tonumber L -1)))))))
			 )
			(do0 ,(logprint "lua_error" `(res (lua_tostring L -1))))))
		  )
		)
	      
	      (defun cleanupLua ()
		(lua_close ,(g `_lua_state))
		)
	      )))
  
  (progn
    (with-open-file (s (asdf:system-relative-pathname 'cl-cpp-generator2
						      (merge-pathnames #P"proto2.h"
								       *source-dir*))
		       :direction :output
		       :if-exists :supersede
		       :if-does-not-exist :create)
      (format s "#ifndef PROTO2_H~%#define PROTO2_H~%")
      		    
      (loop for e in (reverse *module*) and i from 0 do
	   (destructuring-bind (&key name code) e
	     
	     (let ((cuda (cl-ppcre:scan "cuda" (string-downcase (format nil "~a" name)))))
	       
	       (unless cuda
		 #+nil (progn (format t "emit function declarations for ~a~%" name)
			(emit-c :code code :hook-defun 
				#'(lambda (str)
				    (format t "~a~%" str))))
		 (emit-c :code code :hook-defun 
			 #'(lambda (str)
			     (format s "~a~%" str))))

	       #+nil (format t "emit cpp file for ~a~%" name)
	       (write-source (asdf:system-relative-pathname
			      'cl-cpp-generator2
			      (format nil
				      "~a/vis_~2,'0d_~a.~a"
				      *source-dir* i name
				      (if cuda
					  "cu"
					  "cpp")))
			     code))))
      (format s "#endif"))
    (write-source (asdf:system-relative-pathname
		   'cl-cpp-generator2
		   (merge-pathnames #P"utils.h"
				    *source-dir*))
		  `(do0
		    "#ifndef UTILS_H"
		    " "
		    "#define UTILS_H"
		    " "
		    (include <vector>
			     <array>
			     <iostream>
			     <iomanip>)
		    
		    " "
		    (do0
		     
		     " "
		     ,@(loop for e in (reverse *utils-code*) collect
			  e)
					;"#define length(a) (sizeof((a))/sizeof(*(a)))"
					;"#define max(a,b)  ({ __typeof__ (a) _a = (a);  __typeof__ (b) _b = (b);  _a > _b ? _a : _b; })"
					;"#define min(a,b)  ({ __typeof__ (a) _a = (a);  __typeof__ (b) _b = (b);  _a < _b ? _a : _b; })"
					;"#define max(a,b) ({ __auto_type _a = (a);  __auto_type _b = (b); _a > _b ? _a : _b; })"
					;"#define min(a,b) ({ __auto_type _a = (a);  __auto_type _b = (b); _a < _b ? _a : _b; })"
		     
		     " "
		     
		     )
		    " "
		    "#endif"
		    " "))
    (write-source (asdf:system-relative-pathname 'cl-cpp-generator2 (merge-pathnames
								     #P"globals.h"
								     *source-dir*))
		  `(do0
		    "#ifndef GLOBALS_H"
		    " "
		    "#define GLOBALS_H"
		    " "
		    #+glad (include <glad/gl.h>)
		    " "
		    (include <GLFW/glfw3.h>)
		    (space "extern \"C\""
			   (progn
			     (include "lua/lua.h"
				      )))

		    " "
					;(include <winsock2.h>)
		    " "
					;(include "SPinGW/serialport.h")
		    " "
		    (include <thread>
			     <mutex>
			     <queue>
			     <deque>
			     <map>
			     <string>
			     <condition_variable>
			     <complex>)
		    #+nil (include "sqlite/sqlite-preprocessed-3310100/sqlite3.h")
		    ;(include <sqlite3.h>)
		    " "

		    (include <glm/vec2.hpp>)
		    (include <glm/vec4.hpp>)
		    (include <glm/geometric.hpp>) ;; for distance
		    " "
		    (include "proto2.h")
		    " "
		    ,@(loop for e in (reverse *global-code*) collect
			 e)

		    
		    
		    (defstruct0 CommunicationTransaction
			,@(loop for e in `(start_loop_time tx_time rx_time)
			     collect
			       `(,e "long long int")
			       
			       )
		      (tx_message "std::string")
		      (rx_message "std::string"))

		    
		    
		    (do0
		     "template <typename T, int MaxLen>"
		     (defclass FixedDequeT "public std::deque<T>"
		       "// https://stackoverflow.com/questions/56334492/c-create-fixed-size-queue"
		       "public:"
		       (defun push_back (val)
			 (declare (type "const T&" val))
			 (when (== MaxLen (this->size))
			   (this->pop_front))
			 ("std::deque<T>::push_back" val))))
		    (do0
		     "template <typename T, int MaxLen>"
		     (defclass FixedDequeTM "public std::deque<T>"
		       "// https://stackoverflow.com/questions/56334492/c-create-fixed-size-queue"
		       
		       "public:"
		       (let ((mutex))
			 (declare (type "std::mutex" mutex)))
		       (defun push_back (val)
			 (declare (type "const T&" val))
			 (when (== MaxLen (this->size))
			   (this->pop_front))
			 ("std::deque<T>::push_back" val))))
		    (do0
		     "template <int MaxLen>"
		     (defclass FixedDeque "public std::deque<float>"
		       "// https://stackoverflow.com/questions/56334492/c-create-fixed-size-queue"
		       "public:"
		       (defun push_back (val)
			 (declare (type "const float&" val))
			 (when (== MaxLen (this->size))
			   #+nil (<< "std::cout"
				     (string "size") (this->size)
				     (string "back") (this->back)
				     "std::endl")
					;,(logprint (string "fixed deque is full" )`(this->size MaxLen))
			   (this->pop_front))
			 ("std::deque<float>::push_back" val))))
		    " "
		    (do0
		     "template <int MaxLen>"
		     (defclass FixedGuardedDeque ()
		       "private:"
		       (let ((mutex)
			     (deque))
			 (declare (type "std::mutex" mutex)
				  (type FixedDeque<MaxLen> deque)))
		       "public:"
		       (defun push_back (val)
			 (declare (type "const float&" val))
			 (progn
			   ,(guard 'mutex)
			   (deque.push_back val)))
		       (defun empty ()
			 (declare (values bool))
			 ,(guard 'mutex)
			 (return (deque.empty)))
		       (defun size ()
			 (declare (values size_t))
			 ,(guard 'mutex)
			 (return (deque.size)))
		       (defun back ()
			 (declare (values float))
			 ,(guard 'mutex)
			 (return (deque.back)))
		       (defun "operator[]" (n)
			 (declare (type size_t n)
				  (values float))
			 ,(guard 'mutex)
			 (return (aref deque n)))))
		    (do0
		     "template <int MaxLen>"
		     (defclass FixedGuardedWaitingDeque ()
		       "// https://baptiste-wicht.com/posts/2012/04/c11-concurrency-tutorial-advanced-locking-and-condition-variables.html"
		       "private:"
		       (let ((mutex)
			     (not_empty)
			     (deque))
			 (declare (type "std::mutex" mutex)
				  (type "std::condition_variable" not_empty)
				  (type FixedDeque<MaxLen> deque)))
		       "public:"
		       (defun push_back (val)
			 (declare (type "const float&" val))
			 (progn
			   ,(guard 'mutex)
			   (deque.push_back val))
			 (dot not_empty
			      (notify_one)))
		       (defun back ()
			 (declare (values float))
			 ,(lock 'mutex)
			 (while (== 0 (deque.size))
			   (dot not_empty (wait lk)))
			 (return (deque.back)))
		       (defun "operator[]" (n)
			 (declare (type size_t n)
				  (values float))
			 ,(lock 'mutex)
			 (while (== 0 (deque.size))
			   (dot not_empty (wait lk)))
			 (return (aref deque n)))))


		    (do0
		     "template <typename T, int MaxLen>"
		     (defclass FixedQueue "public std::queue<T>"
		       "// https://stackoverflow.com/questions/56334492/c-create-fixed-size-queue"
		       "public:"
		       (defun push (val)
			 (declare (type "const T&" val))
			 (when (== MaxLen (this->size))
			   (this->pop))
			 ("std::queue<T>::push" val))))
		    
		    (do0
		     "template <typename T, int MaxLen>"
		     
		     (defclass GuardedWaitingQueue ()
		       "// https://baptiste-wicht.com/posts/2012/04/c11-concurrency-tutorial-advanced-locking-and-condition-variables.html"
		       "private:"
		       (let ((mutex)
			     (not_empty)
			     (not_full)
			     (queue))
			 (declare (type "std::mutex" mutex)
				  (type "std::condition_variable" not_empty not_full)
					;(type "FixedQueue<T,MaxLen>" queue)
				  (type "std::queue<T>" queue)
				  ))
		       "public:"
		       (defun push (val)
			 (declare (type "const T&" val))
			 (progn
			   ,(lock 'mutex)
			   (while (== MaxLen (queue.size))
			     (dot not_full (wait lk)))
			   (queue.push val)
			   (lk.unlock))
			 (dot not_empty
			      (notify_one)))
		       (defun front_no_wait ()
			 (declare (values T))
			 ,(guard 'mutex)
			 (return (queue.front)))
		       (defun size_no_wait ()
			 (declare (values size_t))
			 ,(guard 'mutex)
			 (return (queue.size)))
		       (defun front_and_pop ()
			 (declare (values T))
			 "// fixme: should this return a reference?"
			 ,(lock 'mutex)
			 (while (== 0 (queue.size))
			   (dot not_empty (wait lk)))
			 (if (queue.size)
			     (do0
			      (let ((result (queue.front)))
				(queue.pop)
				(lk.unlock)
				(not_full.notify_one)
				(return result)))
			     (do0
			      (throw ("std::runtime_error" (string "can't pop empty"))))))))
		    
		    " "
		    ,(emit-globals)
		    " "
		    "#endif"
		    " "))))
