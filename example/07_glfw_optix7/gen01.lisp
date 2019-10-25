;; current state: i'm trying to get nuklear gui running, no optix yet

(setf *features* (union *features* '(:generic-c)))

(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre"))

(in-package :cl-cpp-generator2)

;; if nolog is off, then validation layers will be used to check for mistakes
;; if surface is on, then a window surface is created; otherwise only off-screen render
;; if nolog-frame is off then draw frame prints lots of stuff
;;(setf *features* (union *features* '()))
;;(setf *features* (set-difference *features* '()))

;; to find cglm
;; export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib64/pkgconfig


;; https://github.com/nvpro-samples/optix_advanced_samples/blob/master/src/optixIntroduction/optixIntro_01/src/main.cpp
;; https://cdn.statically.io/gh/vurtun/nuklear/master/doc/nuklear.html#nuklear/example
;; https://github.com/vurtun/nuklear/issues/226
;; https://github.com/vurtun/nuklear/blob/master/example/file_browser.c
(progn
  (progn
    (defun vkprint (msg
		    &optional rest)
      ;;"{sec}.{nsec} {__FILE__}:{__LINE__} {__func__}"
      (let* ((m `(string ,(format nil " ~a: " msg)))
	     (l `(((string "%6.6f") (- current_time ,(g `_start_time)))
					;((printf_dec_format tp.tv_sec) tp.tv_sec)
					;((string "."))
					;((printf_dec_format tp.tv_nsec) tp.tv_nsec)
		  ((string " "))
		  ((printf_dec_format __FILE__) __FILE__)
		  ((string ":"))
		  ((printf_dec_format __LINE__) __LINE__)
		  ((string " "))
		  ((printf_dec_format __func__) __func__)
		  (,m)
		  ,@(loop for e in rest appending
			 `(((string ,(format nil " ~a=" (emit-c :code e))))
			   ((printf_dec_format ,e) ,e)
			   ((string " (%s)") (type_string ,e))
			   ))
		  ((string "\\n")))))
	`(progn
	   (let (;(tp)
		 (current_time (now)))
	     ;(declare (type "struct timespec" tp))
	     ;; https://stackoverflow.com/questions/6749621/how-to-create-a-high-resolution-timer-in-linux-to-measure-program-performance
	     ;(clock_gettime CLOCK_REALTIME &tp)
	     ,@(loop for e in l collect
		    (destructuring-bind (fmt &optional value) e
		      (if value
			  `(printf ,fmt ,value)
			  `(printf ,fmt))))))))
    
    (progn
      (defun set-members (params)
	"setf on multiple member variables of an instance"
	(destructuring-bind (instance &rest args) params
	  `(setf ,@(loop for i from 0 below (length args) by 2 appending
			(let ((keyword (elt args i))
			      (value (elt args (+ i 1))))
			  `((dot ,instance ,keyword) ,value))))))
      ))
  (progn
    ;; collect code that will be emitted in utils.h
    (defparameter *utils-code* nil)
    (defun emit-utils (&key code)
      (push code *utils-code*)
      " "))
  (progn
    (defparameter *module-global-parameters* nil)
    (defparameter *module* nil)

    (defun emit-globals (&key init)
      (let ((l `(
		 (_start_time double)
		 ;; 
		 (_window GLFWwindow* NULL)
		 (_framebufferResized _Bool)
		 (_fontTex GLuint)
		 )))
	(if init
	    `(curly
	      ,@(remove-if
		 #'null
		 (loop for e in l collect
		      (destructuring-bind (name type &optional value) e
			(when value
			  `(= ,(format nil ".~a" (elt (cl-ppcre:split "\\[" (format nil "~a" name)) 0)) ,value))))))
	    `(do0
	      "enum {_N_IMAGES=4,_MAX_FRAMES_IN_FLIGHT=2};"
	      (defstruct0 State
		  ,@(loop for e in l collect
			 (destructuring-bind (name type &optional value) e
			   `(,name ,type))))))))
  
    (defun define-module (args)
      
      (destructuring-bind (module-name global-parameters module-code) args
	(let ((header ()))
	
	  (push `(do0
		
		  " "
		  (include <GLFW/glfw3.h>)
		  " "
		  (include "utils.h")
		  " "
		  (include "globals.h")
		  " "
		  (include "proto2.h")
			
		  " "
		  )
		header)
	  (unless (cl-ppcre:scan "main" (string-downcase (format nil "~a" module-name)))
	    (push `(do0 "extern State state;")
		  header)
	    )
	
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
      `(main ()
	     (do0
	      (let ((state ,(emit-globals :init t)))
		(declare (type State state)))
	      (defun now ()
		 (declare (values double))
		 (let ((tp))
		  (declare (type "struct timespec" tp))
		  ;; https://stackoverflow.com/questions/6749621/how-to-create-a-high-resolution-timer-in-linux-to-measure-program-performance
		  (clock_gettime CLOCK_REALTIME &tp)
		  (return (+ (cast double tp.tv_sec)
			     (* 1d-9 tp.tv_nsec)))))
	      (defun mainLoop ()
		,(vkprint "mainLoop")
		
		(while (not (glfwWindowShouldClose ,(g `_window)))
		  (glfwPollEvents)
		  (drawFrame)
		  )
		
		)
	      (defun run ()
		(initWindow)
		(initDraw)
		(mainLoop)
		;(cleanup)
		)
	      
	      (defun main ()
		(declare (values int))
		(setf ,(g `_start_time) (now))
		(run)
		(cleanupDraw)
		(cleanupWindow)))))
    (define-module
      `(glfw_window
	((_window :direction 'out :type GLFWwindow* ) )
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
	   ,(vkprint "error" `(err description)))
	 (defun framebufferResizeCallback (window width height)
	   (declare (values "static void")
		    ;; static because glfw doesnt know how to call a member function with a this pointer
		    (type GLFWwindow* window)
		    (type int width height))
	   ,(vkprint "resize" `(width height))
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
	      (setf ,(g `_window) (glfwCreateWindow 800 600
						    (string "optix window")
						    NULL
						    NULL))
	      ;; store this pointer to the instance for use in the callback
	      (glfwSetKeyCallback ,(g `_window) keyCallback)
	      (glfwSetWindowUserPointer ,(g `_window) (ref state))
	      (glfwSetFramebufferSizeCallback ,(g `_window)
					      framebufferResizeCallback)
	      (glfwMakeContextCurrent ,(g `_window))
	      )))
	 (defun cleanupWindow ()
	   (declare (values void))
	   (glfwDestroyWindow ,(g `_window))
	   (glfwTerminate)
	   ))))
    (define-module
      `(draw ()
	     (do0
	      "#define NK_IMPLEMENTATION"
	      "#define NK_INCLUDE_DEFAULT_ALLOCATOR"
	      "#define NK_INCLUDE_FONT_BAKING"
	      "#define NK_INCLUDE_DEFAULT_FONT"
	      (include "nuklear.h")
	      "enum {EASY, HARD};"
	      (let ((ctx)
		    (atlas)
		    (font)
		    (null)
		    (op EASY))
		(declare (type "struct nk_context" ctx)
			 (type "struct nk_font_atlas" atlas)
			 (type "struct nk_font*" font)
			 (type "struct nk_draw_null_texture" null)
			 (type int op)))
	      (defun uploadAtlas (image w h)
		(declare (type "const void*" image)
			 (type int w h))
		(glGenTextures 1 (ref ,(g `_fontTex)))
		(glBindTexture GL_TEXTURE_2D ,(g `_fontTex))
		(glTexParameteri GL_TEXTURE_2D GL_TEXTURE_MIN_FILTER GL_LINEAR)
		(glTexParameteri GL_TEXTURE_2D GL_TEXTURE_MAG_FILTER GL_LINEAR)
		(glTexImage2D GL_TEXTURE_2D 0 GL_RGBA w h 0 GL_RGBA GL_UNSIGNED_BYTE image))
	      (defun initDraw ()
		(do0
		 (nk_font_atlas_init_default &atlas)
		 (nk_font_atlas_begin &atlas)
		 (nk_font_atlas_add_default &atlas 13s0 NULL)
		 (let ((w)
		       (h)
		       (image)
		       )
		   (declare (type int h w)
			    (type "const void*" image)
			    )
		   (setf image (nk_font_atlas_bake &atlas &w &h NK_FONT_ATLAS_RGBA32))
		   (uploadAtlas image w h)
		   (nk_font_atlas_end &atlas (nk_handle_id ,(g `_fontTex))  &null)))
		(nk_init_default &ctx &font->handle)
		(glEnable GL_TEXTURE_2D)
		
		(glClearColor 1 0 0 1)
		)
	      (defun cleanupDraw ()
		(nk_font_atlas_clear &atlas)
		(nk_free &ctx)
		(glDeleteTextures 1 (ref ,(g `_fontTex))))
	      (defun drawFrame ()
		(do0
		 "// poll event is called before"
		 ,@(loop for e in `((DEL DELETE)
				    (ENTER)
				    (TAB)
				    (BACKSPACE)
				    (LEFT)
				    (RIGHT)
				    (UP)
				    (DOWN)) collect
			(destructuring-bind (key &optional (glfwkey key)) e
			  `(nk_input_key &ctx ,(string-upcase (format nil "NK_KEY_~a" key))
					 (== GLFW_PRESS
					     (glfwGetKey ,(g `_window)
						      ,(string-upcase (format nil "GLFW_KEY_~a" glfwkey))))
					 )))
		 (let ((x)
		       (y))
		   (declare (type double x y))
		   (glfwGetCursorPos ,(g `_window) &x &y)
		   (nk_input_motion &ctx (cast int x)
				    (cast int y))
		   ,@(loop for key in `(
				      LEFT
				      RIGHT
				      MIDDLE) collect
			`(nk_input_button &ctx ,(string-upcase (format nil "NK_BUTTON_~a" key))
					  (cast int x)
					  (cast int y)
					  (== GLFW_PRESS
					      (glfwGetMouseButton ,(g `_window)
								  ,(string-upcase (format nil "GLFW_MOUSE_BUTTON_~a" key))))
					  ))
		   )
		 (nk_input_end &ctx))
		(glClear GL_COLOR_BUFFER_BIT)
		#+nil
		(do0 (when (nk_begin &ctx (string "show")
				 (nk_rect 50 50 220 220)
				 (logior NK_WINDOW_BORDER
					 NK_WINDOW_MOVABLE
					 NK_WINDOW_CLOSABLE))
		   (nk_layout_row_static &ctx 30 80 1)
		   (when (nk_button_label &ctx (string "button")))
		   (nk_layout_row_dynamic &ctx 30 2)
		   (when (nk_option_label &ctx (string "easy") (== op EASY))
		     (setf op EASY))
		   (when (nk_option_label &ctx (string "hard") (== op HARD))
		     (setf op HARD)))
		     (nk_end &ctx))
		(glfwSwapBuffers ,(g `_window))
					;(nk_clear &ctx)
		(nk_input_begin &ctx)
		))))

    
    ;; we need an empty proto2.h. it has to be written before all c files so that make proto will work
    (write-source (asdf:system-relative-pathname 'cl-cpp-generator2 "example/07_glfw_optix7/source/proto2.h")
		  `(do0)  (user-homedir-pathname) t)

    (loop for e in (reverse *module*) and i from 0 do
	 (destructuring-bind (&key name code) e
	   (write-source (asdf:system-relative-pathname
			  'cl-cpp-generator2
			  (format nil
				  "example/07_glfw_optix7/source/optix_~2,'0d_~a.c"
				  i name))
			 code)))
    (write-source (asdf:system-relative-pathname
		   'cl-cpp-generator2
		   "example/07_glfw_optix7/source/utils.h"
		   )

		  `(do0
		    "#ifndef UTILS_H"
		    " "
		    "#define UTILS_H"
		    " "
		    
		    (do0
		     (include <stdio.h>)
		     " "
		     (include <stdbool.h>)
		     ;;"#define _POSIX_C_SOURCE 199309L"
		     " "
		     ;;(include <unistd.h>)
		     (include <time.h>)

		     " "
		     ;(include <cglm/cglm.h>)
		     " "
		     ,@(loop for e in (reverse *utils-code*) collect
			  e)
		     "#define length(a) (sizeof((a))/sizeof(*(a)))"
					;"#define max(a,b)  ({ __typeof__ (a) _a = (a);  __typeof__ (b) _b = (b);  _a > _b ? _a : _b; })"
					;"#define min(a,b)  ({ __typeof__ (a) _a = (a);  __typeof__ (b) _b = (b);  _a < _b ? _a : _b; })"
		     "#define max(a,b) ({ __auto_type _a = (a);  __auto_type _b = (b); _a > _b ? _a : _b; })"
		     "#define min(a,b) ({ __auto_type _a = (a);  __auto_type _b = (b); _a < _b ? _a : _b; })"
		     "#define printf_dec_format(x) _Generic((x), default: \"%p\", char: \"%c\", signed char: \"%hhd\", unsigned char: \"%hhu\", signed short: \"%hd\", unsigned short: \"%hu\", signed int: \"%d\", unsigned int: \"%u\", long int: \"%ld\", unsigned long int: \"%lu\", long long int: \"%lld\", float: \"%f\", double: \"%f\", long double: \"%Lf\", char*: \"%s\", const char*: \"%s\", unsigned long long int: \"%llu\",void*: \"%p\",bool:\"%d\")"
		     ,(format nil "#define type_string(x) _Generic((x), ~{~a: \"~a\"~^,~})"
			      (loop for e in `(default
						  
						  ,@(loop for h in
							 `(bool
							   ,@(loop for f in `(char short int "long int" "long long int") appending
								  `(,f ,(format nil "unsigned ~a" f)))
							   float double "long double"
							   "char*"
							   "void*"
							   )
						       appending
							 `(,h ,(format nil "const ~a" h)))
						  
						  )
				 appending
				   `(,e ,e)))
		     

		     
		     
		     " "
		     
		     )
		    " "
		    "#endif"
		    " ")
		  )
    (write-source (asdf:system-relative-pathname 'cl-cpp-generator2 "example/07_glfw_optix7/source/globals.h")
		  `(do0
		    "#ifndef GLOBALS_H"
		    " "
		    "#define GLOBALS_H"
		    " "
		    
		    
		    ,(emit-globals)
		    " "
		    "#endif"
		    " "))
    
    
    ;; we need to force clang-format to always have the return type in the same line as the function: PenaltyReturnTypeOnItsOwnLine
					;(sb-ext:run-program "/bin/sh" `("gen_proto.sh"))
    (sb-ext:run-program "/usr/bin/make" `("-C" "source" "-j12" "proto2.h")))
 

