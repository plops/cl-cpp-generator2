(declaim (optimize 
	  (safety 3)
	  (speed 0)
	  (debug 3)))

(eval-when (:compile-toplevel :execute :load-toplevel)
     (ql:quickload "cl-cpp-generator2")
     (ql:quickload "cl-ppcre"))

(in-package :cl-cpp-generator2)



(setf *features* (union *features* `()))

(setf *features* (set-difference *features*
				 '()))

(progn
  (defparameter *source-dir* #P"example/14_skia/source/")
  
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
		      (string "::")
		      (dot (typeid ,e)
			   (name))
		      (string "'")))
	     "std::endl"
	     "std::flush"))))
    (defun emit-globals (&key init)
      (let ((l `((_start_time ,(emit-c :code `(typeof (dot ("std::chrono::high_resolution_clock::now")
							   (time_since_epoch)
							   (count)))))
		 ,@(loop for e in *module-global-parameters* collect
			(destructuring-bind (&key name type default)
			    e
			  (declare (ignorable default))
			  `(,name ,type))))))
	(if init
	    `(curly
	      ,@(remove-if
		 #'null
		 (loop for e in l collect
		      (destructuring-bind (name type &optional value) e
			(declare (ignorable type))
			(when value
			  `(= ,(format nil ".~a" (elt (cl-ppcre:split "\\[" (format nil "~a" name)) 0)) ,value))))))
	    `(do0
	      (include <chrono>)
	      (defstruct0 State
		  ,@(loop for e in l collect
 			 (destructuring-bind (name type &optional value) e
			   (declare (ignorable value))
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
		  ;(include "proto2.h")
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
	       (declare (ignorable direction))
	       (push `(:name ,parameter-name :type ,type :default ,default)
		     *module-global-parameters*))))))
  (defun g (arg)
    `(dot state ,arg))

  (let*  ()
    (define-module
       `(base ((_main_version :type "std::string")
		    (_code_repository :type "std::string")
		    (_code_generation_time :type "std::string")
		 )
	      (do0
	    
	    
		    (include <iostream>
			     <chrono>
			     <thread>
			     <typeinfo>
			     )
		    "#define SK_GL"
		    "#define GR_GL_LOG_CALLS 0"
		    "#define GR_GL_CHECK_ERROR 0"
		    
		    (include <include/gpu/GrBackendSurface.h>
			     <include/gpu/GrDirectContext.h>
			     ;<include/gpu/GrContext.h>
			     <SDL2/SDL.h>
			     <include/core/SkCanvas.h>
			     <include/core/SkGraphics.h>
			     <include/core/SkSurface.h>
					;<include/core/SkString.h>
					;<include/core/SkImageEncoder.h>
			     <include/gpu/gl/GrGLInterface.h>
			     <src/gpu/gl/GrGLUtil.h>
			     <GL/gl.h>
			     )
		    " "

		    (split-header-and-code
		     (do0
		      "// header"
		      )
		     (do0
		      "// implementation"
		      ))
		    ;; https://skia.googlesource.com/skia/+/master/example/SkiaSDLExample.cpp
		    ;; https://github.com/QtSkia/QtSkia/blob/master/QtSkia/QtSkiaGui/QSkiaOpenGLWindow.cpp
		    (defclass* SkiaGLPrivate ()
		      "public:"
		      "sk_sp<GrContext> context=nullptr;"
		      "sk_sp<SkSurface> gpu_surface=nullptr;"
		      "SkImageInfo info;"
		      "int old_w;"
		      "int old_h;"
		      )
		    (defun skia_init (s w h)
		      (declare (type int w h)
			       (type SkiaGLPrivate& s))
		      
		      (let (;(interface (GrGLMakeNativeInterface))
			    
			    )
			(setf s->context (GrDirectContext--MakeGL))
			
			(SkASSERT s->context)
			)
		      (setf s->info (SkImageInfo--MakeN32Premul w h))
		      (setf s->gpu_surface (SkSurface--MakeRenderTarget (dot s->context
									     (get))
									SkBudgeted--kNo
									s->info))
		      (unless s->gpu_surface
			,(logprint "surface failed"))
		      (glViewport 0 0 w h)
		      (setf s->old_w w
			    s->old_h h))

		    
		    (defun main (argc argv
				 )
		      (declare (type int argc)
			       (type char** argv)
			       (values int))
		      ,(logprint "start" `(argc (aref argv 0)))
		      (do0
		       (SDL_GL_SetAttribute SDL_GL_CONTEXT_MAJOR_VERSION 3)
		       (SDL_GL_SetAttribute SDL_GL_CONTEXT_MINOR_VERSION 0)
		       (do0 
			 (SDL_GL_SetAttribute SDL_GL_CONTEXT_PROFILE_MASK
					      SDL_GL_CONTEXT_PROFILE_CORE)
			 (let ((windowFlags (logior SDL_WINDOW_OPENGL
						    SDL_WINDOW_RESIZABLE))
			       )
			   ,@(loop for (e f) in `((red_size 8)
						  (green_size 8)
						  (blue_size 8)
						  (doublebuffer 1)
						  (depth_size 0)
						  (stencil_size 8)
						  (accelerated_visual 1)
						  (multisamplebuffers 1)
						  (multisamplesamples 0))
				   collect
				   `(SDL_GL_SetAttribute ,(string-upcase (format nil "SDL_GL_~a" e))
							 ,f)
				   ))
			 (unless (== 0 (SDL_Init (logior SDL_INIT_VIDEO SDL_INIT_EVENTS)))
			   ,(logprint "init error"))
			 (let ((window (SDL_CreateWindow (string "sdl window")
							 SDL_WINDOWPOS_CENTERED
							 SDL_WINDOWPOS_CENTERED
							 512 200 windowFlags)))
			   (unless window
			     ,(logprint "window error"))
			   (let ((ctx (SDL_GL_CreateContext window)))
			     (unless ctx
			       ,(logprint "ctx error"))
			     (let ((success (SDL_GL_MakeCurrent window ctx)))
			      (unless (== 0 success )
				,(logprint "makecurrent error"))
			       (let ((windowFormat (SDL_GetWindowPixelFormat window))
				     (dw (int 0))
				     (dh dw)
				     (contextType dw))
				 (SDL_GL_GetAttribute SDL_GL_CONTEXT_PROFILE_MASK &contextType)
				 (SDL_GL_GetDrawableSize window &dw &dh)
				 ,(logprint "" `(windowFormat contextType dw dh))
				 ,(logprint "" `((SDL_GetPixelFormatName windowFormat)))
				 (glViewport 0 0 dw dh)
				 (glClearColor 1 0 1 1)
				 (glClearStencil 0)
				 (glClear (logior GL_COLOR_BUFFER_BIT
						  GL_STENCIL_BUFFER_BIT))
				 (SDL_GL_SetSwapInterval 1)
				 #+nil (let ((options (GrContextOptions))
				      (interface  (GrGLMakeNativeInterface))
				      (sContext (GrDirectContext--MakeGL interface
									 ;nullptr
						      options
						 )
					;  (release)
						
						)
				       (image_info (SkImageInfo--MakeN32Premul dw dh))
				       )
				   ;(declare (type "sk_sp<GrContext>" sContext))
				  #+nil (setf image_info.fFBOID 0 ;; default framebuffer
					   image_info.fFormat GL_RGBA8
					   )
				   #+nil (let ((gpu_surface (SkSurface--MakeRenderTarget sContext
										   SkBudgeted--kNo
										   image_info
										   )))
				     (unless gpu_surface
					  ,(logprint "sksurface error"))
				     (let ((canvas (gpu_surface->getCanvas)))
				       (dotimes (i (* 60 3)) ; while true
					;(glClear GL_COLOR_BUFFER_BIT)
					 (let ((paint (SkPaint)))
					   (paint.setColor SK_ColorWHITE)
					   (canvas->drawPaint paint)
					   (paint.setColor SK_ColorBLUE)
					   (canvas->drawRect (curly 10 20 30 50)
							     paint)
					   (sContext->flush))
					 (SDL_GL_SwapWindow window))))
				   #+nil(let ((fb_info (GrGLFramebufferInfo))
					 (colorType kRGBA_8888_SkColorType))
				     (setf fb_info.fFBOID 0 ;; default framebuffer
					   fb_info.fFormat GL_RGBA8
					   )
				     (let ((render_target (GrBackendRenderTarget dw dh
										 0 ;; msaa
										 8 ;; stencil
										 fb_info
										 ))
					   (props (SkSurfaceProps))
					   (sSurface (dot (SkSurface--MakeFromBackendRenderTarget
							   sContext render_target
							   kBottomLeft_GrSurfaceOrigin
							   colorType
							   nullptr
							   &props)
							  (release)))
					   (canvas (sSurface->getCanvas)))
				       (dotimes (i (* 60 3)) ; while true
					 ;(glClear GL_COLOR_BUFFER_BIT)
					 (let ((paint (SkPaint)))
					   (paint.setColor SK_ColorWHITE)
					   (canvas->drawPaint paint)
					   (paint.setColor SK_ColorBLUE)
					   (canvas->drawRect (curly 10 20 30 50)
							     paint)
					   (sContext->flush))
					 (SDL_GL_SwapWindow window))
				       )))
				 (let ((state (SkiaGLPrivate)))
				   (skia_init &state dw dh))
				 
				 #+nil (let ((interface (GrGLMakeNativeInterface))
				       (grContext (GrDirectContext--MakeGL interface))
				       )
				   
				   
				   (SkASSERT grContext)
				   ,(logprint "" `(grContext))
				    (let (#-nil (buffer (GrGLint 0)))
				     #-nil (do0 (GR_GL_GetIntegerv (interface.get)
							       GR_GL_FRAMEBUFFER_BINDING
							       &buffer)
						,(logprint "" `(buffer)))
				      
				     (let ((info (GrGLFramebufferInfo)))
				       (setf info.fFBOID (static_cast<GrGLuint> buffer)
					     )
				       
				       )
				     (let ((target (GrBackendRenderTarget dw dh
									  0 ;; msaa
									  8 ;; stencil
									  info
									  ))))
				      ,(logprint "" `((target.width)))
				      (let ((props (SkSurfaceProps))
					    (surface (SkSurface--MakeFromBackendRenderTarget
						      (grContext.get)
						      target
						      kBottomLeft_GrSurfaceOrigin
						      kRGBA_8888_SkColorType ;;colorType
						      nullptr
						      &props))
					    (canvas (surface->getCanvas))
					    )
					,(logprint "" `(surface))
				      (dotimes (i (* 60 3)) ; while true
					(glClear GL_COLOR_BUFFER_BIT)
					#+nil (let ((paint (SkPaint)))
					  (paint.setColor SK_ColorWHITE)
					  (canvas->drawPaint paint)
					  (paint.setColor SK_ColorBLUE)
					  (canvas->drawRect (curly 10 20 30 50)
							    paint)
					  (grContext->flush))
					(SDL_GL_SwapWindow window))))
				   ,(logprint "shutdown")
				       )))
			     #+nil (do0
					;"delete sSurface;"
			      ;"delete gpu_surface;"
			      "delete sContext;"
			      )
			     (do0
			      
			      ,(logprint "destroy gl ctx")
			      (when ctx
				    (SDL_GL_DeleteContext ctx)))
			     (do0 ,(logprint "destroy window")
				  (SDL_DestroyWindow window))
			     (do0 ,(logprint "quit")
				  (SDL_Quit)))
			   )))
		      #+nil
		      (let (;(ag (SkAutoGraphics))
			    (path (SkString (string "skhello.png")))
			    (paint (SkPaint))
			    )
			;; https://gist.github.com/zester/1177738/16f3e6a5fe20086f8db3359173d613bd5d154901
			(paint.setARGB 255 255 255 255)
			(paint.setAntiAlias true)
			;(paint.setTextSize (SkIntToScalar 30))
			(let ((width (SkScalar 800))
			      (height (SkScalar 600))
			      (bitmap (SkBitmap)))
			  (bitmap.allocPixels (SkImageInfo--MakeN32Premul width height))
			  
			  (let ((canvas (SkCanvas bitmap)))
			    (canvas.drawColor SK_ColorWHITE)
			    )
			  (let ((src (SkEncodeBitmap
					 bitmap
					 SkEncodedImageFormat--kPNG 100))
				(img (src.get))))))
		      (return 0)))))
    
    
  )
  
  (progn
    (progn ;with-open-file
      #+nil (s (asdf:system-relative-pathname 'cl-cpp-generator2
					(merge-pathnames #P"proto2.h"
							 *source-dir*))..
	 :direction :output
	 :if-exists :supersede
	 :if-does-not-exist :create)
      #+nil (format s "#ifndef PROTO2_H~%#define PROTO2_H~%~a~%"
		    (emit-c :code `(include <cuda_runtime.h>
					    <cuda.h>
					    <nvrtc.h>)))

      ;; include file
      ;; http://www.cplusplus.com/forum/articles/10627/
      
      (loop for e in (reverse *module*) and i from 0 do
	   (destructuring-bind (&key name code) e
	     
	     (let ((cuda (cl-ppcre:scan "cuda" (string-downcase (format nil "~a" name)))))
	       
	       (unless cuda
		 #+nil (progn (format t "emit function declarations for ~a~%" name)
			      (emit-c :code code
				      :hook-defun #'(lambda (str)
						      (format t "~a~%" str))
				      :header-only t))
		 #+nil (emit-c :code code
			 :hook-defun #'(lambda (str)
					 (format s "~a~%" str)
					 )
			 :hook-defclass #'(lambda (str)
					    (format s "~a;~%" str)
					    )
			 :header-only t
			 )
		 (let* ((file (format nil
				      "vis_~2,'0d_~a"
				      i name
				      ))
			(file-h (string-upcase (format nil "~a_H" file))))
		   (with-open-file (sh (asdf:system-relative-pathname 'cl-cpp-generator2
								      (format nil "~a/~a.hpp"
									      *source-dir* file))
				       :direction :output
				       :if-exists :supersede
				       :if-does-not-exist :create)
		     (format sh "#ifndef ~a~%" file-h)
		     (format sh "#define ~a~%" file-h)
		     
		     (emit-c :code code
			     :hook-defun #'(lambda (str)
					     (format sh "~a~%" str)
					     )
			     :hook-defclass #'(lambda (str)
						(format sh "~a;~%" str)
						)
			     :header-only t
			     )
		     (format sh "#endif")
		     ))

		 )

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
      #+nil (format s "#endif"))
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
			  e))
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

		    #+nil (include <complex>)
		    #+nil (include <deque>
			     <map>
			     <string>)
		    #+nil (include <thread>
			     <mutex>
			     <queue>
			     <condition_variable>
			     )
		    " "

		    " "
		    ;(include "proto2.h")
		    " "
		    ,@(loop for e in (reverse *global-code*) collect
			 e)

		    " "
		    ,(emit-globals)
		    " "
		    "#endif"
		    " "))))



