(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre"))

(in-package :cl-cpp-generator2)


(setf *features* (union *features* '(:safety
					;:nolog
					;:log-brc
				     ;:log-consume
				     )))
(setf *features* (set-difference *features* '(;:safety
					      :nolog
					      :log-brc
					      :log-consume
					      )))


(progn
  ;; make sure to run this code twice during the first time, so that
  ;; the functions are defined

  (defparameter *source-dir* #P"example/08_copernicus_radar/source_doppler/")

  (progn
    ;; collect code that will be emitted in utils.h
    (defparameter *utils-code* nil)
    (defun emit-utils (&key code)
      (push code *utils-code*)
      " "))
  (progn
    (defparameter *module-global-parameters* nil)
    (defparameter *module* nil)
    (defun cuprint (call &optional rest)
      `(progn (let ((r ,call))
		(<< "std::cout"
		  (- (dot ("std::chrono::high_resolution_clock::now")
			  (time_since_epoch)
			  (count))
		     ,(g `_start_time))
		  (string " ")
		  __FILE__
		  (string ":")
		  __LINE__
		  (string " ")
		  __func__
		  (string ,(format nil " ~a => " (emit-c :code call)))
		  r
		  (string " '")
		  (cudaGetErrorString r)
		  (string "' ")
		  ,@(loop for e in rest appending
			 `((string ,(format nil " ~a=" (emit-c :code e)))
			   ,e))
		  "std::endl"))
	      (assert (== cudaSuccess r))
	  )
      )
    (defun cufftprint (call &optional rest)
      `(progn (let ((r ,call))
		(<< "std::cout"
		  (- (dot ("std::chrono::high_resolution_clock::now")
			  (time_since_epoch)
			  (count))
		     ,(g `_start_time))
		  (string " ")
		  __FILE__
		  (string ":")
		  __LINE__
		  (string " ")
		  __func__
		  (string ,(format nil " ~a => " (emit-c :code call)))
		  r
		  
		  ,@(loop for e in rest appending
			 `((string ,(format nil " ~a=" (emit-c :code e)))
			   ,e))
		  "std::endl"))
	      (assert (== cudaSuccess r))
	  )
      )
    (defun vkprint (call &optional rest)
      `(do0 #-nolog
	    (do0
	     ,call
	     (<< "std::cout"
		 (- (dot ("std::chrono::high_resolution_clock::now")
			 (time_since_epoch)
			 (count))
		    ,(g `_start_time))
		 (string " ")
		 __FILE__
		 (string ":")
		 __LINE__
		 (string " ")
		 __func__
		 (string ,(format nil " ~a " (emit-c :code call)))
		 ,@(loop for e in rest appending
			`((string ,(format nil " ~a=" (emit-c :code e)))
			  ,e))
		 "std::endl"))))
    (defun logprint (msg &optional rest)
      `(do0
	#-nolog
	(do0
	 ("std::setprecision" 3)
	 (<< "std::cout"
	     ("std::setw" 10)
	     (- (dot ("std::chrono::high_resolution_clock::now")
		     (time_since_epoch)
		     (count))
		,(g `_start_time))
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
		      (string ,(format nil " ~a=" (emit-c :code e)))
		      ,e))
	     "std::endl"))))
    
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
  
  (let ((data-filename "/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar/source/o_range24890_echoes48141.cf"))
     (cl-ppcre:register-groups-bind (range-s echo-s) (".*range(\\d*).*echoes(\\d*)\\.cf" data-filename)
       (let ((range-value (parse-integer range-s))
	     (echo-value (parse-integer echo-s)))
	 (define-module
	      `(main ((_filename :direction 'out :type "char const *")
		     (_range :direction 'out :type int)
		     (_echo :direction 'out :type int)
		     )
		    (do0
		     (include <iostream>
			      <chrono>
			      <cstdio>
			      <cassert>
			      <unordered_map>
			      <string>
			      <fstream>)

		 
		     (let ((state ,(emit-globals :init t)))
		       (declare (type "State" state)))


		     (do0
		      (defun now ()
			(declare (values double))
			(let ((tp))
			  (declare (type "struct timespec" tp))
			  ;; https://stackoverflow.com/questions/6749621/how-to-create-a-high-resolution-timer-in-linux-to-measure-program-performance
			  (clock_gettime CLOCK_REALTIME &tp)
			  (return (+ (cast double tp.tv_sec)
				     (* 1d-9 tp.tv_nsec)))))
		      (defun mainLoop ()
			,(logprint "mainLoop" `())
		    
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
		      )
		 
		     (defun main ()
		       (declare (values int))
		       (setf ,(g `_start_time) (dot ("std::chrono::high_resolution_clock::now")
						    (time_since_epoch)
						    (count))
			     ,(g `_echo) ,echo-value
			     ,(g `_range) ,range-value)
					;(vkprint "main" )
		       (setf ,(g `_filename)
			     (string
			      ,data-filename)) 
		       (init_mmap ,(g `_filename))
		       (initProcessing)
		       (runProcessing 0)
		       (do0
			(run)
			(cleanupDraw)
			(cleanupWindow)
			(cleanupProcessing))
		       (return 0))))))))

  
  		 
		 
  
  (define-module
      `(mmap
	((_mmap_data :direction 'out :type void*)
	 (_mmap_filesize :direction 'out :type size_t))
	(do0
	 (include <sys/mman.h>
		  <sys/stat.h>
		  <sys/types.h>
		  <fcntl.h>
		  <cstdio>
		  <cassert>
		  <iostream>)
	 (defun get_filesize (filename)
	   (declare (type "const char*" filename)
		    (values size_t))
	   (let ((st))
	     (declare (type "struct stat" st))
	     (stat filename &st)
	     (return st.st_size)))
	 (defun destroy_mmap ()
	   (let ((rc (munmap ,(g `_mmap_data)
			     ,(g `_mmap_filesize))))  
	     (unless (== 0 rc)
	       ,(logprint "fail munmap" `(rc)))
	     (assert (== 0 rc))))
	 (defun init_mmap (filename)
	   (declare (type "const char*" filename))
	   (let ((filesize (get_filesize filename))
		 (fd (open filename O_RDONLY 0)))
	     ,(logprint "size" `(filesize filename))
	     (when (== -1 fd)
	       ,(logprint "fail open" `(fd filename)))
	     (assert (!= -1 fd))
	     (let ((data (mmap NULL filesize PROT_READ MAP_PRIVATE fd 0)))
	       (when (== MAP_FAILED data)
		 ,(logprint "fail mmap"`( data)))
	       (assert (!= MAP_FAILED data))
	       (setf ,(g `_mmap_filesize) filesize
		     ,(g `_mmap_data) data)))))))

  (define-module
      `(glfw_window
	((_window :direction 'out :type GLFWwindow* )
	 (_framebufferResized :direction 'out :type bool)
	 )
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
	      (setf ,(g `_window) (glfwCreateWindow 800 600
						    (string "doppler window")
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
      `(draw ((_fontTex :direction 'out :type GLuint))
	     (do0
	      (defun uploadTex (image w h)
		(declare (type "const void*" image)
			 (type int w h))
		(glGenTextures 1 (ref ,(g `_fontTex)))
		(glBindTexture GL_TEXTURE_2D ,(g `_fontTex))
		(glTexParameteri GL_TEXTURE_2D GL_TEXTURE_MIN_FILTER GL_LINEAR)
		(glTexParameteri GL_TEXTURE_2D GL_TEXTURE_MAG_FILTER GL_LINEAR)
		(glTexImage2D GL_TEXTURE_2D 0 GL_RGBA w h 0 GL_RGBA GL_UNSIGNED_BYTE image))
	      
	      (defun initDraw ()
		(glEnable GL_TEXTURE_2D)
		(glClearColor 0 0 0 1))
	      (defun cleanupDraw ()
		(glDeleteTextures 1 (ref ,(g `_fontTex))))
	      (defun drawFrame ()
		(glClear GL_COLOR_BUFFER_BIT)
		(do0
		 (glBegin GL_QUADS)
		 ,@(loop for (e f) in `((0 0)
					(0 1)
					(1 1)
					(1 0))
		      collect
			`(do0
			  (glVertex2f ,(* 1 (- e .5))
				      ,(* 1 (- f .5)))
			  (glTexCoord2f ,e ,f)))
		 (glEnd))
		(glfwSwapBuffers ,(g `_window))))))

  (define-module
      `(cuda_processing ()
		   (do0
		    "// https://github.com/NVIDIA/cuda-samples/blob/master/Samples/simpleCUFFT/simpleCUFFT.cu"
		    (include <cassert>)
		    " "
		    (include
		     "/opt/cuda/targets/x86_64-linux/include/cufftw.h"
		     "/opt/cuda/targets/x86_64-linux/include/cufft.h"
		     "/opt/cuda/targets/x86_64-linux/include/cuda_runtime.h"
		     )
		    " "
		    "typedef float2 Complex;"
		    " "
		    (defun initProcessing ()
		      (do0
		       (let ((n_cuda 0))
			 ,(cuprint `(cudaGetDeviceCount &n_cuda) `(n_cuda)))
		       ,(cuprint `(cudaSetDevice 0)))
		      )
		    (defun runProcessing (index)
		      (declare (type int index))
		      
		      ;; ,(g `_echo)
		      (let ((p (reinterpret_cast<Complex*> ,(g `_mmap_data)))
			    (range ,(g `_range))
			    (h_signal (ref (aref p (* range index))))
			    (d_signal)
			    (d_kernel)
			    (memsize (* (sizeof Complex) range)))
			(declare (type Complex* d_signal d_kernel))
			(do0
			 ,(cuprint `(cudaMalloc (reinterpret_cast<void**> &d_signal)
						memsize))
			 ,(cuprint `(cudaMemcpy d_signal h_signal memsize cudaMemcpyHostToDevice)))
			(let ((plan ))
			  (declare (type cufftHandle plan))
			  ,(cufftprint `(cufftPlan1d &plan range CUFFT_C2C 1))
			  ,(cufftprint `(cufftExecC2C plan d_signal d_signal CUFFT_FORWARD)))
			
			(do0
			 
			 ,(cuprint `(cudaMalloc (reinterpret_cast<void**> &d_kernel)
						(* (sizeof Complex)
						   range)))))
		      )
		    (defun cleanupProcessing ()))))
  
  
  (progn
    (with-open-file (s (asdf:system-relative-pathname 'cl-cpp-generator2
						 (merge-pathnames #P"proto2.h"
								  *source-dir*))
		       :direction :output
		       :if-exists :supersede
		       :if-does-not-exist :create)
      (loop for e in (reverse *module*) and i from 0 do
	   (destructuring-bind (&key name code) e  
	     (emit-c :code code :hook-defun 
		     #'(lambda (str)
			 (format s "~a~%" str)))
	     
	     (write-source (asdf:system-relative-pathname
			    'cl-cpp-generator2
			    (let ((cuda (cl-ppcre:scan "cuda" (string-downcase (format nil "~a" name)))))
			      (format nil
				     "~a/doppler_~2,'0d_~a.~a"
				     *source-dir* i name
				     (if cuda
					 "cu"
					 "cpp"))))
			   code))))
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
		    (include <GLFW/glfw3.h>)
		    " "
		    ,(emit-globals)
		    " "
		    "#endif"
		    " "))
    
    
    
    ;; we need to force clang-format to always have the return type in the same line as the function: PenaltyReturnTypeOnItsOwnLine
					;(sb-ext:run-program "/bin/sh" `("gen_proto.sh"))
    #+nil (sb-ext:run-program "/usr/bin/make" `("-C" "source" "-j12" "proto2.h"))))

