(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "read-csv"))

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

(with-open-file (s (first (directory
                           "/home/martin/stage/cl-cpp-generator2/example/08_copernicus_radar/source/df.csv")))
  (defparameter *dfa-csv* (read-csv:parse-csv s)))





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
		     (_range_line :direction 'out :type "std::complex<float>*"))
		    (do0
		     (include <iostream>
			      <chrono>
			      <cstdio>
			      <cassert>
			      <unordered_map>
			      <string>
			      <fstream>
			      )
		     " "
		     (include <complex>)
		     " "
		     (include <cmath>)
		     " "
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
			  (drawGui)
			  (glfwSwapBuffers ,(g `_window))
			  )
		    
			)
		      (defun run ()
			(initWindow)
			(initGui)
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
			     ,(g `_range) ,range-value
			     ,(g `_range_line) nullptr)
					;(vkprint "main" )
		       (setf ,(g `_filename)
			     (string
			      ,data-filename)) 
		       (init_mmap ,(g `_filename))
		       (initProcessing)

		       (setf ,(g `_range_line)
			(runProcessing 0))
		       
		       (do0
			(run)
			(cleanupDraw)
			(cleanupGui)
			(cleanupWindow)
			(cleanupProcessing))
		       (return 0))))))

       
       
       
       
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
			  ,(g `_mmap_data) data)))))))))

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
		))))

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
		    (include "data.h")
		    " "
		    (include <complex>)
		    " " 
		    (include <cmath>)
		    " "
		    "// fixme: tx configuration for each pulse is currently always the same. for iw datasets i have to figure out how to get the tx configuration rank packets in the past."
		    
		    " "
		    ,(emit-utils :code
				 `(do0
				   "void initProcessing();"
				   "std::complex<float>* runProcessing(int);"
				   "void cleanupProcessing();"
				   ))
		    "typedef float2 Complex;"
		    " "
		    (defun ComplexMul (a b)
		      (declare (type Complex a b)
			       (values "static __device__ __host__ inline Complex"))
		      (let ((c))
			(declare (type Complex c))
			(setf c.x (- (* a.x b.x)
				     (* a.y b.y))
			      c.y (+ (* a.x b.y)
				     (* a.y b.x)))
			(return c)))
		    (defun ComplexPointwiseMul (a b size)
		      (declare (type Complex* a b)
			       (type int size)
			       (values "static __global__ void"))
		      (let ((numThreads (* blockDim.x
					   gridDim.x))
			    (threadID (+ (* blockIdx.x
					    blockDim.x)
					 threadIdx.x)))
			(for (("int i" threadID)
			      (< i size)
			      (incf i numThreads))
			     (setf (aref a i)
				   (ComplexMul (aref a i)
					       (aref b i))))))
		    
		    (defun initProcessing ()
		      (do0
		       (let ((n_cuda 0))
			 ,(cuprint `(cudaGetDeviceCount &n_cuda) `(n_cuda)))
		       ,(cuprint `(cudaSetDevice 0)))
		      )
		    (defun runProcessing (index)
		      (declare (type int index)
			       (values "std::complex<float>*"))
		      
		      ;; ,(g `_echo)
		      
		      (let ((p (reinterpret_cast<Complex*> ,(g `_mmap_data)))
			    (range ,(g `_range))
			    (h_signal (ref (aref p (* range index))))
			    
			    (d_signal)
			    (d_signal_out)
			    (d_kernel)
			    (d_kernel_out)
			    (memsize (* (sizeof Complex) range))
			    (h_signal2 (static_cast<Complex*> (malloc memsize))))
			(declare (type Complex* d_signal d_kernel d_signal_out p h_signal d_kernel_out)
				 (type "static Complex*" h_signal2))
			#+nil (do0
			 ,(logprint "runProcessing"
				    `(,@(loop for i below 5 collect
					     `(aref ("reinterpret_cast<std::complex<float>*>" h_signal)
						    ,i)))))
			(do0
			 ,(cuprint `(cudaMalloc (reinterpret_cast<void**> &d_signal)
						memsize)
				   `(memsize))
			 ,(cuprint `(cudaMalloc (reinterpret_cast<void**> &d_signal_out)
						memsize)
				   `(memsize))
			 ,(cuprint `(cudaMemcpy d_signal ;; dst
						h_signal ;; src
						memsize cudaMemcpyHostToDevice)
				   `(memsize)))


			#+nil
			(do0
			 "// copy data back"
			 (progn
			  (let ((h_signal3 (static_cast<Complex*> (malloc memsize)))
				(v ("reinterpret_cast<std::complex<float>*>" h_signal3)))
			    ,(cuprint `(cudaMemcpy h_signal3 ;; dst
						   d_signal ;; src
						   memsize cudaMemcpyDeviceToHost)
				      `(memsize))
			    ,(logprint "runProcessing"
				       `(,@(loop for i below 5 collect
						`(aref v
						       ,i))))
			    (free h_signal3))))
			
			(let ((plan ))
			  (declare (type cufftHandle plan))
			  ,(cufftprint `(cufftPlan1d &plan range CUFFT_C2C 1))
			  ,(cufftprint `(cufftExecC2C plan
						      d_signal ;; in
						      d_signal_out ;; out
						      CUFFT_FORWARD)))
			#+nil
			(do0
			 "// copy data back"
			 (progn
			  (let ((h_signal3 (static_cast<Complex*> (malloc memsize)))
				(v ("reinterpret_cast<std::complex<float>*>" h_signal3)))
			    ,(cuprint `(cudaMemcpy h_signal3 ;; dst
						   d_signal_out ;; src
						   memsize cudaMemcpyDeviceToHost)
				      `(memsize))
			    ,(logprint "runProcessing"
				       `(,@(loop for i below 5 collect
						`(aref v
						       ,i))))
			    (free h_signal3))))
			
			(do0
			 
			 ,(cuprint `(cudaMalloc (reinterpret_cast<void**> &d_kernel)
						memsize)
				   `(memsize))
			 ,(cuprint `(cudaMalloc (reinterpret_cast<void**> &d_kernel_out)
						memsize)
				   `(memsize)))
			#+nil (setf fdec (dot (aref dfc.iloc 0)
				 fdec)
		       xs_a_us  (/ (cp.arange (aref ss.shape 1)) fdec)
		       xs_off (- xs_a_us (* .5 (aref dfc.txpl 0))
				 .5)
		       xs_mask (& (< (* -.5 (aref dfc.txpl 0)) xs_off)
				  (< xs_off (* .5 (aref dfc.txpl 0))))
		       arg_nomchirp (* -2 np.pi
				       (+ (* xs_off  (+ (aref dfc.txpsf 0)
							(* .5
							   (aref dfc.txpl 0)
							   (aref dfc.txprr 0))))
					  (* (** xs_off 2)
					     .5
					     (aref dfc.txprr 0))))
		       z (* xs_mask (np.exp (* 1j arg_nomchirp))))

			(do0
			 (let ,(loop for e in `(txprr
						txpl
						txpsf
						fdec
					;ses_ssb_tx_pulse_number
						)
				  collect
				    `(,(format nil "_~a" e) (aref ,e index)))
			   (let ((h_kernel ("static_cast<std::complex<float>*>"
					    (malloc memsize)))
				 (xs_off 0))
			     (for (("int i" 0)
				   (< xs_off _txpl)
				   (incf i))
			       "const std::complex<float> imag(0, 1);"
			       (let ((xs_us (/ i _fdec))
				     )
				 (setf xs_off (- xs_us (* .5 _txpl)
						 .5))
				 (let ((arg (* 2 ;; -1 * -2 for conjugation
					       ,(coerce pi 'single-float) (+ (* xs_off (+ _txpsf
											    (* .5s0
											       _txpl
											       _txprr)))
									       (* (* xs_off xs_off)
										  .5s0 _txprr))))
				      (cplx ("std::exp" (* imag arg))))
				  (setf (aref h_kernel i) cplx))))
			     ,(cuprint `(cudaMemcpy d_kernel ;; dst
						    h_kernel ;; src
						    memsize cudaMemcpyHostToDevice)
				       `(memsize))
			     ,(cufftprint `(cufftExecC2C plan
						      d_kernel ;; in
						      d_kernel_out ;; out
						      CUFFT_FORWARD))
			     (free h_kernel))))
			(do0
			 ;; numblocks, threads per block
			 ("ComplexPointwiseMul<<<128,1024>>>" d_signal_out
							    d_kernel_out
							    range)
			 (do0
			 "// copy data back"
			 (progn
			  (let ((h_signal3 (static_cast<Complex*> (malloc memsize)))
				(v ("reinterpret_cast<std::complex<float>*>" h_signal3)))
			    ,(cuprint `(cudaMemcpy h_signal3 ;; dst
						   d_signal_out ;; src
						   memsize cudaMemcpyDeviceToHost)
				      `(memsize))
			    ,(logprint "runProcessing"
				       `(,@(loop for i below 5 collect
						`(aref v
						       ,i))))
			    (free h_signal3))))
			 ,(cufftprint `(cufftExecC2C plan
						     d_signal_out ;; in
						     d_signal ;; out
						     CUFFT_INVERSE))
			 ,(cuprint `(cudaMemcpy h_signal2 ;; dst
						d_signal ;; src
						memsize cudaMemcpyDeviceToHost
						)
				   `(memsize)))
			(do0
			 ,(cufftprint `(cufftDestroy plan))
			 ,(cuprint `(cudaFree d_signal))
			 ,(cuprint `(cudaFree d_signal_out))
			 ,(cuprint `(cudaFree d_kernel))
			 ,(cuprint `(cudaFree d_kernel_out))
			 (return ("reinterpret_cast< std::complex<float>* >" h_signal2))
			 ))
		      )
		    (defun cleanupProcessing ()))))
  
  (define-module
      `(gui ()
	    (do0
	     "// https://youtu.be/nVaQuNXueFw?t=317"
	      (include "imgui/imgui.h"
		       "imgui/imgui_impl_glfw.h"
		       "imgui/imgui_impl_opengl2.h")  
	      (defun initGui ()
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
	      (defun drawGui ()
		(ImGui_ImplOpenGL2_NewFrame)
		(ImGui_ImplGlfw_NewFrame)
		("ImGui::NewFrame")
		#+nil
		(let ((b true))
		 ("ImGui::ShowDemoWindow" &b))
		(do0
		 ,@(loop for (e mi ma) in `((a 0 100)
						  ) collect
			(let ((name (format nil "slider_~a" e)))
			  `(let ((,name ,(floor (* .5 (+ mi ma)))))
			     (declare (type "static int" ,name))
			     ("ImGui::SliderInt"
			      (string ,name)
			      (ref ,name)
			      ,mi
			      ,(- ma 1)
			      ))))
		 (setf ,(g `_range_line)
		       (runProcessing slider_a))

		 (progn
		  (do0
		  
		   "// plot raw data (real)"
		   (let ((p ("reinterpret_cast<std::complex<float>*>" ,(g `_mmap_data)))
			 (range ,(g `_range))
			 (h_signal (ref (aref p (* range slider_a))))
			 (range_raw_re (static_cast<float*> (malloc (* (sizeof float) range))))
			 )
		     (declare (type "static float*" range_raw_re))
		     (dotimes (i range)
		       (setf (aref range_raw_re i)
			     ("std::real" (aref h_signal
						i))))
		     (when range_raw_re
		       (do0
			("ImGui::PlotLines"
			 (string "range_raw_re") ;; label
			 ;("reinterpret_cast<float*>" h_signal)
					range_raw_re ;; values
			 range			  ;; count
			 0			  ;; offset
			 NULL			  ;; overlay_text
			 FLT_MAX		  ;; scale_min
			 FLT_MAX		  ;; scale_max
			 (ImVec2 3700 400)	  ;; graph_size
			 (sizeof float)		  ;;stride
			 ))))))
		 
		 (let ((range ,(g `_range))
		       ;(memsize (* (sizeof Complex) range))
		       (range_abs (static_cast<float*> (malloc (* (sizeof float) range))))
		       (range_re (static_cast<float*> (malloc (* (sizeof float) range)))))
		   (declare (type "static float*" range_abs range_re))

		   (when ,(g `_range_line)
		    (do0
		     (dotimes (i range)
		       (setf (aref range_abs i)
			     ("std::abs" (aref ,(g `_range_line)
					       i)))
		       (setf (aref range_re i)
			     ("std::real" (aref ,(g `_range_line)
					       i))))
		    
		    
		     ("ImGui::PlotLines"
		      (string "range_abs") ;; label
		      range_abs	       ;; values
		      range	       ;; count
		      0		       ;; offset
		      NULL	       ;; overlay_text
		      FLT_MAX	       ;; scale_min
		      FLT_MAX	       ;; scale_max
		      (ImVec2 3200 400) ;; graph_size
		      (sizeof float)	;;stride
		      )
		     ("ImGui::PlotLines"
		      (string "range_re") ;; label
		      range_re	       ;; values
		      range	       ;; count
		      0		       ;; offset
		      NULL	       ;; overlay_text
		      FLT_MAX	       ;; scale_min
		      FLT_MAX	       ;; scale_max
		      (ImVec2 3200 400) ;; graph_size
		      (sizeof float)	;;stride
		      )))))
		("ImGui::Render")
		(ImGui_ImplOpenGL2_RenderDrawData
		 ("ImGui::GetDrawData"))
		))))
  
  
  (progn
    (with-open-file (s (asdf:system-relative-pathname 'cl-cpp-generator2
						 (merge-pathnames #P"proto2.h"
								  *source-dir*))
		       :direction :output
		       :if-exists :supersede
		       :if-does-not-exist :create)
      (loop for e in (reverse *module*) and i from 0 do
	   (destructuring-bind (&key name code) e
	     (let ((cuda (cl-ppcre:scan "cuda" (string-downcase (format nil "~a" name)))))
	       (unless cuda
		(emit-c :code code :hook-defun 
			#'(lambda (str)
			    (format s "~a~%" str))))
	       
	       (write-source (asdf:system-relative-pathname
			      'cl-cpp-generator2
			      (format nil
				      "~a/doppler_~2,'0d_~a.~a"
				      *source-dir* i name
				      (if cuda
					  "cu"
					  "cpp")))
			     code)))))
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
		    (include <complex>)
		    " "
		    (include <cmath>)
		    " "
		    
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

    (let ((data
	   `(do0
	     ,@(loop for e_ in `(txprr
				 txpl
				 txpsf
				 fdec
				 ;ses_ssb_tx_pulse_number
				 )
					;(elt *dfa-csv* 0)
		  collect
		    (let* ((e e_ ;(format nil "ranked_~a" e_)
			     )
			   (colindex (position (format nil "~a" e) (elt *dfa-csv* 0) :test #'string=))
			   (name `(aref ,e ,(- (length *dfa-csv*) 1))))
		      `(let ((,name (curly ,@(loop for row in (cdr *dfa-csv*) collect
						  (elt row colindex)))))
			 (declare (type float ,name))))))))
      (write-source (asdf:system-relative-pathname 'cl-cpp-generator2 (merge-pathnames
								     #P"data.h"
								     *source-dir*))
		  `(do0
		    "#ifndef DATA_H"
		    " "
		    "#define DATA_H"
		    " "
		    ,data
		    " "
		    "#endif"
		    " ")))
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
		    (include <complex>)
		    " "
		    (include <cmath>)
		    " "
		    
		    ,(emit-globals)
		    " "
		    "#endif"
		    " "))
    
    
    
    ;; we need to force clang-format to always have the return type in the same line as the function: PenaltyReturnTypeOnItsOwnLine
					;(sb-ext:run-program "/bin/sh" `("gen_proto.sh"))
    #+nil (sb-ext:run-program "/usr/bin/make" `("-C" "source" "-j12" "proto2.h"))))

