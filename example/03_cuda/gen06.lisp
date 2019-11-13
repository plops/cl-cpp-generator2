s(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

;; 2019 duane storti cuda for engineers p. 90

;; http://www.findinglisp.com/blog/2004/06/basic-automaton-macro.html

(progn
    (defun vkprint (call &optional rest)
      `(do0 ,call
	    (<< "std::cout"
	      (- (dot ("std::chrono::high_resolution_clock::now")
		    (time_since_epoch)
		    (count))
		 g_start)
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
	      "std::endl")
	  )
      )
    (defun cuprint (call &optional rest)
      `(progn (let ((r ,call))
		(<< "std::cout"
		  (- (dot ("std::chrono::high_resolution_clock::now")
			  (time_since_epoch)
			  (count))
		     g_start)
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
    (defun cuprint_ (call &optional rest)
      `(progn (let ((r ,call))
		(unless (== cudaSuccess r)
		 (<< "std::cout"
		     (- (dot ("std::chrono::high_resolution_clock::now")
			     (time_since_epoch)
			     (count))
			g_start)
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
		(assert (== cudaSuccess r)))
	  )
    )

  (defparameter *code-file* (asdf:system-relative-pathname 'cl-cpp-generator2 "example/03_cuda/source/06interop.cu"))
  (let* ((code
	  `(do0
	    ;; -Xcompiler=-fsanitize=address
	    "// glad --generator=c-debug --spec=gl --out-path=GL --extensions=GL_EXT_framebuffer_multisample,GL_EXT_texture_filter_anisotropic"
	    "// nvcc -o 06interop GL/src/glad.c 06interop.cu -IGL/include -lglfw -lGL --std=c++14 -O3 -g -Xcompiler=-march=native -Xcompiler=-ggdb -ldl"
	    "// note that nvcc requires gcc 8"
	    "// nvprof 06interop"
					;"#define GLAD_DEBUG"
	    (include <glad/glad.h>)
	    " "
	    
	    (include  
		     <GLFW/glfw3.h>
		     <cassert>
		     <cstdio>
		     <iostream>
		     <cuda_runtime.h>
		     <GL/glu.h>
		     <chrono>)

	    " "
	    (include <cuda_gl_interop.h>)


	    (do0
	     "struct uchar4;"
	     (defstruct0 BC
		 (x int)
	       (y int)
	       (rad float)
	       (chamfer int)
	       (t_s float)
	       (t_a float)
	       (t_g float)
	       (d_temp float*)
	       (width int)
	       (height int))
	     (do0
	      "enum {TX=32, TY=32,RAD=1, ITERS_PER_RENDER=1};"
	      (defun divUp (a b)
		(declare (type int a b)
			 (values int))
		(return (/ (+ a b -1)
			   b)))
	      (defun clip (n)
		(declare (type int n)
			 (values "__device__ unsigned char"))
		(if (< 255 n)
		    (return 255)
		    (if (< n 0)
			(return 0)
			(return n))))
	      (defun idxClip (n ma)
		(declare (type int n ma)
			 (values "__device__ int"))
		(if (< (- ma 1) n)
		    (return (- ma 1))
		    (if (< n 0)
			(return 0)
			(return n))))
	       (defun flatten (col row w h)
		(declare (type int col row w h)
			 (values "__device__ int"))
		(return (+ (idxClip col w)
			   (* w (idxClip row h)))))))
	    
	    (let ((g_start (,(format nil "static_cast<~a>" (emit-c :code `(typeof (dot ("std::chrono::high_resolution_clock::now")
										       (time_since_epoch)
										       (count)))))
			     0))))
	    (defun resetKernel (d_temp w h bc)
	      (declare (type float* d_temp)
		       (type int w h)
		       (type BC bc)
		       (values "__global__ void"))
	      (let ((col (+ (* blockIdx.x
			       blockDim.x)
			    threadIdx.x))
		    (row (+ (* blockIdx.y
			       blockDim.y)
			    threadIdx.y)))
		(when (or (<= w col)
			  (<= h row))
		  (return)
		  )
		(setf (aref d_temp (+ col (* row w)))
		      bc.t_a)))
	    (defun resetTemperature (d_temp w h bc)
	      (declare (type float* d_temp)
		       (type int w h)
		       (type BC bc))
	      (let (((blockSize TX TY) )
		    ((gridSize (divUp w TX)
			       (divUp h TY))))
		(declare (type "const dim3"
			       (blockSize TX TY)
			       (gridSize (divUp w TX)
					 (divUp h TY))))
		("resetKernel<<<gridSize,blockSize>>>" d_temp w h bc)))
	    (defun key_callback (window key scancode action mods)
	      (declare (type GLFWwindow* window)
		       (type int key scancode action mods))
	      (let ((bc (static_cast<BC*> (glfwGetWindowUserPointer window)))
		    (DT 1s0)))
	      (when (and (or (== key GLFW_KEY_ESCAPE)
			     (== key GLFW_KEY_Q))
			 (== action GLFW_PRESS)) 
		,(vkprint `(glfwSetWindowShouldClose window GLFW_TRUE)))
	      (when (and (or (== key GLFW_KEY_M))
			 (== action GLFW_PRESS)) 
		,(vkprint `(resetTemperature bc->d_temp bc->width bc->height *bc)))
	      ,@(loop for e in `((t_s 1 2 DT)
				 (t_a 3 4 DT)
				 (t_g 5 6 DT)
				 (chamfer 7 8 1)
				 (rad 9 0 2s0)
				 ) collect
		     (destructuring-bind (var up down delta) e
		      `(do0
			,@(loop for f in `(up-dir down-dir) collect 
			       `(when (and (== key ,(format nil "GLFW_KEY_~a" (case f
										(up-dir up)
										(down-dir down))))
					   (or (== action GLFW_PRESS)
					        (== action GLFW_REPEAT)))
				  (,(case f
				      (up-dir 'incf)
				      (down-dir 'decf)) (-> bc ,var) ,delta))))))
	      (let ((s[1024]))
		(declare (type "char" s[1024]))
		(snprintf s 1023 (string "cuda pipe=%.2g air=%.2g ground=%.2g chamfer=%d radius=%.2g")
			  ,@(loop for e in `(t_s t_a t_g chamfer rad) collect
				 `(-> bc ,e))
			  )
	       (glfwSetWindowTitle window s)))
	    (defun error_callback (err description)
             (declare (type int err)
                      (type "const char*" description)
                      )
             (fprintf stderr (string "Error: %s\\n")
                      description))

	    "using namespace std;"
	    

	    
	    
	    (defun tempKernel (d_out d_temp w h bc)
	      (declare (type float* d_temp)
		       (type uchar4* d_out)
		       (type int w h)
		       (type BC bc)
		       (values "__global__ void"))
	      (let ((s_in[])
		    (col (+ (* blockIdx.x
			       blockDim.x)
			    threadIdx.x))
		    (row (+ (* blockIdx.y
			       blockDim.y)
			    threadIdx.y)))
		(declare (type "extern __shared__ float" s_in[]))
		(when (or (<= w col)
			  (<= h row))
		  (return))
		(let ((idx (flatten col row w h))
		      (s_w (+ blockDim.x (* 2 RAD)))
		      (s_h (+ blockDim.y (* 2 RAD)))
		      (s_col (+ threadIdx.x RAD))
		      (s_row (+ threadIdx.y RAD))
		      (s_idx (flatten s_col s_row s_w s_h)))
		  ,@(loop for (e f) in `((x 0)
					 (y 0)
					 (z 0)
					 (w 255))
			 collect
			 `(setf (dot (aref d_out idx)
				     ,e) ,f))
		  (do0
		   (setf (aref s_in s_idx)
			 (aref d_temp idx))
		   (when (< threadIdx.x RAD)
		     (setf (aref s_in (flatten (- s_col RAD)
					       s_row
					       s_w
					       s_h))
			   (aref d_temp (flatten (- col RAD)
						 row
						 w
						 h)))
		     (setf (aref s_in (flatten (+ s_col RAD)
					       s_row
					       s_w
					       s_h))
			   (aref d_temp (flatten (+ col RAD)
						 row
						 w
						 h))))
		   (when (< threadIdx.y RAD)
		     (setf (aref s_in (flatten s_col
					       (- s_row RAD)
					       s_w
					       s_h))
			   (aref d_temp (flatten col
						 (- row RAD)
						 w
						 h)))
		     (setf (aref s_in (flatten s_col
					       (+ s_row blockDim.y)
					       s_w
					       s_h))
			   (aref d_temp (flatten col
						 (+ row blockDim.y)
						 w
						 h))))
		   (let ((dSq (+ (* (- col bc.x)
				    (- col bc.x))
				 (* (- row bc.y)
				    (- row bc.y)))))
		     (when (< dSq (* bc.rad bc.rad))
		       (setf (aref d_temp idx)
			     bc.t_s)
		       (return)))
		   (when (or (== 0 col)
			     (== (- w 1) col)
			     (== 0 row)
			     (< (+ col row)
				bc.chamfer)
			     (< (- w bc.chamfer)
				(- col row)))
		     (setf (aref d_temp idx)
			   bc.t_a)
		     (return))
		   (when (== (- h 1)
			     row)
		     (setf (aref d_temp idx)
			   bc.t_g)
		     (return))
		   (__syncthreads)
		   (let ((temp (* .25s0
				  (+ (aref s_in (flatten (- s_col 1) s_row s_w s_h))
				     (aref s_in (flatten (+ s_col 1) s_row s_w s_h))
				     (aref s_in (flatten s_col (- s_row 1) s_w s_h))
				     (aref s_in (flatten s_col (+ s_row 1) s_w s_h))))))
		     (setf (aref d_temp idx)
			   temp)
		     (let ((intensity (clip (cast int temp))))
		       (setf (dot (aref d_out idx) x) intensity)
		       (setf (dot (aref d_out idx) z) (- 255 intensity))))))))
	    
	    (defun kernelLauncher (d_out d_temp w h bc)
	      (declare (type float* d_temp)
		       (type uchar4* d_out)
		       (type int w h)
		       (type BC bc))
	      (let (((blockSize TX TY) )
		    ((gridSize (divUp w TX)
			       (divUp h TY)))
		    (smSz (* (sizeof float)
			     (+ TX (* 2 RAD))
			     (+ TY (* 2 RAD)))))
		(declare (type "const dim3"
			       (blockSize TX TY)
			       (gridSize (divUp w TX)
					 (divUp h TY))))
		("tempKernel<<<gridSize,blockSize,smSz>>>" d_out d_temp w h bc)))

	    (do0
	     (let ((g_cuda_pbo_resource ("static_cast<struct cudaGraphicsResource*>" 0))
		   ))
	     (defun render (d_temp w h bc)
	       (declare (type int w h)
			(type BC bc)
			(type float* d_temp ))
	       ,(cuprint_ `(cudaGraphicsMapResources 1 &g_cuda_pbo_resource 0) `(g_cuda_pbo_resource))
	       (let ((d_out (static_cast<uchar4*> 0)))
		 ,(cuprint_ `(cudaGraphicsResourceGetMappedPointer (reinterpret_cast<void**> &d_out)
								nullptr
								g_cuda_pbo_resource))
		 #+nil (let ((r ))
		   (unless (== cudaSuccess r)
		     ,(vkprint `() `(r (cudaGetErrorString r)))))
		 
		 (dotimes (i ITERS_PER_RENDER)
		   (kernelLauncher d_out d_temp w h bc))
		 ,(cuprint_ `(cudaGraphicsUnmapResources 1 &g_cuda_pbo_resource 0)))
	       ))

	    (defun draw_texture (w h)
	      (declare (type int w h))
	      (glTexImage2D GL_TEXTURE_2D
			    0
			    GL_RGBA
			    w h
			    0
			    GL_RGBA
			    GL_UNSIGNED_BYTE
			    nullptr)
	      (glEnable GL_TEXTURE_2D)
	      (do0 (glBegin GL_QUADS
			    )
		   #+nil (progn
		     (let ((gl_error_code (glGetError))
			  (gl_error_string (gluErrorString gl_error_code)))
		      ,(vkprint `() `(0 gl_error_code gl_error_string))))
		   ,@(loop for (e f) in `((0 0) 
					  (0 1)
					  (1 1)
					  (1 0))
			  and count from 0
			collect
			  `(do0
			    
			    (glTexCoord2f ,e ,f)
			    #+nil (progn (let ((gl_error_code (glGetError))
				   (gl_error_string (gluErrorString gl_error_code)))
			       ,(vkprint `() `(,(+ 10 count) gl_error_code gl_error_string))))
			    (glVertex2f ,(* 2 (- e .5s0))
					     ,(* 2 (- f .5s0)))
			    #+nil (progn (let ((gl_error_code (glGetError))
				   (gl_error_string (gluErrorString gl_error_code)))
			       ,(vkprint `() `(,(+ 20 count) gl_error_code gl_error_string))))))
		   (glEnd))
	      #+nil (let ((gl_error_code (glGetError))
		    (gl_error_string (gluErrorString gl_error_code)))
		,(vkprint `() `(30 gl_error_code gl_error_string)))
	      (glDisable GL_TEXTURE_2D))

	    (defun main ()
	      (declare (values int))
	      (setf g_start (dot ("std::chrono::high_resolution_clock::now")
										       (time_since_epoch)
										       (count)))

	      (do0
					;cudaSetDevice(0);
					;cudaDeviceSynchronize();
					;cudaThreadSynchronize();
	       (let ((n_cuda 0))
		 ,(cuprint `(cudaGetDeviceCount &n_cuda) `(n_cuda)))
	       ,(cuprint `(cudaSetDevice 0))
	       ;,(cuprint `(cudaDeviceSynchronize))
	       ;,(cuprint `(cudaThreadSynchronize))
	       )
	      
	      (when (glfwInit)
		,(vkprint `(glfwSetErrorCallback error_callback))
		
		#+nil (do0
		 (glfwWindowHint GLFW_CONTEXT_VERSION_MAJOR 4)
		 (glfwWindowHint GLFW_CONTEXT_VERSION_MINOR 0))

		(let ((window (glfwCreateWindow 640 480 (string "cuda interop")
						NULL NULL)))

		  (assert window)
		  ,(vkprint `(glfwSetKeyCallback window key_callback) `(window))
		  ,(vkprint `(glfwMakeContextCurrent window))
		  (assert (gladLoadGL))
		  (<< cout (string "GL version " ) GLVersion.major
		      (string " ") GLVersion.minor endl)
		  ,(vkprint `(gladLoadGLLoader (reinterpret_cast<GLADloadproc> glfwGetProcAddress)))
		  (do0
                   (let ((width)
                         (height))
                     (declare (type int width height))
                     ,(vkprint `(glfwGetFramebufferSize window &width &height) `(width height))
                     ,(vkprint `(glViewport 0 0 width height)))
                   ,(vkprint `(glfwSwapInterval 1))
                   ,(vkprint `(glClearColor 0 0 0 0))
                   #+nil (do0 (glBlendFunc GL_SRC_ALPHA GL_ONE_MINUS_SRC_ALPHA)
			      (glEnable GL_BLEND)
			      (glEnable GL_LINE_SMOOTH)
			      (glHint GL_LINE_SMOOTH GL_NICEST))
		   (do0
		    (let ((d_temp  (static_cast<float*> 0))
			  ;(d_out (static_cast<uchar4*> 0))
			  )
		      ,(cuprint `(cudaMalloc &d_temp (* width height (sizeof float)))
				`(width height (/ (* width height (sizeof float))
						  (* 1024 1024s0))))
		      (let ((bc (cast BC (curly
					  (/ width 2)
					  (/ height 2)
					  (/ width 10s0)
					  150
					  212s0
					  70s0
					  0s0
					  d_temp
					  width
					  height))))
			(glfwSetWindowUserPointer window (static_cast<void*> &bc))
		       (resetTemperature d_temp width height bc))))
		   (do0
		    (let ((pbo 0)
			  (tex 0))
		      (declare (type GLuint pbo tex))
		      ,(vkprint `(glGenBuffers 1 &pbo) `(pbo))
		      ,(vkprint `(glBindBuffer GL_PIXEL_UNPACK_BUFFER pbo) `(pbo))
		      ,(vkprint `(glBufferData GL_PIXEL_UNPACK_BUFFER
						    (* width height (sizeof GLubyte) 4)
						    0 GL_STREAM_DRAW) `(width height (/ (* width height (sizeof GLubyte) 4)
											(* 1024 1024s0))))
		      
		      (do0
		       ,(vkprint `(glGenTextures 1 &tex) `(tex))
		       ,(vkprint `(glBindTexture GL_TEXTURE_2D tex))
		       ,(vkprint `(glTexParameteri GL_TEXTURE_2D GL_TEXTURE_MIN_FILTER GL_NEAREST)))
		      ,(cuprint `(cudaGraphicsGLRegisterBuffer &g_cuda_pbo_resource
						      pbo
						      cudaGraphicsMapFlagsWriteDiscard
						      ;; cuda will not read from here and discards all contents by overwriting it
						      )
				`(g_cuda_pbo_resource
				  pbo))
		      

		      (while (not (glfwWindowShouldClose window))
			(glfwPollEvents)
			(let ((time (glfwGetTime)))
			  (glClear GL_COLOR_BUFFER_BIT)
					;(kernelLauncher d_out d_temp width height bc)
			  (render d_temp width height bc)
			  (draw_texture width height)
			  (glfwSwapBuffers window)))
		      (when pbo
			,(vkprint `(cudaGraphicsUnregisterResource g_cuda_pbo_resource))
			,(vkprint `(glDeleteBuffers 1 &pbo))
			,(vkprint `(glDeleteTextures 1 &tex)))
		      ,(vkprint `(cudaFree d_temp))
		      ,(vkprint `(glfwDestroyWindow window)))))))
	      (do0
	       
	       ,(vkprint `(glfwTerminate)))))))
    (write-source *code-file* code)))
