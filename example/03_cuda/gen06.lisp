(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

;; 2019 duane storti cuda for engineers p. 90

;; http://www.findinglisp.com/blog/2004/06/basic-automaton-macro.html

(progn
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
		     <cuda_gl_interop.h>)

	    (defun key_callback (window key scancode action mods)
	      (declare (type GLFWwindow* window)
                      (type int key scancode action mods))
             (when (and (or (== key GLFW_KEY_ESCAPE)
                            (== key GLFW_KEY_Q))
                        (== action GLFW_PRESS)) 
               (glfwSetWindowShouldClose window GLFW_TRUE)))
	    (defun error_callback (err description)
             (declare (type int err)
                      (type "const char*" description)
                      )
             (fprintf stderr (string "Error: %s\\n")
                      description))

	    "using namespace std;"
	    (do0
	     "struct uchar4;"
	     (defstruct0 BC
		 (x int)
	       (y int)
	       (rad float)
	       (chamfer int)
	       (t_s float)
	       (t_a float)
	       (t_g float))
	     (do0
	      "enum {TX=32, TY=32,RAD=1, ITERS_PER_RENDER=50};"
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
	       (cudaGraphicsMapResources 1 &g_cuda_pbo_resource 0)
	       (let ((d_out (static_cast<uchar4*> 0)))
		 (cudaGraphicsResourceGetMappedPointer (reinterpret_cast<void**> &d_out)
						       nullptr
						       g_cuda_pbo_resource)
		 (dotimes (i ITERS_PER_RENDER)
		  (kernelLauncher d_out d_temp w h bc))
		 (cudaGraphicsUnmapResources 1 &g_cuda_pbo_resource 0))
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
	      (do0 (glBegin GL_QUADS)
		   ,@(loop for (e f) in `((0 0) 
					  (0 1)
					  (1 1)
					  (1 0))
			collect
			  `(do0
			    (glTexCoord2f ,e ,f)
			    (glVertex2f ,(* 2 (- e .5s0))
					,(* 2 (- f .5s0)))))
		   (glEnd))
	      (glDisable GL_TEXTURE_2D))
	    #+nil (defun _post_call_callback_default (name funcptr len_args a...)
	      (declare (type "const char*" name)
		       (type void* funcptr)
		       (type int len_args)
		       (type " " a...))
	      (let ((error_code (glad_glGetError)))
		(unless (== GL_NO_ERROR
			    error_code)
		  (<< cerr (string "glad error: ")
		      error_code
		      (string " ")
		      name))))
	    (defun main ()
	      (declare (values int))
	      (<< cout (string "bla") endl)
	      (when (glfwInit)
		
		(glfwSetErrorCallback error_callback)

		(do0
		 (glfwWindowHint GLFW_CONTEXT_VERSION_MAJOR 4)
		 (glfwWindowHint GLFW_CONTEXT_VERSION_MINOR 0))

		(let ((window (glfwCreateWindow 640 480 (string "cuda interop")
						NULL NULL)))

		  (assert window)
		  (glfwSetKeyCallback window key_callback)
		  (glfwMakeContextCurrent window)
		  (assert (gladLoadGL))
		  (<< cout (string "GL version " ) GLVersion.major
		      (string " ") GLVersion.minor endl)
		  (gladLoadGLLoader (reinterpret_cast<GLADloadproc> glfwGetProcAddress))
		  ;(glad_set_post_callback _post_call_callback_default)
		  (do0
                   (let ((width)
                         (height))
                     (declare (type int width height))
                     (glfwGetFramebufferSize window &width &height)
                     (glViewport 0 0 width height))
                   (glfwSwapInterval 1)
                   (glClearColor 0 0 0 0)
                   #+nil (do0 (glBlendFunc GL_SRC_ALPHA GL_ONE_MINUS_SRC_ALPHA)
			      (glEnable GL_BLEND)
			      (glEnable GL_LINE_SMOOTH)
			      (glHint GL_LINE_SMOOTH GL_NICEST))
		   (do0
		    (let ((d_temp  (static_cast<float*> 0))
			  (d_out (static_cast<uchar4*> 0))
			  (bc (cast BC (curly
					(/ width 2)
					(/ height 2)
					(/ width 10s0)
					150
					212s0
					70s0
					0s0))))
		      (cudaMalloc &d_temp (* width height (sizeof *d_temp)))
		      (resetTemperature d_temp width height bc)))
		   (do0
		    (let ((pbo 0)
			  (tex 0))
		      (declare (type GLuint pbo tex))
		      (glGenBuffers 1 &pbo)
		      (glBindBuffer GL_PIXEL_UNPACK_BUFFER pbo)
		      (glBufferData GL_PIXEL_UNPACK_BUFFER (* width height (sizeof GLubyte) 4)
				    0 GL_STREAM_DRAW)
		      (glGenTextures GL_TEXTURE_2D &tex)
		      (glBindTexture GL_TEXTURE_2D tex)
		      (glTexParameteri GL_TEXTURE_2D GL_TEXTURE_MIN_FILTER GL_NEAREST)
		      (cudaGraphicsGLRegisterBuffer &g_cuda_pbo_resource
						    pbo
						    cudaGraphicsMapFlagsWriteDiscard)
		      

		      (while (not (glfwWindowShouldClose window))
			(glfwPollEvents)
			(let ((time (glfwGetTime)))
			  (glClear GL_COLOR_BUFFER_BIT)
					;(kernelLauncher d_out d_temp width height bc)
			  (render d_temp width height bc)
			  (draw_texture width height)
			  (glfwSwapBuffers window)))
		      (when pbo
			(cudaGraphicsUnregisterResource g_cuda_pbo_resource)
			(glDeleteBuffers 1 &pbo)
			(glDeleteTextures 1 &tex))
		      (glfwDestroyWindow window))))))
	      (do0
	       
	       (glfwTerminate))))))
    (write-source *code-file* code)))
