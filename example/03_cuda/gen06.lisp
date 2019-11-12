(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

;; 2019 duane storti cuda for engineers p. 90

;; http://www.findinglisp.com/blog/2004/06/basic-automaton-macro.html

(progn
  (defparameter *code-file* (asdf:system-relative-pathname 'cl-cpp-generator2 "example/03_cuda/source/06interop.cu"))
  (let* ((code
	  `(do0
	    "// nvcc -o 06interop 06interop.cu -lglfw -lGL -march=native --std=c++14 -O3 -g"
	    "// note that nvcc requires gcc 8"
	    "// nvprof 06interop"
	    (include <GLFW/glfw3.h>
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

	    (defstruct0 BC
		(x int)
	      (y int)
	      (rad float)
	      (chamfer int)
	      (t_s float)
	      (t_a float)
	      (t_g float))

	    "enum {TX=32, TY=32};"
	    (defun divUp (a b)
	      (declare (type int a b)
		       (values int))
	      (return (/ (+ a b -1)
			 b)))

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
	    
	    (defun main ()
	      (declare (values int))
	      (<< cout (string "bla") endl)
	      (when (glfwInit)
		
		(glfwSetErrorCallback error_callback)

		(do0
		 (glfwWindowHint GLFW_CONTEXT_VERSION_MAJOR 2)
		 (glfwWindowHint GLFW_CONTEXT_VERSION_MINOR 0))

		(let ((window (glfwCreateWindow 640 480 (string "cuda interop")
						NULL NULL)))

		  (assert window)
		  (glfwSetKeyCallback window key_callback)
		  (glfwMakeContextCurrent window)
                      
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
			    (bc (cast BC (curly
					  (/ width 2)
					  (/ height 2)
					  (/ width 10s0)
					  150
					  212s0
					  70s0
					  0s0))))
			(cudaMalloc &d_temp (* width height (sizeof *d_temp)))
			(resetTemperature d_temp width height bc)
			
			))
                     )

		  (while (not (glfwWindowShouldClose window))
		     (glfwPollEvents)
                      (let ((time (glfwGetTime)))
                        (glClear GL_COLOR_BUFFER_BIT)
                        
                        (glfwSwapBuffers window)))
		   (glfwDestroyWindow window)))
	      
	      (glfwTerminate)))))
    (write-source *code-file* code)))
