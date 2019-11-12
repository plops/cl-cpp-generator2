(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

;; 2019 duane storti cuda for engineers p. 90

;; http://www.findinglisp.com/blog/2004/06/basic-automaton-macro.html

(progn
  (defparameter *code-file* (asdf:system-relative-pathname 'cl-cpp-generator2 "example/03_cuda/source/06interop.cu"))
  (let* ((code
	  `(do0
	    "// nvcc -o 06interop 06interop.cu -lglfw -lGL"
	    "// note that nvcc requires gcc 8"
	    "// nvprof 06interpo"
	    (include <GLFW/glfw3.h>)
	    (defun key_callback (window key scancode action mods)
	      (declare (type GLFWwindow* window)
                      (type int key scancode action mods))
             (when (and (or (== key GLFW_KEY_ESCAPE)
                            (== key GLFW_KEY_Q))
                        (== action GLFW_PRESS))
               (glfwSetWindowShouldClose window GLFW_TRUE))

	      
	      )
	    (defun main ()
	      (declare (values int))
	      (unless (glfwInit)

		(glfwWindowHint GLFW_CONTEXT_VERSION_MAJOR 2)
		(glfwWindowHint GLFW_CONTEXT_VERSION_MINOR 0)

		(let ((window (glfwCreateWindow 640 480 (string "cuda interop")
						NULL NULL)))
		  (unless window
		    (exit -1))
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
                     (glBlendFunc GL_SRC_ALPHA GL_ONE_MINUS_SRC_ALPHA)
                     (glEnable GL_BLEND)
                     (glEnable GL_LINE_SMOOTH)
                     (glHint GL_LINE_SMOOTH GL_NICEST)
                     )

		   (while (not (glfwWindowShouldClose window))
		     (glfwPollEvents)
                      (let ((time (glfwGetTime)))
                        (glClear GL_COLOR_BUFFER_BIT)
                        
                        (glfwSwapBuffers window))))))


		  (glfwDestroyWindow window))
		(glfwTerminate))
	      ))))
    (write-source *code-file* code)))
 
