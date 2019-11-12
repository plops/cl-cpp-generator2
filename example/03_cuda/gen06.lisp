(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2"))

(in-package :cl-cpp-generator2)

;; 2019 duane storti cuda for engineers p. 90

;; http://www.findinglisp.com/blog/2004/06/basic-automaton-macro.html

(defun define-automaton (state)
  
  ((init
    (t init_glfw))
   (init_glfw
    (t (do0
	(unless (glfwInit)
	  (setf state error)))))
   (idle
       (t idle
	  (case )))
   (error
       (t error))))

(progn
  (defparameter *code-file* (asdf:system-relative-pathname 'cl-cpp-generator2 "example/03_cuda/source/06interop.cu"))
  (let* ((code
	  `(do0
	    "// nvcc -o 06interop 06interop.cu -lglfw"
	    "// note that nvcc requires gcc 8"
	    "// nvprof 06interpo"
	    (include <GLFW/glfw3.h>)
	    (defun main ()
	      (declare (values int))
	      (unless (glfwInit)
		(let ((window (glfwCreateWindow 640 480 (string "cuda interop")
						NULL NULL)))
		  (unless window))
		(glfwTerminate))
	      ))))
    (write-source *code-file* code)))
 
