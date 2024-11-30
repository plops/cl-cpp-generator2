(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more
						    )))
  (setf *features* (set-exclusive-or *features* (list ;:more
						      ))))

(let ()
  (defparameter *source-dir* #P"example/162_glfwpp_grating/source01/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)

  (defun lprint (&key (msg "")
		   (vars nil)
		   )
    `(<< std--cout
	 (std--format
	  (string ,(format nil "(~a~{:~a '{}'~^ ~})\\n"
			   msg
			   (loop for e in vars collect (emit-c :code e  :omit-redundant-parentheses t)) ))
	  ,@vars))
    #+nil
    `(<< std--cout
	 (string ,(format nil "~a"
			  msg
			
			  ))
	 ,@(loop for e in vars
		 appending
		 `((string ,(format nil " ~a='" (emit-c :code e :omit-redundant-parentheses t)))
		   ,e
		   (string "' ")))   
	 std--endl))
  
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2 
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0

     (include /home/martin/src/glfw/deps/linmath.h)
     (include<>
      iostream
      format

      ;GL/glew.h
      
      glfwpp/glfwpp.h
      
      ;Eigen/Core 
				
      ;thread
      ;popl.hpp
      )
     ;"using namespace std;"
     ;"using namespace Eigen;"
     
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
      
       #+Nil ,(let ((l `(
		   (:name numberPoints :default 1024 :short p)
		   )))
	  `(let ((op (popl--OptionParser (string "allowed options")))
		 ,@(loop for e in l collect
				    (destructuring-bind (&key name default short (type 'int)) e
				      `(,name (,type ,default))))
		 ,@(loop for e in `((:long help :short h :type Switch :msg "produce help message")
				    (:long verbose :short v :type Switch :msg "produce verbose output")
				    (:long mean :short m :type Switch :msg "Print mean and standard deviation statistics, otherwise print median and mean absolute deviation from it")
				    ,@(loop for f in l
					    collect
					    (destructuring-bind (&key name default short (type 'int)) f
					      `(:long ,name
						:short ,short
						:type ,type :msg "parameter"
						:default ,default :out ,(format nil "&~a" name))))

				    )
			 appending
			 (destructuring-bind (&key long short type msg default out) e
			   `((,(format nil "~aOption" long)
			      ,(let ((cmd `(,(format nil "add<~a>"
						     (if (eq type 'Switch)
							 "popl::Switch"
							 (format nil "popl::Value<~a>" type)))
					    (string ,short)
					    (string ,long)
					    (string ,msg))))
				 (when default
				   (setf cmd (append cmd `(,default)))
				   )
				 (when out
				   (setf cmd (append cmd `(,out)))
				   )
				 `(dot op ,cmd)
				 ))))
			 ))
	     (op.parse argc argv)
	     (when (helpOption->is_set)
	       (<< cout
		   op
		   endl)
	       (exit 0))
	     ))

       ;,(lprint :vars `((thread--hardware_concurrency)))

       (let ((GLFW (glfw--init))
	     (hints (space glfw--WindowHints (designated-initializer
					:clientApi glfw--ClientApi--OpenGl
					:contextVersionMajor 2
					:contextVersionMinor 0))))
	 (hints.apply)
	 (let ((w 800)
	       (h 600)
	       (window (glfw--Window
			w h
			(string "GLFWPP Grating")
			)))
	   (glfw--makeContextCurrent window)
	   #+nil (when (!= GLEW_OK
		     (glewInit))
	     (throw (std--runtime_error (string "Could not initialize GLEW"))))
	   (while (!window.shouldClose)
		  (let ((time (glfw--getTime)))
		    (glClearColor 0s0 .0s0 .0s0 1s0)
		    (glClear GL_COLOR_BUFFER_BIT)

		    
		    
		    (glfw--pollEvents)


		    #+nil
		    (do0 (glPushMatrix)
			 (progn
			   (comments "Setup modelview matrix (flat XY view)")
			   "mat4x4 view;"
			   (space vec3 eye (curly 0s0 0s0 1s0))
			   (space vec3 center (curly 0s0 0s0 0s0))
			   (space vec3 up  (curly 0s0 1s0 0s0))
			   
			   (mat4x4_look_at view eye center up)
			   (glLoadMatrixf ("reinterpret_cast<const GLfloat*>" view))
			   ))

		    (do0
		     (glColor4f 1s0 1s0 1s0 1s0)
		     (glBegin GL_LINES)
		     (let ((skip 2)))
		     (let ((N (/ h skip))))
		     (let ((Nx (/ w skip))))
		     (dotimes (i N)
		       (let ((y (/ (- i (/ N 2)) (* (/ .5s0 skip) h)) )))
		       (glVertex2f -1s0 y)
		       (glVertex2f 1s0 y))

		     (dotimes (i Nx)
		       (let ((x (/ (- i (/ Nx 2)) (* (/ .5s0 skip) w)) )))
		       (glVertex2f x -1s0)
		       (glVertex2f x 1s0))
		     (glEnd))
		    
		    #+nil (glPopMatrix)
		    (window.swapBuffers))))
	 )

       (return 0)))
   :omit-parens t
   :format t
   :tidy nil))
