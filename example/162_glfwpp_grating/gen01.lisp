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
				
      thread
      chrono
					;popl.hpp
      )
     "using namespace std;"
     "using namespace chrono;"
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
	 (let ((w 512)
	       (h 512)
	       (window (glfw--Window
			w h
			(string "GLFWPP Grating")
			)))
	   

	   
	   (glfw--makeContextCurrent window)

	   (comments "an alternative to increase swap interval is to change screen update rate `xrandr --output HDMI-A-0 --mode 1920x1080 --rate 24`")
	   (glfw--swapInterval 1)
	   #+nil (when (!= GLEW_OK
			   (glewInit))
		   (throw (std--runtime_error (string "Could not initialize GLEW"))))
	   (let ((t0 (high_resolution_clock--now))))
	   (while (!window.shouldClose)
		  (let ((time (glfw--getTime)))
		    
		    (glfw--pollEvents)
		    #+nil (std--this_thread--sleep_for
			   (std--chrono--milliseconds (/ 1000 10)))
		    
		    (do0
		     (comments "show a sequence of horizontal bars and vertical bars that split the image into 1/2, 1/4th, ... . each image is followed by its inverted version. the lcd of the projector is too slow to show this pattern exactly with 60Hz. that is why we set swap interval to 2 (we wait for two frames for every image so that the display has time to settle)")

		     (do0
		      "static int current_level = 0;"
		      "static bool horizontal = true;"

		      (setf current_level (+ current_level 1))
		      (when (== current_level (* 8 2) )
			(setf horizontal !horizontal
			      current_level 0))
		      (let ((white (== 0 (% current_level 2))))))
		     
		     (if white
			 (do0 (glClearColor 0s0 0s0 0s0 1s0)
			      (glClear GL_COLOR_BUFFER_BIT)
			      (glColor4f 1s0 1s0 1s0 1s0))
			 (do0 (glClearColor 1s0 1s0 1s0 1s0)
			      (glClear GL_COLOR_BUFFER_BIT)
			      (glColor4f 0s0 0s0 0s0 1s0)))
		     
		     (glPushMatrix)
		     (glTranslatef -1s0 -1s0 0s0)
		     (glScalef (/ 2s0 w) (/ 2s0 h) 1s0)
		     (glBegin GL_QUADS)

		     (let ((level (/ current_level 2))))
		     (let ((y (/ 1024 (pow 2s0 level)))))
		     (if horizontal
			 (dotimes (i (pow 2s0 level))
			   (do0
			    (let ((x 512)
				  (o (* 2 i y))))
			    (glVertex2f 0 o)
			    (glVertex2f 0 (+ o y))
			    (glVertex2f x (+ o y))
			    (glVertex2f x o)))
			 (dotimes (i (pow 2s0 level))
			   (do0
			    (let ((x 512)
				  (o (* 2 i y))))
			    (glVertex2f  o 0)
			    (glVertex2f  (+ o y) 0)
			    (glVertex2f  (+ o y) x )
			    (glVertex2f  o x))))
		     (glEnd)
		     (glPopMatrix))
		    
		    #+nil
		    (do0
		     (comments "lines")
		     (glColor4f 1s0 1s0 1s0 1s0)
		     (glBegin GL_LINES)
		     (let ((skip 32)))
		     (space static int (setf offset 0))
		     (setf offset (% (+ offset 1) skip))
		     (let ((N (/ h skip))))
		     (let ((Nx (/ w skip))))
		     #+nil
		     (dotimes (i N)
		       (let ((y (/ (- i (/ N 2) (/ offset (* 1s0 N)))
				   (* (/ .5s0 skip) h)) )))
		       (glVertex2f -1s0 y)
		       (glVertex2f 1s0 y))
		     
		     (dotimes (i (+ Nx 2))
		       (let ((x (/ (- i (/ Nx 2) (/ offset (* 1s0 N)))
				   (* (/ .5s0 skip) w)) )))
		       (glVertex2f x -1s0)
		       (glVertex2f x 1s0))
		     (glEnd))
		    
		    #+nil (glPopMatrix)
		    (progn
		     (let ((t2 (high_resolution_clock--now))
			   (frameTimens (dot (duration_cast<nanoseconds> (- t2 t0))
					     (count)))
			   (targetns (/ "1000'000'000"
					30)))
		       (std--this_thread--sleep_for
			(std--chrono--nanoseconds (- targetns frameTimens)))
		      
		       ))
		    (window.swapBuffers)
		    (let ((t1 (high_resolution_clock--now)))
		      (let ((frameTimens (dot (duration_cast<nanoseconds> (- t1 t0))
					       (count)))
			    (frameTimems (/ frameTimens
					  1s6))
			    (frameRateHz (/ 1s9 frameTimens) ))
			,(lprint :vars `(frameTimems frameRateHz)))
		      (setf t0 t1)))))
	 )

       (return 0)))
   :omit-parens t
   :format t
   :tidy nil))
