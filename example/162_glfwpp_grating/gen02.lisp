(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more ;; command line parsing
						    :invert ;; show inverted frames
						    :barcode ;; show frame id as a vertical barcode
						    )))
  (setf *features* (set-exclusive-or *features* (list :more
						      :invert
						      :barcode
						      ))))

(let ()
  (defparameter *source-dir* #P"example/162_glfwpp_grating/source02/src/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)

  (defun begin (arr)
    `(ref (aref ,arr 0)) )
  (defun end (arr)
    `(+ (ref (aref ,arr 0)) (dot ,arr (size))))
  
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
     #+more (include "/home/martin/src/popl/include/popl.hpp")
     (include<>
      iostream
      format
      cmath
      deque
      valarray

					;GL/glew.h
      
      glfwpp/glfwpp.h
      
					;Eigen/Core 
				
      thread
      chrono
      algorithm
      numeric
	
      )
     "using namespace std;"
     "using namespace chrono;"
					;"using namespace Eigen;"
     "using Scalar = float;"
     "using Vec = std::vector<Scalar>;"
     "using VecI = const Vec;"

     #+more
     (defclass+ Statistics ()
       "public:"
       (defmethod getSignificantDigits (num)
	 (declare (type Scalar num)
		  (values int))
	 (when (== num 0s0)
	   (return 1))
	 (when (< num 0)
	   (setf num -num))
	 (let ((significantDigits 0))
	   (while (<= num 1s0)
		  (*= num 10s0)
		  (incf significantDigits))
	   (return significantDigits)))

       (defmethod printStat (m_md_d_dd)
	 (declare			
	  (type "tuple<Scalar,Scalar,Scalar,Scalar>" m_md_d_dd)
	  (values "string"))
	 (let (((bracket m md d dd) m_md_d_dd)))
	 (letc ((rel (* 100s0 (/ d m)))
		(mprecision  (getSignificantDigits md))
		(dprecision  (getSignificantDigits dd))
		(rprecision  (getSignificantDigits rel))

		(fmtm (+ (std--string (string "{:."))
			 (to_string mprecision)
			 (string "f}")))
		(fmtd (+ (std--string (string "{:."))
			 (to_string dprecision)
			 (string "f}")))
		(fmtr (+ (std--string (string " ({:."))
			 (to_string rprecision)
			 (string "f}%)")))
		(format_str (+  fmtm (string "±") fmtd fmtr)))
	       (return (vformat format_str (make_format_args m d rel)))
	       ))

       (defmethod Statistics (n)
	 (declare (type int n)
		  (values :constructor)
		  (construct (numberFramesForStatistics n)
			     (fitres (deque<float>))))
	 )


       "deque<float> fitres;"
       "int numberFramesForStatistics;"

       (defmethod push_back (frameTimems)
			  (declare (type float frameTimems))
			  (fitres.push_back frameTimems)
			  (when (< numberFramesForStatistics (fitres.size))
			    (fitres.pop_front)))
       
       (defmethod compute ()
	 (declare (values "tuple<Scalar,Scalar,Scalar,Scalar>"))
	 (let ((computeStat
		 (lambda (fitres
			  filter) 
		   (declare (type "const auto&" fitres)
			    (capture "")
			    (values "tuple<Scalar,Scalar,Scalar,Scalar>"))
		   (comments "compute mean and standard deviation Numerical Recipes 14.1.2 and 14.1.8")
		   (do0
		    (let ((data (valarray<Scalar> (fitres.size)))))
		    (data.resize (fitres.size))

		    
		    (transform (fitres.begin)
			       (fitres.end)
			       ,(begin 'data)
			       filter))
		   
		   (letc ((N (static_cast<Scalar> (data.size)))
			  (mean (/ (dot data (sum))
				   N))
			  
			  ;; 14.1.8 corrected two-pass algorithm from bevington 2002


			  ;; (comments "pass1 = sum( (data-s) ** 2 )")
			  ;; (comments "pas2 = sum( data-s )")
			  (stdev
			   (sqrt (/ (- (dot (pow (- data mean) 2)
					    (sum))
				       (/ (pow
					   (dot (paren (- data mean))
						(sum))
					   2)
					  N)
				       )
				    (- N 1s0))))
			  ;; error in the mean due to sampling
			  (mean_stdev (/ stdev (sqrt N)))
			  ;; error in the standard deviation due to sampling
			  (stdev_stdev (/ stdev (sqrt (* 2 N)))))
			 
			 (return (make_tuple mean mean_stdev stdev stdev_stdev)))))))
	(return (computeStat fitres
		       (lambda (f)
			 (declare (type "const auto&" f))
			 (return f    ;(,(format nil "get<~a>" e-i) f)
				 ))))
	 ))
     
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
      
       #+more ,(let ((l `(
			  (:name swapInterval :default 2 :short s)
			  (:name numberFramesForStatistics :default 211 :short F)
			  (:name darkLevel :default 0 :short D)
			  (:name brightLevel :default 255 :short B)
			  )))
		 `(let ((op (popl--OptionParser (string "allowed options")))
			,@(loop for e in l collect
					   (destructuring-bind (&key name default short (type 'int)) e
					     `(,name (,type ,default))))
			,@(loop for e in `((:long help :short h :type Switch :msg "produce help message")
					   (:long verbose :short v :type Switch :msg "produce verbose output")
					  
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

       

       (defclass+ DelayEstimator ()
	 "public:"
	 (defmethod DelayEstimator (numberFramesForStatistics)
	   (declare (type int numberFramesForStatistics)
		    (construct (numberFramesForStatistics numberFramesForStatistics)
			       (frameRateStats (Statistics numberFramesForStatistics)))
		    (values :constructor))
	   (setf t0 (high_resolution_clock--now)))
	 "int numberFramesForStatistics;"
	 "Statistics frameRateStats;"
	 
	 (letd ((t0 (high_resolution_clock--now))
		 (t1
		  (high_resolution_clock--now))
		 ))
	 
	 (defmethod update ()
	   (setf t1 (high_resolution_clock--now))
	   (let ((frameTimens (dot (duration_cast<nanoseconds> (- t1 t0))
				   (count)))
		 (frameTimems (/ frameTimens
				 1s6))
		 (frameRateHz (/ 1s9 frameTimens) ))
	     (frameRateStats.push_back frameTimems)
		       
	     (letc ((cs (frameRateStats.compute) )
		    (pcs (frameRateStats.printStat cs))))
	     (let (
		   ((bracket frameTime_ frameTime_Std frameTimeStd frameTimeStdStd)
		     cs
		    )
		   ))
	     ,(lprint :vars `(pcs frameTimems frameRateHz)))
	   (setf t0 t1))
	 )
       
       (let ((frameDelayEstimator (DelayEstimator numberFramesForStatistics))))
       
       

       (let ((dark (/ darkLevel 255s0))
	     (bright (/ brightLevel 255s0))))
       
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
	   (glfw--swapInterval swapInterval)
	   #+nil (when (!= GLEW_OK
			   (glewInit))
		   (throw (std--runtime_error (string "Could not initialize GLEW"))))
	   
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
		      (when (== current_level (* 8 #+invert 2 #-invert 1))
			(setf horizontal !horizontal
			      current_level 0))
		      #+invert (let ((white (== 0 (% current_level 2)))))
		      )

		     
		     
		     (#-invert do0 #+invert if #+invert white
		      (do0 (glClearColor dark dark dark 1s0)
			   (glClear GL_COLOR_BUFFER_BIT)
			   #+invert (glColor4f bright bright bright 1s0)
			   )
		      #+invert (do0 (glClearColor bright bright bright 1s0)
				    (glClear GL_COLOR_BUFFER_BIT)
				    (glColor4f dark dark dark 1s0)
				    ))
		     
		     (glPushMatrix)
		     (comments "scale coordinates so that 0..w-1, 0..h-1 cover the screen")
		     (glTranslatef -1s0 -1s0 0s0)
		     (glScalef (/ 2s0 w) (/ 2s0 h) 1s0)
		     (glBegin GL_QUADS)

		     (let ((level (/ current_level #-invert 1 #+invert 2
						   ))))
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
		    
		    
		    (window.swapBuffers)
		    (frameDelayEstimator.update)
		    )))
	 )

       (return 0)))
   :omit-parens t
   :format t
   :tidy nil))
