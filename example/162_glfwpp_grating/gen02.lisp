(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (setf *features* (set-difference *features* (list :more ;; command line parsing and print frame time statistics
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
      string
      vector
      array
					;numeric
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
			     (fitres (deque<float>)))))


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
				 ))))))

     #+more
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
	 (setf t0 t1)))
     
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       #+more
       ,(let ((l `((:name swapInterval :default 2 :short s)
		   (:name numberFramesForStatistics :default 211 :short F)
		   (:name darkLevel :default 0 :short D)
		   (:name brightLevel :default 255 :short B))))
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
						:default ,default :out ,(format nil "&~a" name)))))
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
				   (setf cmd (append cmd `(,default))))
				 (when out
				   (setf cmd (append cmd `(,out))))
				 `(dot op ,cmd)
				 ))))))
	     (op.parse argc argv)
	     (when (helpOption->is_set)
	       (<< cout
		   op
		   endl)
	       (exit 0))))
       ;;,(lprint :vars `((thread--hardware_concurrency)))
       #+more
       (let ((frameDelayEstimator (DelayEstimator numberFramesForStatistics))))    
       (let ((dark (/ darkLevel 255s0))
	     (bright (/ brightLevel 255s0))))
       ,(let* ((pattern-w 512)
	       (pattern-h 512)
	       (levels-w (floor (- (log pattern-w 2) 2)))
	       (levels-h (floor (- (log pattern-h 2) 2)))
	       (bright "bright")
	       (dark "dark")
	       (l0 `((:name all-white :draw ((:color (,bright ,bright ,bright) :type GL_QUADS :coords ((0 0 wf hf)))))
		     (:name all-dark :draw ((:color (,dark ,dark ,dark) :type GL_QUADS :coords ((0 0 wf hf)))))
		     ,@(loop for direction in `(vertical horizontal)
			    and pattern-size in `(,pattern-w ,pattern-h)
			    and levels-size in `(,levels-w ,levels-h)
			    appending
			    (loop for level below levels-size
				  appending
				  (let ((y (/ (* pattern-size) (expt 2 level))))
				    (loop for illum in `(normal invert)
					  and bg in `(,dark ,bright)
					  and fg in `(,bright ,dark)
					  collect
					  `(:name ,(format nil "~a-stripes-~a-~a" direction level illum)
					    :draw ((:color (,bg ,bg ,bg) :type GL_QUADS :coords ((0 0 wf hf)))
						   (:color (,fg ,fg ,fg) :type GL_QUADS
						    :coords 
						    ,(loop for i below (expt 2 level)
							    collect
							    (let ((o (* 2 i y)))
							      (if (eq direction 'vertical)
								  `(,o 0 ,(+ o y) hf)
								  `(0 ,o wf ,(+ o y))))))))))))))
	       (l (loop for e in l0 and e-i from 0 collect
			`(:id ,e-i ,@e))))
	  (defparameter *bla* l)
	  `(do0
	    (defclass+ DrawPrimitive ()
	      "public:"
	      "array<float,3> color;"
	      "decltype(GL_QUADS) type;"
	      "vector<array<float,4>> coords;")
	    (defclass+ DrawFrame ()
	      "public:"
	      "int id;"
	      "string name;"
	      "vector<DrawPrimitive> draw;"
	      (defun execute ()
		(for-range ((bracket color type coords) draw)
			   (glColor4f (aref color 0)
				      (aref color 1)
				      (aref color 2)
				      1s0)
			   (glBegin type)
			   (for-range ((bracket x0 y0 x1 y1) coords)
				      (glVertex2f x0 y0)
				      (glVertex2f x1 y0)
				      (glVertex2f x1 y1)
				      (glVertex2f x0 y1))
			   (glEnd))))
	    (let ((w ,pattern-w)
		  (h ,pattern-h)
		  (wf (static_cast<float> w))
		  (hf (static_cast<float> h))))
	    (comments "show a sequence of horizontal bars and vertical bars that split the image into 1/2, 1/4th, ... . each image is followed by its inverted version. the lcd of the projector is too slow to show this pattern exactly with 60Hz. that is why we set swap interval to 2 (we wait for two frames for every image so that the display has time to settle)")
	    (space
	     vector<DrawFrame>
	     (setf drawFrames
		   (curly
		    #+nil
		    (designated-initializer :id 0
					    :name (string "bright")
					    :draw (curly (designated-initializer :color (curly bright .2 .3)
										 :type GL_QUADS
										 :coords (curly (curly 0 0 (static_cast<float> w) 512)))))
		    		    
		    ,@(loop for e in l
			    collect
			    (destructuring-bind (&key id name draw) e
			      `(designated-initializer
				:id ,id
				:name (string ,name)
				:draw (curly
				       ,@(loop for d in draw
					       collect
					       (destructuring-bind (&key color type coords) d
						 `(designated-initializer
						   :color (curly ,@color)
						   :type ,type
						   :coords (curly ,@(loop for c in coords
									  collect
									  `(curly ,@(loop for c0 in c
											  collect
											  (if (numberp c0)
											      (coerce c0 'single-float)
											      c0)))))))))))))))))
       (let ((GLFW (glfw--init))
	     (hints (space glfw--WindowHints (designated-initializer
					      :clientApi glfw--ClientApi--OpenGl
					      :contextVersionMajor 2
					      :contextVersionMinor 0))))
	 (hints.apply)
	 (let ((idStripeWidth 16)
	       (idBits 9)
	       (wId (* idStripeWidth idBits))
	       ;(w 512)
	       (wAll (+ w wId))
	       ;(h 512)
	       (window (glfw--Window
			wAll h
			(string "GLFWPP Grating"))))
	   (glfw--makeContextCurrent window)
	   (comments "an alternative to increase swap interval is to change screen update rate `xrandr --output HDMI-A-0 --mode 1920x1080 --rate 24`")
	   (glfw--swapInterval swapInterval)
	   (let ((frameId 0)))
	   (while (!window.shouldClose)
		  (let ((time (glfw--getTime)))
		    
		    (glfw--pollEvents)		    
		    (do0
		     (if (< frameId (drawFrames.size))
			 (incf frameId)
			 (setf frameId 0))
		     		     
		     (glPushMatrix)
		     (comments "scale coordinates so that 0..w-1, 0..h-1 cover the screen")
		     (glTranslatef -1s0 -1s0 0s0)
		     (glScalef (/ 2s0 wAll) (/ 2s0 h) 1s0)

		     (dot (aref drawFrames frameId)
			  (execute))

		     #+nil
		     (do0 (glBegin GL_QUADS)
			  
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
			  (glEnd))

		     (do0
		      (comments "green on black barcode for the id on the right")
		      (do0
		       (glColor4f 0s0 0s0 0s0 1s0)
		       (glBegin GL_QUADS)
		       (glVertex2f w 0)
		       (glVertex2f wAll 0)
		       (glVertex2f wAll h)
		       (glVertex2f w h)
		       (glEnd ))
		      (do0
		       (glColor4f 0s0 1s0 0s0 1s0)
		       (glBegin GL_QUADS)
		       (dotimes (i 8)
			 (when (& frameId (<< 1 i))
			   (let ((x0 (+ w (* i idStripeWidth)))
				 (x1 (- (+ w (* (+ 1 i) idStripeWidth))
					(/ idStripeWidth 2))))
			     (do0
			      (glVertex2f x0 0)
			      (glVertex2f x1 0)
			      (glVertex2f x1 h)
			      (glVertex2f x0 h)))))
		       (glEnd )))
		     (glPopMatrix))
		    (window.swapBuffers)
		    #+more
		    (frameDelayEstimator.update)
		    ))))
       (return 0)))
   :omit-parens t
   :format t
   :tidy nil))
