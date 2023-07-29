(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(setf *features* (set-difference *features* (list :more)))
(setf *features* (set-exclusive-or *features* (list :more)))


(let* ( ;; (elem-type "float") (acq-sdr-type 'SOAPY_SDR_CF32)
      (elem-type "short") (acq-sdr-type 'SOAPY_SDR_CS16)
      (acq-type (format nil "std::complex<~a>" elem-type))
      (fifo-type (format nil "std::deque<~a>" acq-type)))
	
  (progn
    (defparameter *source-dir* #P"example/131_sdr/source03/src/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*)))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "sunday"))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")
  
  (let* ((name `GpsCACodeGenerator)
	(sat-def `((2 6)
		   (3 7)
		   (4 8)
		   (5 9)
		   (1 9)
		   (2 10)
		   (1 8)
		   (2 9)
		   (3 10)
		   (2 3)
		   (3 4)
		   (5 6)
		   (6 7)
		   (7 8)
		   (8 9)
		   (9 10)
		   (1 4)
		   (2 5)
		   (3 6)
		   (4 7)
		   (5 8)
		   (6 9)
		   (1 3)
		   (4 6)
		   (5 7)
		   (6 8)
		   (7 9)
		   (8 10)
		   (1 6)
		   (2 7)
		  (3 8)
		   (4 9)))
	(members `((registerSize :type "static constexpr size_t" :initform-class 10 :param nil)
		   (g1FeedbackBits :type "static constexpr std::array<int,2>"
				   :initform-class (curly 3 10) :param nil)
		   (g2FeedbackBits :type "static constexpr std::array<int,6>"
				   :initform-class (curly 2 3 6 8 9 10) :param nil)
		   (g2Shifts :type ,(format nil "static constexpr std::array<std::pair<int,int>,~a>" (length sat-def))
			     :initform-class
			     (curly ,@(loop for (e f)
					      in sat-def
					    collect
					    `(std--make_pair ,e ,f)) )
			     :param nil)
		   (g1 :type "std::deque<bool>")
		   (g2 :type "std::deque<bool>")
		   (prn :type int :param t)
		   )))
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include<> vector
				   array
				   deque
				   cstddef)
			)
     :implementation-preamble
     `(do0
       (include<> stdexcept
		  cstring
		  cmath
		  iostream)
       )
     :code `(do0
	     (defclass ,name "public std::exception"	 
	       "public:"
	       (defmethod ,name (,@(remove-if #'null
				    (loop for e in members
					  collect
					  (destructuring-bind (name &key type param initform initform-class) e
					    (let ((nname (intern
							  (string-upcase
							   (cl-change-case:snake-case (format nil "~a" name))))))
					      (when param
						nname))))))
		 (declare
		  ,@(remove-if #'null
			       (loop for e in members
				     collect
				     (destructuring-bind (name &key type param initform initform-class) e
				       (let ((nname (intern
						     (string-upcase
						      (cl-change-case:snake-case (format nil "~a" name))))))
					 (when param
					   
					   `(type ,type ,nname))))))
		  (construct
		   ,@(remove-if #'null
				(loop for e in members
				      collect
				      (destructuring-bind (name &key type param initform initform-class) e
					(let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					      (nname_ (format nil "~a_"
							      (cl-change-case:snake-case (format nil "~a" name)))))
					  (cond
					    (param
					     `(,nname_ ,nname))
					    (initform
					     `(,nname_ ,initform)))))))
		   )
		  (explicit)	    
		  (values :constructor))
		 (when (logior (< prn_ 1)
			       (< ,(length sat-def) prn_ ))
		   (throw (std--invalid_argument (+ (string "Invalid PRN: ")
						    (std--to_string prn_))))
		   )
		 (do0 (g1_.resize register_size_ true)
			(g2_.resize register_size_ true))
		 )
	       
	       (defmethod generate_sequence (n)
		 (declare (type size_t n)
			  (values "std::vector<bool>"))
		 (let ((sequence (std--vector<bool>)))
		   (sequence.reserve n)
		   (dotimes (i n)
		     (declare (type size_t i))
		     (sequence.push_back (step)))
		   (return sequence)))

	       (defmethod print_square (v)
		 (declare (type "const std::vector<bool> &" v))
		 (let ((size (v.size))
		       (side (static_cast<size_t> (std--sqrt size))))
		   (dotimes (i size)
		     (declare (type size_t i))
		     (let ((o (? (aref v i)
				 (string "\\u2588")
				 (string " "))))
		      (<< std--cout o)
		       (when (== 0 (% (+ i 1 ) side))
			 (<< std--cout (string "\\n")))))))
	       "private:"
	       (defmethod step ()
		 (declare (values bool))
		 (let ((new_g1_bit false))
		   (for-range (i g1_feedback_bits_)
			      (setf new_g1_bit (^ new_g1_bit (aref g1_ (- i 1))))))
		 (let ((new_g2_bit false))
		   (for-range (i g2_feedback_bits_)
			      (setf new_g2_bit (^ new_g2_bit (aref g2_ (- i 1))))))
		 (do0 (g1_.push_front new_g1_bit)
		      (g1_.pop_back))
		 (do0 (g2_.push_front new_g2_bit)
		      (g2_.pop_back))
		 (let ((delay1 (dot (aref g2_shifts_ (- prn_ 1))
				    first))
		       (delay2 (dot (aref g2_shifts_ (- prn_ 1))
				    second))))
		 (return (^ (g1_.back)
			    (^ (aref g2_ (- delay1 1))
			       (aref g2_ (- delay2 1))))))
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (name &key type param initform initform-class) e
				    (let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					  (nname_ (format nil "~a_" (cl-change-case:snake-case (format nil "~a" name)))))
				      (if initform-class
					  `(space ,type (setf ,nname_ ,initform-class))
					  `(space ,type ,nname_))))))

	       )

	     )))
  
  
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     
     (include<>
      iostream
      string
					;complex
      vector
					;algorithm
      
					;chrono

      filesystem
      unistd.h
      cstdlib

      cmath)
     (include
      immapp/immapp.h
      implot/implot.h
      imgui_md_wrapper.h
      )
     (include ArgException.h
	      ArgParser.h
	      SdrManager.h
	      GpsCACodeGenerator.h)
					;(include "cxxopts.hpp")

     (comments "./my_project -b $((2**10))")

     #+nil (defun DemoImplot ()
	     (let ((x)
		   (y1)
		   (y2))
	       (declare (type "static std::vector<double>" x y1 y2))
	       (when (x.empty)
		 (dotimes (i 1000)
		   (let ((x_ (* #.pi (/ 4d0 1000d0) i)))
		     (x.push_back x_)
		     (y1.push_back (cos x_))
		     (y2.push_back (sin x_)))))
	       (ImGuiMd--Render (string "# This is a plot"))
	       (when (ImPlot--BeginPlot (string "Plot"))
		 ,@(loop for e in `(y1 y2)
			 collect
			 `(ImPlot--PlotLine (string ,e)
					    (x.data)
					    (dot ,e (data))
					    (x.size)))
		 (ImPlot--EndPlot))))

     (defun DemoSdr (manager)
       (declare (type SdrManager& manager))

       (do0
	,@(loop for e in `((:name IF :min 0 :max 59)
			   (:name RF :min 0 :max 3))
		collect
		(destructuring-bind (&key name min max) e
		  (let ((gain (format nil "gain~a" name)))
		    `(let ((,gain ,min))
		       (declare (type "static int" ,gain))
		       (when (ImGui--SliderInt (string ,gain)
					       (ref ,gain) ,min ,max
					       (string "%02d")
					       ImGuiSliderFlags_AlwaysClamp)
			 (dot manager (,(format nil "setGain~a" name) ,gain))))))))

       ,@(loop for e in `((:name viewBlockSize :min 0 :max ,(expt 2 16))
			  (:name histogramSize :min 0 :max ,(expt 2 10))
			  )
	       collect
	       (destructuring-bind (&key name min max) e
		 (let ((var (format nil "~a" name)))
		   `(let ((,var (/ (+ ,max ,min) 2)))
		      (declare (type "static int" ,var))
		      (when (ImGui--SliderInt (string ,var)
					      (ref ,var) ,min ,max
					      (string "%06d")
					;ImGuiSliderFlags_AlwaysClamp
					      )
			(comments " just use value"))))))
       ,@(loop for e in `((:name histogramAlpha :min 0 :max 1 :default .04))
	       collect
	       (destructuring-bind (&key name min max default) e
		 (let ((var (format nil "~a" name)))
		   `(let ((,var ,default))
		      (declare (type "static float" ,var))
		      (when (ImGui--SliderFloat (string ,var)
						(ref ,var) ,min ,max)
			(comments " just use value"))))))
       
       (let ((x)
	     (y1)
	     (y2)
	     )
	 (declare (type "std::vector<double>" x y1 y2)))

       (let ((maxVal (,(format nil "std::numeric_limits<~a>::min" elem-type)))
	     (minVal (,(format nil "std::numeric_limits<~a>::max" elem-type))))
	 (declare (type ,elem-type maxVal minVal)))
       (do0
	(manager.processFifo
	 (lambda (fifo)
	   (declare (type ,(format nil "const ~a &" fifo-type) fifo)
		    (capture "&"))
	   ;;,(lprint :msg "processFifo_cb")
	   (let ((n (fifo.size)))
	     (dotimes (i n)
	       (x.push_back i)
	       (let ((re (dot  (aref fifo i) (real)) )
		     (im (dot  (aref fifo i) (imag)) )))
	       (y1.push_back re)
	       (y2.push_back im)
	       (setf maxVal (std--max maxVal (std--max re im)))
	       (setf minVal (std--min minVal (std--min re im))))))
	 viewBlockSize			;,(expt 2 16)
	 )

	
	
	(do0 (ImGuiMd--Render (string "# This is a plot"))
	     (when (ImPlot--BeginPlot (string "Plot"))
	       ,@(loop for e in `(y1 y2)
		       collect
		       `(ImPlot--PlotLine (string ,e)
					  (x.data)
					  (dot ,e (data))
					  (x.size)))
	       (ImPlot--EndPlot)))

	(do0
	 (comments "Apply exponential filter to stablilize histogram boundaries")
	 (let ((alpha histogramAlpha)
	       (filteredMax maxVal)
	       (filteredMin minVal))
	   (declare (type "static float" filteredMax filteredMin))
	   (setf filteredMax (+ (* (- 1 alpha) filteredMax)
				(* alpha maxVal)))
	   (setf filteredMin (+ (* (- 1 alpha) filteredMin)
				(* alpha minVal)))
	   
	   )
	 ,(lprint :msg "histogram"
		  :vars `(minVal maxVal)
		  )
	 (when (ImPlot--BeginPlot (string "Histogram"))
	   ,@(loop for e in `(y1 y2)
		   collect
		   `(ImPlot--PlotHistogram (string ,(format nil "histogram ~a" e))
				      
					   (dot ,e (data))
					   (dot ,e (size))
					   histogramSize
					   1.0
					   (ImPlotRange filteredMin filteredMax)
					   ))
	   (ImPlot--EndPlot))))
       #+ni 
       (let ((n (manager.capture)))
	 (when (< 0 n)
	   (let ((buf (manager.getBuf))
		 )
	     (let ((x)
		   (y1)
		   (y2))
	       (declare (type "std::vector<double>" x y1 y2))
	       (dotimes (i n)
		 (x.push_back i)
		 (y1.push_back (dot  (aref buf i) (real)) )
		 (y2.push_back (dot  (aref buf i) (imag)) ))
	       (ImGuiMd--Render (string "# This is a plot"))
	       (when (ImPlot--BeginPlot (string "Plot"))
		 ,@(loop for e in `(y1 y2)
			 collect
			 `(ImPlot--PlotLine (string ,e)
					    (x.data)
					    (dot ,e (data))
					    (x.size)))
		 (ImPlot--EndPlot)))
	     )))) 
    
     ,(let* ((daemon-name "sdrplay_apiService")
	     (daemon-path "/usr/bin/")
	     (daemon-fullpath (format nil "~a~a" daemon-path daemon-name))
	     (daemon-shm-files `("/dev/shm/Glbl\\\\sdrSrvRespSema"
				 "/dev/shm/Glbl\\\\sdrSrvCmdSema"
				 "/dev/shm/Glbl\\\\sdrSrvComMtx"
				 "/dev/shm/Glbl\\\\sdrSrvComShMem")))
	`(do0
	  (defun isDaemonRunning ()
	    (declare (values bool))
	    (let ((exit_code (system (string ,(format nil "pidof ~a > /dev/null" daemon-name))))
		  (shm_files_exist true ))
	      ,@(loop for e in daemon-shm-files
		      collect
		      `(unless (std--filesystem--exists (string ,e))
			 ,(lprint :msg (format nil "file ~a does not exist" e)
				  )
			 (return false)))
	      (return (logand (== 0 (WEXITSTATUS exit_code))
			      shm_files_exist))))
	  (defun startDaemonIfNotRunning ()
	    (unless (isDaemonRunning)
	      ,(lprint :msg "sdrplay daemon is not running. start it")
	      (system (string ,(format nil "~a &" daemon-fullpath)))
	      (sleep 1)))))
     
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))

       (let ((ca (GpsCACodeGenerator 4)))
	 ,(lprint :msg "CA")
	 (ca.print_square (ca.generate_sequence 1023)))
       
       #+nil (let ((cmdlineArgs (std--vector<std--string> (+ argv 1)
							  (+ argv argc))))
	       (handler-case
		   (do0
		    (let ((argParser (ArgParser cmdlineArgs))
			  (parameters (argParser.getParsedArgs))
			  (manager (SdrManager parameters)))
		      (startDaemonIfNotRunning)
		      (manager.initialize)
		      (manager.startCapture)
		
		      (let ((runnerParams (HelloImGui--SimpleRunnerParams
					   (designated-initializer
					    :guiFunction
					    (paren
					     (lambda ()
					       (declare (capture "&manager"))
					; ,(lprint :msg "GUI")
					       (ImGuiMd--RenderUnindented
						(string-r "# Bundle"))
					;(DemoImplot)
					       (DemoSdr manager)))
					    :windowTitle (string "imgui_soapysdr")
					    :windowSize (curly 800 600)
					    )))
			    (addOnsParams (ImmApp--AddOnsParams (designated-initializer
								 :withImplot true
								 :withMarkdown true
								 ))))
			(ImmApp--Run runnerParams
				     addOnsParams)
			(manager.stopCapture)
			(manager.close))))
	   
		 ("const ArgException&" (e)
		   (do0 ,(lprint :msg "Error processing command line arguments"
				 :vars `((e.what)))
			(return -1)))))
       
       
       (return 0)))
   :omit-parens t
   :format nil
   :tidy nil)

  )
