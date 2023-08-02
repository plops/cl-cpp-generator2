(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(setf *features* (set-difference *features* (list :more
						  :memoize-plan
						  :guru-plan
						  :dec-max
						  :dec-min
						  :dec-mean)))
(setf *features* (set-exclusive-or *features* (list :more
						    :dec-max
						    ;:dec-min
						    ;:dec-mean
						    ;:guru-plan
						    ;:memoize-plan
						    )))


(let* (;; (elem-type "float") (acq-sdr-type 'SOAPY_SDR_CF32)
       (elem-type "short") (acq-sdr-type 'SOAPY_SDR_CS16)
       (acq-type (format nil "std::complex<~a>" elem-type))
       (acq-vec-type (format nil "std::vector<std::complex<~a>>" elem-type))
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

  (defun benchmark (code)
    (let ((start "startBenchmark")
		(end "endBenchmark"))
      `(let ((,start (std--chrono--high_resolution_clock--now)))
	 ,code
	 (let ((,end (std--chrono--high_resolution_clock--now))
	       (elapsed (std--chrono--duration<double> (- ,end ,start)))
	       (elapsed_ms (* 1000 (elapsed.count)))
	       )
	   ,(lprint ; :msg (format nil "~a" (emit-c :code code :omit-redundant-parentheses t))
	     :vars `(elapsed_ms))))))

 (defun decimate-plots (args)
   (destructuring-bind (&key x ys idx (points `(dot (ImGui--GetContentRegionAvail)
						    x)))
       args
     `(progn
       (let ((pointsPerPixel (static_cast<int> (/ (dot ,x (size))
						  ,points )))))
       (if (<= pointsPerPixel 1)
	   (do0 ,@(loop for e in ys
			collect
			`(ImPlot--PlotLine (string ,e)
					   (dot ,x (data))
					   (dot ,e (data))
					   windowSize)))

	   (do0 (comments "Calculate upper bound for the number of pixels, preallocate memory for vectors, initialize counter.")
		
		(let ((pixels (/ (+ (dot ,x (size)) pointsPerPixel -1)
				 pointsPerPixel))
		      (x_downsampled (std--vector<double> pixels))
		      )
		  ,@(loop for y1 in ys
			  collect
			  `(do0
			    (let (#+dec-mean ( y_mean (std--vector<double> pixels))
				  #+dec-max (y_max (std--vector<double> pixels))
				  #+dec-min (y_min (std--vector<double> pixels))
				  (count 0))
			      (comments "Iterate over the data with steps of pointsPerPixel")
			      (for ((= "int i" 0)
				    (< i (dot ,x (size)))
				    (incf i pointsPerPixel))
				   (let (#+dec-max (max_val (aref ,y1 i))
					 #+dec-min (min_val (aref ,y1 i))
					 #+dec-mean (sum_val (aref ,y1 i)))
				     (comments "Iterate over the points under the same pixel")
				     (for ((= "int j" (+ i 1))
					   (logand (< j (+ i pointsPerPixel))
						   (< j (dot ,x (size))))
					   (incf j))
					   #+dec-max (setf max_val (std--max max_val (aref ,y1 j)))
					   #+dec-min (setf min_val (std--min min_val (aref ,y1 j)))
					  #+dec-mean (incf sum_val  (aref ,y1 j)))
				     #+dec-max (setf (aref y_max count) max_val)
				     #+dec-min (setf (aref y_min count) min_val)
				     (setf (aref x_downsampled count) (aref ,x i))
				     #+dec-mean
				     (setf (aref y_mean count) (/ sum_val pointsPerPixel))
				     (incf count))))

			    
			    ,@(loop for e in `(x_downsampled
					       #+dec-max y_max
					       #+dec-min y_min
					       #+dec-mean y_mean)
				    collect
				    `(dot ,e (resize count)))
			    ,@(loop for e in `(#+dec-max y_max
					       #+dec-min y_min
					       #+dec-mean y_mean)
				    collect
				    `(ImPlot--PlotLine (dot (paren (+ (std--string (string ,(format nil "~a_~a_" e y1)))
								      ,idx))
							    (c_str))
						       (x_downsampled.data)
						       (dot ,e (data))
						       (x_downsampled.size)))))))))))
  
  #+nil
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
					; cstring
		  cmath
		  iostream)
       )
     :code `(do0
	     (defclass ,name ()
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
		 (let ((sequence (std--vector<bool> n)))
		   (dotimes (i n)
		     (declare (type size_t i))
		     (setf (aref sequence i)  (step)))
		   (return sequence)))
	       
	       #+nil
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

	       ))))


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
	 (members `((g1 :type "std::array<char,11>" :initform-class (curly ,@(loop for i below 11 collect 0)))
		    (g2 :type "std::array<char,11>" :initform-class (curly ,@(loop for i below 11 collect 0)))
		    (tap :type "std::array<char*,2>")
		    (lut :type ,(format nil "static constexpr std::array<std::pair<int,int>,~a>" (length sat-def))
			 :initform-class
			 (curly ,@(loop for (e f) in sat-def
					collect
					`(std--make_pair ,e ,f))))
		    (prn :type "const int" :param t)
		    )))
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include<> vector
				   map
				   array
				   cstddef)
			)
     :implementation-preamble
     `(do0
       (include<> stdexcept
		  cstring
		  cmath
		  iostream
		  
		  )

       #+nil (include GpsCACodeGenerator.h)
       #+nil 
       (space "const std::map<int, std::pair<int,int>> GpsCACodeGenerator::lut_ = "
	      )
       )
     :code `(do0
	     (defclass ,name ()
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
		 (let (
		       (t0 (dot (aref lut_ (- prn_ 1)) first)
			   )
		       (t1 (dot (aref lut_ (- prn_ 1)) second)
			   ))
		   (declare (type "const auto" t0 t1))
		   (setf (aref tap_ 0) (+ (g2_.data) t0))
		   (setf (aref tap_ 1) (+ (g2_.data) t1)))
		 (std--memset (+ (g1_.data) 1) 1 10)
		 (std--memset (+ (g2_.data) 1) 1 10)
		 )
	       
	       (defmethod generate_sequence (n)
		 (declare (type size_t n)
			  (values "std::vector<bool>"))
		 (let ((sequence (std--vector<bool> n)))
		   (dotimes (i n)
		     (declare (type size_t i))
		     (setf (aref sequence i)  (step)))
		   (return sequence)))
	       
	       "private:"

	       (defmethod chip ()
		 (declare (values int))
		 (return (^ (aref g1_ 10)
			    (aref (aref tap_ 0) 0)
			    (aref (aref tap_ 1) 0)
			    )))

	       (defmethod clock ()
		 (setf (aref g1_ 0) (^ (aref g1_ 3)
				      (aref g1_ 10)))
		 (setf (aref g2_ 0) (^ (aref g2_ 2)
				      (aref g2_ 3)
				      (aref g2_ 6)
				      (aref g2_ 8)
				      (aref g2_ 9)
				      (aref g2_ 10)))
		 (do0
		  (std--memmove (+ (g1_.data) 1) (g1_.data) 10)
		  (std--memmove (+ (g2_.data) 1) (g2_.data) 10)))
	       
	       (defmethod step ()
		 (declare (values bool))
		 (let ((value (chip)))
		   (clock)
		   (return value)))
	       
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (name &key type param initform initform-class) e
				    (let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					  (nname_ (format nil "~a_" (cl-change-case:snake-case (format nil "~a" name)))))
				      (if initform-class
					  `(space ,type (setf ,nname_ ,initform-class))
					  `(space ,type ,nname_))))))

	       ))))

  (let* ((name `SdrManager)
	 (members `(
					;(parameters :type "const Args&" :param t)
		    (bufferSize :type "const int" :param t)
		    (fifoSize :type "const int" :param t)
		    (timeout_us :type "const int" :param t)
		    (capture_sleep_us :type "const int" :param t)
		    
		    (direction :type "const int" :initform-class SOAPY_SDR_RX)
		    (channel :type "const int" :initform-class 0)
		    (sdr :type "SoapySDR::Device*" :param nil :initform-class nullptr)
		    (buf :type ,(format nil "std::vector<~a>" acq-type) :initform  (,(format nil "std::vector<~a>" acq-type) buffer_size) :param nil )
		    (rx-stream :type "SoapySDR::Stream*" :initform-class nullptr :param nil)
		    #+more (average-elapsed-ms :type double :initform 0d0 :param nil)
		    #+more (alpha :type double :initform .08 :param nil)
		    (start :type "std::chrono::time_point<std::chrono::system_clock, std::chrono::duration<long,std::ratio<1,1000000000>>>"
			   :initform (std--chrono--high_resolution_clock--now)
			   :param nil)
		  
		    ;; members related to the capture thread
		    (capture-thread :type "std::thread" :initform nil :param nil)
		    (mtx :type "std::mutex" :initform nil :param nil)
		    (stop :type "bool" :initform-class false :param nil)
		    (cv :type "std::condition_variable" :initform nil :param nil)
		    (fifo :type ,fifo-type
			  :initform nil :param nil)
		    
		    )))
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include<> SoapySDR/Device.hpp
				   thread
				   mutex
				   deque
				   condition_variable
				   functional)
			
			)
     :implementation-preamble
     `(do0
       (include<> SoapySDR/Device.hpp
					;SoapySDR/Types.hpp
		  SoapySDR/Formats.hpp
		  SoapySDR/Errors.hpp
		  fstream
                  
		  )
       #+more
       (include<> chrono
		  iostream
		  iomanip))
     :code `(do0
	     (defclass ,name ()
	       
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
		  (explicit)
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
					     `(,nname_ ,initform))))))))
		  (explicit)	    
		  (values :constructor)))
	       
	       (defmethod initialize ()
		 (declare (values bool))
		 (let ((sdrResults (SoapySDR--Device--enumerate)))
		   #+more (dotimes (i (sdrResults.size))
			    (declare (type "unsigned long" i))
			    ,(lprint :msg "found device"
				     :vars `(i)))
		   (let ((soapyDeviceArgs (aref sdrResults 0)))
		     (declare (type "const auto&" soapyDeviceArgs))
		     (setf sdr_ (SoapySDR--Device--make soapyDeviceArgs))
		   
		     (when (== nullptr sdr_)
					;,(lprint :msg "soapysdr device make failed")
		       (throw (std--runtime_error (string "sdr device make failed")))
					;(return false)
		       )
		     (let () 
		       #+more
		       ,@(loop for e in `((:fun listAntennas :el antenna :values antennas)
					  (:fun listGains :el gain :values gains)
					  (:fun listFrequencies :el element :values elements)
					  (:fun getFrequencyRange :el range :values ranges
					   :print ((range.minimum)
						   (range.maximum)
						   (range.step)))
					  )
			       collect
			       (destructuring-bind (&key fun print el values) e
				 `(let ((,values (-> sdr_ (,fun direction_ channel_))))
				    (for-range
				     (,el ,values)
				     ,(lprint :msg (format nil "~a" fun)
					      :vars `(,@(if print
							    `(,@print)
							    `(,el))
						      direction_
						      channel_
						      ))))))

		       (do0
			,(lprint :msg "set highest gain")
			(-> sdr_ (setGainMode direction_ channel_ false))
			(-> sdr_ (setGain direction_ channel_ (string "IFGR") 20))
			(-> sdr_ (setGain direction_ channel_ (string "RFGR") 0)))
		       #+nil (let ((hasAutomaticGain (-> sdr_ (hasGainMode direction_ channel_))))
			       #+more (do0
				       ,(lprint :msg "has automatic gain control"
						:vars `(
							hasAutomaticGain))
				       ,(lprint :msg "balance" ;; none 
						:vars `((-> sdr_ (hasIQBalance direction_ channel_))
							(-> sdr_ (hasIQBalanceMode direction_ channel_))
							))
				       ,(lprint :msg "offset" 
						:vars `((-> sdr_ (hasDCOffset direction_ channel_))
							(-> sdr_ (hasDCOffsetMode direction_ channel_)) ;; supported
							)))
			       (when hasAutomaticGain
				 (let ((automatic false ;true
						  )
				       )
				   
				   #+more
				   (let ((ifgrGain (-> sdr_ (getGain direction_ channel_ (string "IFGR"))))
					 (ifgrGainRange (-> sdr_ (getGainRange direction_ channel_ (string "IFGR"))))
					 (rfgrGain (-> sdr_ (getGain direction_ channel_ (string "RFGR"))))
					 (rfgrGainRange (-> sdr_ (getGainRange direction_ channel_ (string "RFGR")))))
				     ,(lprint :msg "automatic gain"
					      :vars `((-> sdr_ (getGainMode direction_ channel_))
						      ifgrGain
						      (ifgrGainRange.minimum)
						      (ifgrGainRange.maximum)
						      (ifgrGainRange.step)
						      rfgrGain
						      (rfgrGainRange.minimum)
						      (rfgrGainRange.maximum)
						      (rfgrGainRange.step)
						      ))))))
		       #+more
		       ((lambda ()
			  (declare (capture "&"))
			  (let ((fullScale 0d0))
			    ,(lprint :vars `((-> sdr_ (getNativeStreamFormat direction_ channel_ fullScale))
					     fullScale)))))
		       
		       (<< std--cout std--fixed (std--setprecision 10) 1d0  std--endl)
		       
		       #+more ,(lprint :vars `((-> sdr_ (getSampleRate direction_ channel_))
					       (-> sdr_ (getBandwidth direction_ channel_))
					       (-> sdr_ (getFrequency direction_ channel_))
					       (-> sdr_ (getMasterClockRate)) ;; zero
					       (-> sdr_ (getReferenceClockRate)) ;; zero

					       ) )

		       #+more
		       (do0 (for-range (rate (-> sdr_ (listSampleRates direction_ channel_)))
				       ,(lprint :vars `(rate)))
			    (for-range (bw (-> sdr_ (listBandwidths direction_ channel_)))
				       ,(lprint :vars `(bw))
				       )
			    (for-range (clk (-> sdr_ (listClockSources))) ;; none
				       ,(lprint :vars `(clk)))
			    (for-range (time_src (-> sdr_ (listTimeSources))) ;; none
				       ,(lprint :vars `(time_src)))
			    (for-range (sensor (-> sdr_ (listSensors))) ;; none
				       ,(lprint :vars `(sensor)))
			    (for-range (reg (-> sdr_ (listRegisterInterfaces))) ;; none
				       ,(lprint :vars `(reg)))
			    (for-range (gpio (-> sdr_ (listGPIOBanks))) ;; none
				       ,(lprint :vars `(gpio)))
			    (for-range (uart (-> sdr_ (listUARTs))) ;; none
				       ,(lprint :vars `(uart))))

		       #+more
		       ,@(loop for e in `(RF CORR)
			       collect
			       (let ((name (format nil "frequency_range_~a" e)))
				 `(let ((,name (-> sdr_ (getFrequencyRange direction_ channel_ (string ,e)))))
				    (for-range (r ,name)
					       ,(lprint :msg (format nil "~a" name)
							:vars `((dot r (minimum))
								(dot r (maximum))) )))))

		       (do0
			(setf rx_stream_ (-> sdr_ (setupStream direction_
							       ,acq-sdr-type)))
			(when (== nullptr rx_stream_)
					;,(lprint :msg "stream setup failed")
			  (SoapySDR--Device--unmake sdr_)
			  (throw (std--runtime_error (string "sdr stream setup failed")))
					;(return false)
			  )
			,(lprint :vars `((-> sdr_
					     (getStreamMTU rx_stream_))))
			((lambda ()
			   (declare (capture "&"))
			   (let ((flags 0)
				 (timeNs 0)
				 (numElems 0))
			     (-> sdr_ (activateStream rx_stream_ flags timeNs numElems)))))
			(do0
			 (comments "reusable buffer of rx samples")
			 (let ((numElems buffer_size_)
			       #+more (numBytes (* buffer_size_
						   (sizeof ,acq-type))))
			   (setf buf_  
				 (,(format nil "std::vector<~a>" acq-type)
				  numElems))
			   ,(lprint :msg (format nil "allocate ~a buffer" acq-sdr-type) :vars `(numElems numBytes))
			   #+more (let ((expected_ms0 (/ (* 1000d0 numElems)
							 (get_sample_rate))))
				    (setf average_elapsed_ms_ expected_ms0))
			   )))
		       (return true)))))

	       ,@(loop for e in `((:name sample-rate :type double)
				  (:name bandwidth :type double)
				  (:name frequency :type double)
				  (:name gain-mode :type bool))
		       appending
		       (destructuring-bind (&key name type) e
			 (let ((setter (cl-change-case:snake-case (format nil "set-~a" name)))
			       (Setter (cl-change-case:camel-case (format nil "set-~a" name)))
			       (getter (cl-change-case:snake-case (format nil "get-~a" name)))
			       (Getter (cl-change-case:camel-case (format nil "get-~a" name))))
			   `(
			     (defmethod ,setter (v)
			       (declare (type ,type v))
			       ,(lprint :msg setter
					:vars `(v))
			       (-> sdr_ (,Setter direction_ channel_ v)))
			     (defmethod ,getter ()
			       (declare (values ,type))
			       (return (-> sdr_ (,Getter direction_ channel_))))))))

	       #+nil 
	       (do0 (-> sdr_ (setSampleRate direction channel parameters_.sampleRate))
		    (-> sdr_ (setBandwidth direction channel parameters_.bandwidth))
		    (-> sdr_ (setFrequency direction channel parameters_.frequency))
		    (-> sdr_ (setGainMode direction channel automatic)))

	       ,@(loop for e in `(IF RF)
		       collect
		       `(defmethod ,(format nil "setGain~a" e) (value)
			  (declare (type int value))
			  (-> sdr_ (setGain direction_ channel_ (string ,(format nil "~aGR" e)) value))))
	       
	       (defmethod getBuf ()
		 (declare (values ,(format nil "const std::vector<~a>&" acq-type))
			  (const))
		 (return buf_))

	       
	       (defmethod capture ()
		 (declare (values int))
		 (let (	;#+more (start (std--chrono--high_resolution_clock--now))
		       (numElems buffer_size_))
		   #+more (comments "choose alpha in [0,1]. for small values old measurements have less impact on the average"
				    ".04 seems to average over 60 values in the history")
		   (let ((buffs (std--vector<void*> (curly (buf_.data))))
			 (flags 0)
			 (time_ns 0LL)
			 
			 (readStreamRet (-> sdr_
					    (readStream rx_stream_
							(buffs.data)
							numElems
							flags
							time_ns
							timeout_us_)))
			 )
		     #+more (let ((end (std--chrono--high_resolution_clock--now))
				  (elapsed (std--chrono--duration<double> (- end start_)))
				  (elapsed_ms (* 1000 (elapsed.count)))
				  
				  (expected_ms (/ (* 1000d0 readStreamRet)
						  (get_sample_rate))))
			      (setf start_ end)
			      (setf average_elapsed_ms_
				    (+ (* alpha_ elapsed_ms)
				       (* (- 1d0 alpha_)
					  average_elapsed_ms_)))
			      (let ((dataBlockCount 0))
				(declare (type "static int" dataBlockCount))
				(incf dataBlockCount)
				(when (== 0 (% dataBlockCount 100))
				  ,(lprint :msg "data block acquired"
					   :vars `(dataBlockCount elapsed_ms average_elapsed_ms_ expected_ms )))))
		     (cond
		       ((space
			 #-more (setf "auto  readStreamRet"
				      (-> sdr_
					  (readStream rx_stream_
						      (buffs.data)
						      numElems
						      flags
						      time_ns
						      timeout_us_)))
			 
			 (== readStreamRet SOAPY_SDR_TIMEOUT))
			,(lprint :msg "warning: timeout")
			(return 0))
		       ((== readStreamRet SOAPY_SDR_OVERFLOW)
			,(lprint :msg "warning: overflow")
			(return 0))
		       ((== readStreamRet SOAPY_SDR_UNDERFLOW)
			,(lprint :msg "warning: underflow")
			(return 0))
		       ((< readStreamRet 0)
			,(lprint :msg "readStream failed"
				 :vars `(readStreamRet
					 (SoapySDR--errToStr readStreamRet)))
			(return 0))
		       ((!= readStreamRet numElems)
			,(lprint :msg "warning: readStream returned unexpected number of elements"
				 :vars `(readStreamRet flags time_ns))
			(return readStreamRet)))
		     (return numElems))))

	       (defmethod close ()
		 (do0 (-> sdr_ (deactivateStream rx_stream_ 0 0))
		      (-> sdr_ (closeStream rx_stream_))
		      (SoapySDR--Device--unmake sdr_)))

	       (defmethod startCapture ()
		 ,(lprint :msg "startCapture")
		 (setf capture_thread_ (std--thread &SdrManager--captureThread
						    this)))

	       (defmethod stopCapture ()
		 ,(lprint :msg "stopCapture")
		 (progn
		   (let ((lock (std--scoped_lock mtx_)))
		     (setf stop_ true)))
		 (dot cv_ (notify_all))
		 (when (capture_thread_.joinable)
		   (capture_thread_.join)))

	       (defmethod getFifo ()
		 (declare (values ,fifo-type))
		 (let ((lock (std--scoped_lock mtx_)))
		   (return fifo_)))

	       (defmethod processFifo (func &key (n (std--numeric_limits<std--size_t>--max)))
		 (declare (type ,(format nil "const std::function<void(const ~a&)> &" fifo-type) func)
			  (type "std::size_t" n))
		 #+nil ,(lprint :msg "processFifo"
				:vars `(n))
		 (let ((lock (std--scoped_lock mtx_))
		       (n0 (std--min n (fifo_.size))))
		   (comments "If n covers the entire fifo_, pass the whole fifo_ to func")
		   (if (<= (fifo_.size) n0)
		       (do0
			(func fifo_))
		       (do0
			(comments "If n is less than fifo_.size(), create a sub-deque with the last n elements and pass it to func")
			(comments "Get an iterator to the nth element from the end")
			(let ((start (- (fifo_.end)
					n0))
			      (lastElements (,fifo-type start (fifo_.end))))
			  (func lastElements))))))

	       (defmethod processFifoT (func &key (n (std--numeric_limits<std--size_t>--max)))
		 (declare (type "Func " func)
			  (type "std::size_t" n)
			  (values "template<typename Func> void"))
		 (let ((lock (std--scoped_lock mtx_))
		       (n0 (std--min n (fifo_.size))))
		   (comments "If n covers the entire fifo_, pass the whole fifo_ to func")
		   (if (<= (fifo_.size) n0)
		       (do0
			(func fifo_))
		       (do0
			(comments "If n is less than fifo_.size(), create a sub-deque with the last n elements and pass it to func")
			(comments "Get an iterator to the nth element from the end")
			(let ((start (- (fifo_.end)
					n0))
			      (lastElements (,fifo-type start (fifo_.end))))
			  (func lastElements))))))
	       
	       "private:"
	       (defmethod captureThread ()
		 
		 (let ((outputFile (std--ofstream (string "capturedData.bin")
						  (or std--ios--binary
						      std--ios--app))))
		   (let ((captureSleepUs capture_sleep_us_))
		     ,(lprint :msg "captureThread starting"
			      :vars `(captureSleepUs
				      fifo_size_
				      buffer_size_
				      timeout_us_))
		     (while true
					;,(lprint :msg "get lock")
			    (progn
			      (let ((lock (std--scoped_lock mtx_)))
			  
				(when stop_
				  ,(lprint :msg "stopping captureThread")
				  break))
			      (do0 
			       #+nil ,(lprint :msg "capture and push to fifo")
			       (when (space (setf "auto numElems" (capture))
					    (< 0 numElems))
				 (comments "Insert new elements into the deque")
				 (dot fifo_ (insert (fifo_.end)
						    (buf_.begin)
						    (+
						     (buf_.begin)
						     numElems)))
				 #+nil (do0
					(comments "Write data to file")
					(outputFile.write ("reinterpret_cast<const char*>" (buf_.data))
							  (* numElems (sizeof ,acq-type))))
				 ))

			      (when (< fifo_size_ (fifo_.size))
				(fifo_.erase (fifo_.begin)
					     (+ (fifo_.begin)
						(- (fifo_.size)
						   fifo_size_ )))))
			    (when (< 0 captureSleepUs)
			      (std--this_thread--sleep_for (std--chrono--microseconds captureSleepUs))))
		     (outputFile.close))))
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (name &key type param initform initform-class) e
				    (let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					  (nname_ (format nil "~a_" (cl-change-case:snake-case (format nil "~a" name)))))
				      (if initform-class
					  `(space ,type (= ,nname_ ,initform-class))
					  `(space ,type ,nname_))))))))))
  
  (let* ((name `MemoryMappedComplexShortFile)
	 (members `((file :type "boost::iostreams::mapped_file_source" :param nil)
		    (data :type "std::complex<short>*" :initform-class nullptr :param nil)
		    (filename :type "const std::string&" :param t)
		    (ready :type "bool" :initform-class false :param nil))))
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include<> iostream
				   fstream
				   vector
				   complex
				   boost/iostreams/device/mapped_file.hpp)
			)
     :implementation-preamble
     `(do0
       (include<> stdexcept
		  filesystem)
       )
     :code `(do0
	     (defclass ,name ()
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
		 ,(lprint :msg "try to mmap file"
			  :vars `(filename (std--filesystem--exists filename)))
		 (when (std--filesystem--exists filename)
		   (file_.open filename)
		   (when (file_.is_open)
		     (setf data_ (reinterpret_cast<std--complex<short>*> (const_cast<char*> (file_.data))))
		     (setf ready_ true))))

	       (defmethod "operator[]" (index)
		 (declare (type "std::size_t" index)
			  (values "std::complex<short>&")
			  (const))
		 (return (aref data_ index)))

	       (defmethod size ()
		 (declare (values "std::size_t")
			  (const))
		 (return (dot file_ (size))))
	       (defmethod ready ()
		 (declare (values bool)
			  (const)
			  )
		 (return ready_))
	       
	       (defmethod ~MemoryMappedComplexShortFile ()
		 (declare (values :constructor))
		 (file_.close))
	       
	       "private:"
	       
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (name &key type param initform initform-class) e
				    (let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					  (nname_ (format nil "~a_" (cl-change-case:snake-case (format nil "~a" name)))))
				      (if initform-class
					  `(space ,type (setf ,nname_ ,initform-class))
					  `(space ,type ,nname_)))))))))

    )

  (let* ((name `FFTWManager)
	 (members `(#+memoize-plan (plans :type "std::map<std::pair<int,int>,fftw_plan>" :param nil)
					;(window_size :type "int" :initform 512 :param t)
		    (number_threads :type "int" :initform-class 6 :param t)
		    )))
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include<> fftw3.h
				   map
				   vector
				   complex
				   )
			)
     :implementation-preamble
     `(do0
       (include<> stdexcept
		  fstream
		  #+more iostream)
       )
     :code `(do0
	     (defclass ,name ()
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
		 )

	       #+nil (defmethod print_plans ()
		       (declare (const))
		       (for-range (pair plans_)
				  (let ((windowSize pair.first.first)
					(nThreads pair.first.second)
					(planAddress pair.second))
				    ,(lprint :msg "print_plans"))))

	       

	       
	       (defmethod fftshift (in)
		 (declare
		  (const)
		  (type "const std::vector<std::complex<double>>&" in)
		  (values "std::vector<std::complex<double>>"))
		 (let ((mid (+ (in.begin)
			       (/ (in.size) 2)))
		       (out (std--vector<std--complex<double>> (in.size))))
		   (std--copy mid (in.end) (out.begin))
		   (std--copy (in.begin) mid (+ (out.begin)
						(std--distance mid (in.end))))
		   (return out)))

	       (defmethod fft (in windowSize)
		 (declare (type int windowSize)
			  (const)
			  (type "const std::vector<std::complex<double>>&" in)
			  (values "std::vector<std::complex<double>>"))
		 (when (!= windowSize (in.size))
		   (throw (std--invalid_argument (string "Input size must match window size."))))
		 (let ((out (std--vector<std--complex<double>> windowSize)))
		   (fftw_execute_dft (get_plan windowSize FFTW_FORWARD number_threads_)
				     (reinterpret_cast<fftw_complex*>
				      (const_cast<std--complex<double>*> (in.data)))
				     (reinterpret_cast<fftw_complex*>
				      (const_cast<std--complex<double>*> (out.data))))
		   (return (fftshift out))))

	       (defmethod ifft (in windowSize)
		 (declare (type int windowSize)
			  (const)
			  (type "const std::vector<std::complex<double>>&" in)
			  (values "std::vector<std::complex<double>>"))
		 (when (!= windowSize (in.size))
		   (throw (std--invalid_argument (string "Input size must match window size."))))
		 (let ((in2 (fftshift in))
		       (out (std--vector<std--complex<double>> windowSize)))
		   (fftw_execute_dft (get_plan windowSize FFTW_BACKWARD number_threads_)
				     (reinterpret_cast<fftw_complex*>
				      (const_cast<std--complex<double>*> (in2.data)))
				     (reinterpret_cast<fftw_complex*>
				      (const_cast<std--complex<double>*> (out.data))))
		   (return out)))

	       (defmethod ~FFTWManager ()
		 (declare (values :constructor))
		 #+memoize-plan
		 (for-range (kv plans_)
			    (fftw_destroy_plan kv.second)))
	       
	       "private:"
	       (defmethod get_plan (windowSize &key (direction FFTW_FORWARD) (nThreads 1) )
		 (declare (type int windowSize nThreads direction)
			  (values fftw_plan)
			  (const))
		 (when (<= windowSize 0)
		   (throw (std--invalid_argument (string "window size must be positive"))))

		 
		 #+nil (print_plans)
		 #+nil ,(lprint :msg "lookup plan"
				:vars `(windowSize nThreads))
		 #+memoize-plan (let ((iter (plans_.find (curly windowSize nThreads))))) 
		 (do0 
		  (if #-memoize-plan true #+memoize-plan (== (plans_.end) iter)
		      (do0 #+memoize-plan ,(lprint :msg "The plan hasn't been used before. Try to load wisdom or generate it."
						   :vars `(windowSize nThreads))

			   (do0 (let ((wisdom_filename (+ (string "wisdom_")
							  (std--to_string windowSize)
							  (string ".wis")))))
				(when (< 1 nThreads)
				  (setf wisdom_filename
					(+ (string "wisdom_")
					   (std--to_string windowSize)
					   (string "_threads")
					   (std--to_string nThreads)
					   (string ".wis")))))
			    
			   (let ((*in (fftw_alloc_complex windowSize))
				 (*out (fftw_alloc_complex windowSize)))
			     (when (logior !in
					   !out)
			       (do0 (fftw_free in)
				    (fftw_free out)
				    (throw (std--runtime_error (string "Failed to allocate memory for fftw plan")))))

			     (let ((wisdomFile (std--ifstream wisdom_filename)))
			       (if (wisdomFile.good)
				   (do0 #+nil ,(lprint :msg "read wisdom from existing file"
						       :vars `(wisdom_filename))
					(wisdomFile.close)
					(fftw_import_wisdom_from_filename (wisdom_filename.c_str)))
				   ,(lprint :msg "can't find wisdom file"
					    :vars `(wisdom_filename)))
			       (if (< 1 nThreads)
				   (do0 #+nil ,(lprint :msg "plan 1d fft with threads"
						       :vars `(nThreads))
					(fftw_plan_with_nthreads nThreads))
				   #+nil ,(lprint :msg "plan 1d fft without threads"
						  :vars `(nThreads)))
			       #-guru-plan (let ((p (fftw_plan_dft_1d windowSize
								      in out
								      direction ;FFTW_FORWARD
								      FFTW_MEASURE
								      ))))
			       #+guru-plan (let ((dim (fftw_iodim (designated-initializer :n windowSize
											  :is 1
											  :os 1)))
						 (p 
						   (fftw_plan_guru_dft 1 ;; rank
								       &dim ;; dims
								       0 ;; howmany_rank
								       nullptr ;; howmany_dims
								       in ;; in 
								       out ;; out
								       direction ;FFTW_FORWARD ;; sign
					;FFTW_MEASURE ;; flags
								       (or FFTW_MEASURE FFTW_UNALIGNED)
								       )))
					     )
			       (when !p
				 (do0 (fftw_free in)
				      (fftw_free out)
				      (throw (std--runtime_error (string "Failed to create fftw plan")))))
			       #+nil ,(lprint :msg "plan successfully created")
			       (unless (wisdomFile.good)
				 ,(lprint :msg "store wisdom to file"
					  :vars `(wisdom_filename))
				 (wisdomFile.close)
				 (fftw_export_wisdom_to_filename (wisdom_filename.c_str)))
			       )
			     (do0 (do0
					;,(lprint :msg "free in and out")
				   (fftw_free in)
				   (fftw_free out)
				   #-memoize-plan (return p)
				   )
				  #+memoize-plan(do0
						 ,(lprint :msg "store plan in class"
							  :vars `(windowSize nThreads))
						 (let ((insertResult (dot plans_
									  (insert (curly (curly windowSize nThreads) p))
									  )))
						   (setf iter (dot insertResult first))
						   ,(lprint :msg "inserted new key"
							    :vars `((plans_.size) insertResult.second))))
				  )
			     ))
		      #+memoize-plan (do0
				      ,(lprint :msg "plan has been used recently, reuse it.")))
		   
 		  #+memoize-plan (return iter->second)))
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (name &key type param initform initform-class) e
				    (let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					  (nname_ (format nil "~a_" (cl-change-case:snake-case (format nil "~a" name)))))
				      (if initform-class
					  `(space ,type (setf ,nname_ ,initform-class))
					  `(space ,type ,nname_)))))))))

    )
  
  
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     
     (include<>
      iostream
					;string
					;complex
      vector
					;algorithm
      
					;chrono

      filesystem
					;unistd.h
					;cstdlib

      cmath)
     (include
      
      implot.h
      imgui.h
      imgui_impl_glfw.h
      imgui_impl_opengl3.h
      GLFW/glfw3.h
      )
     (include
      GpsCACodeGenerator.h
      MemoryMappedComplexShortFile.h
      FFTWManager.h
      SdrManager.h)
	

     (defun glfw_error_callback (err desc)
       (declare (type int err)
		(type "const char*" desc))
       ,(lprint :msg "GLFW erro:"
		:vars `(err desc)))

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
	    ,(lprint :msg "verify that sdr daemon is running")
	    (unless (isDaemonRunning)
	      ,(lprint :msg "sdrplay daemon is not running. start it")
	      (let ((daemon_exit (system (string ,(format nil "~a &" daemon-fullpath))))))
	      ,(lprint :msg "return value"
		       :vars `(daemon_exit))
	      (sleep 1)))))
     
     (defun DrawPlot (file sdr fftw codes)
       (declare (type "const MemoryMappedComplexShortFile&" file)
		(type "const FFTWManager&" fftw)
		(type "const std::vector<std::vector<std::complex<double>>> &" codes)
		(type SdrManager& sdr))

       ,@(loop for e in `((:name sample-rate :type double)
			  (:name bandwidth :type double)
			  (:name frequency :type double)
			  (:name gain-mode :type bool))
	       collect
	       (destructuring-bind (&key name type) e
		 (let ((getter (cl-change-case:snake-case (format nil "get-~a" name))))
		   `(do0
		     (ImGui--Text (string ,(format nil "~a:" name)))
		     (ImGui--SameLine)
		     (ImGui--TextColored (ImVec4 1 1 0 1)
					 ,(case type
					    (`double `(string "%f"))
					    (`bool `(string "%s")))
					 ,(case type
					    (`double `(dot sdr (,getter)))
					    (`bool `(? (dot sdr (,getter))
						       (string "True")
						       (string "False"))))
					
					 )
		     ))))

       (let ((automaticGainMode true)
	     (old_automaticGainMode true))
	 (declare (type "static bool" automaticGainMode old_automaticGainMode))
	 (ImGui--Checkbox (string "Automatic Gain Mode")
			  &automaticGainMode)
	 (when (!= automaticGainMode
		   old_automaticGainMode)
	   (sdr.set_gain_mode automaticGainMode)
	   (setf old_automaticGainMode
		 automaticGainMode)))
       
       ,@(loop for e in `((:name IF :min 20 :max 59)
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
			(dot sdr (,(format nil "setGain~a" name) ,gain)))))))
       
       (let ((start 0)
	    
	     
	     (maxStart (static_cast<int> (/ (file.size)
					    (sizeof "std::complex<short>")))))
	 (declare (type "static int" start ))
	 (when (file.ready)
	  (ImGui--SliderInt (string "Start")
			    &start 0 maxStart)))


       ,(let* ((combo-name "windowSize")
	       (l-combo `(1024 5456 8192 10000 65536 80000 1048576))
	       (l-default-index 1)
	       (var-index (format nil "~aIndex" combo-name))
	       (old-var-index (format nil "old_~aIndex" combo-name))
	       (items-num (format nil "~aItemsNum" combo-name))
	       (items-str (format nil "~aItemsStr" combo-name))
	       (var-value (format nil "~a" combo-name)))
	  `(let ((,var-index ,l-default-index)
		 (,old-var-index ,l-default-index)
		 (,items-num (std--vector<double> (curly ,@l-combo)))
		 (,var-value (static_cast<int> (aref ,items-num ,var-index)))
		 (,items-str (std--vector<std--string> (curly ,@(mapcar #'(lambda (x) `(string ,x))
									l-combo)))))
	     (declare (type "static int" ,var-index ,old-var-index))
	     (when (ImGui--BeginCombo (string ,combo-name)
				      (dot (aref ,items-str ,var-index)
					   (c_str)))
	       (dotimes (i (dot ,items-str (size)))
		 (let ((is_selected (== ,var-index i)))
		   (when (ImGui--Selectable (dot (aref ,items-str i)
						 (c_str))
					    is_selected)
		     (setf ,var-index i
			   ,var-value (static_cast<int> (aref ,items-num i))))
		   (when is_selected
		     (ImGui--SetItemDefaultFocus))))
	       (ImGui--EndCombo))
	     (when (!= ,old-var-index
		       ,var-index)
					;(sdr.set_bandwidth (aref ,items-num ,var-index))
	       (setf ,old-var-index ,var-index))))

       ,(let* ((combo-name "bandwidth")
	       (l-combo `(200d3 300d3 600d3 1.536d6 5d6 6d6 7d6 8d6))
	       (l-default-index (- (length l-combo) 1))
	       (var-index (format nil "~aIndex" combo-name))
	       (old-var-index (format nil "old_~aIndex" combo-name))
	       (items-num (format nil "~aItemsNum" combo-name))
	       (items-str (format nil "~aItemsStr" combo-name))
	       (var-value (format nil "~aValue" combo-name)))
	  `(let ((,var-index ,l-default-index)
		 (,old-var-index ,l-default-index)
		 (,items-num (std--vector<double> (curly ,@l-combo)))
		 (,var-value (aref ,items-num ,var-index))
		 (,items-str (std--vector<std--string> (curly ,@(mapcar #'(lambda (x) `(string ,x))
									l-combo)))))
	     (declare (type "static int" ,var-index ,old-var-index))
	     (when (ImGui--BeginCombo (string ,combo-name)
				      (dot (aref ,items-str ,var-index)
					   (c_str)))
	       (dotimes (i (dot ,items-str (size)))
		 (let ((is_selected (== ,var-index i)))
		   (when (ImGui--Selectable (dot (aref ,items-str i)
						 (c_str))
					    is_selected)
		     (setf ,var-index i
			   ,var-value (aref ,items-num i)))
		   (when is_selected
		     (ImGui--SetItemDefaultFocus))))
	       (ImGui--EndCombo))
	     (when (!= ,old-var-index
		       ,var-index)
	       (sdr.set_bandwidth (aref ,items-num ,var-index))
	       (setf ,old-var-index ,var-index))))

       
       (when (logior (not (file.ready))
		     (logand (<= (+ start windowSize) maxStart)
			     (< 0 windowSize)))

	 (let ((x (std--vector<double> windowSize))
	       (y1 (std--vector<double> windowSize))
	       (y2 (std--vector<double> windowSize))
	       (zfifo (,acq-vec-type windowSize)))


	   (let ((realtimeDisplay true))
	     (declare (type "static bool" realtimeDisplay))
	     (when (file.ready)
	       (ImGui--Checkbox (string "Realtime display")
				&realtimeDisplay)))

	   (if realtimeDisplay
	       (do0
		(sdr.processFifo
		 (lambda (fifo)
		   (declare (type ,(format nil "const ~a &" fifo-type) fifo)
			    (capture "&"))
		   (let ((n windowSize))
		     (dotimes (i n)
		       (let ((z (aref fifo i)))
			 (setf (aref zfifo i) z)
			 (setf (aref x i) i
			       (aref y1 i) (z.real)
			       (aref y2 i) (z.imag))))))
		 windowSize))
	       (dotimes (i windowSize)
		 (let ((z (aref file (+ start i))))
		   (setf (aref x i) i	;(+ start i)
			 (aref y1 i) (z.real)
			 (aref y2 i) (z.imag)))))

	   (do0
	    ;(ImPlot--SetNextAxisLimits ImAxis_Y1 -30000 30000  ImPlotCond_Always)				
	    (when (ImPlot--BeginPlot (string "Waveform (I/Q)"))
	      
	      ,@(loop for e in `(y1 y2)
		      collect
		      `(ImPlot--PlotLine (string ,e)
					 (x.data)
					 (dot ,e (data))
					 windowSize))
	      (ImPlot--EndPlot)))

	   (let ((logScale false))
	     (declare (type "static bool" logScale))
	     (ImGui--Checkbox (string "Logarithmic Y-axis")
			      &logScale)
	     (handler-case
		 (let ((in (std--vector<std--complex<double>> windowSize))
		       (nyquist (/ windowSize 2d0))
		       (sampleRate (? realtimeDisplay
				      10d6
				      5.456d6))
		       (gps_freq 1575.42d6)
		       (lo_freq 4.092d6)
		       (centerFrequency (? realtimeDisplay
					   (sdr.get_frequency)
					   (- gps_freq lo_freq))))
		   (declare (type "static double" centerFrequency lo_freq))
		   (dotimes (i windowSize)
		     (setf (aref x i) (+ centerFrequency
					 (* sampleRate
					    (/ (static_cast<double> (- i (/ windowSize 2)))
					       windowSize))) ))

		   (let ((lo_phase 0d0)
			 
			 (lo_rate (* (/ lo_freq sampleRate) 4)))
		     (if realtimeDisplay
			 (do0
			  (dotimes (i windowSize)
			   (let ((zs (aref zfifo i))
				 (zr (static_cast<double> (zs.real)))
				 (zi (static_cast<double> (zs.imag)))
				 (z (std--complex<double> zr zi)))
			     (setf (aref in i) z))))
			 (do0
			  (dotimes (i windowSize)
			   (let ((zs (aref file (+ start i)))
				 (zr (static_cast<double> (zs.real)))
				 (zi (static_cast<double> (zs.imag)))
				 (z (std--complex<double> zr zi)))
			     (let ((lo_sin ("std::array<int,4>" (curly 1 1 0 0)))
				   (lo_cos ("std::array<int,4>" (curly 1 0 0 1))))
			       (declare (type "const auto " lo_sin lo_cos))
			       (let ((re (? (^ (zs.real)
					       (aref lo_sin (static_cast<int> lo_phase)))
					    -1 1))
				     (im (? (^ (zs.real)
					       (aref lo_cos (static_cast<int> lo_phase)))
					    -1 1)))
				 (setf (aref in i)
				       (std--complex<double>
					re im)))
			       (incf lo_phase lo_rate)
			       (when (<= 4 lo_phase)
				 (decf lo_phase 4)))
			     )))))
		   (let ((out (fftw.fft in windowSize)))
		     (if logScale
			 (dotimes (i windowSize)
			   (setf (aref y1 i) (* 10d0 (log10 (/ (std--abs (aref out i))
							       (std--sqrt windowSize))))))
			 (dotimes (i windowSize)
			   (setf (aref y1 i) (/ (std--abs (aref out i))
						(std--sqrt windowSize)))))
		     (do0
		      ;; (ImPlot--SetNextAxisLimits ImAxis_X1 (* -.5 sampleRate) (* .5 sampleRate))
		      #+nil 
		      (ImPlot--SetNextAxisLimits ImAxis_X2
						 (- centerFrequency (* .5 sampleRate))
						 (+ centerFrequency (* .5 sampleRate)))

		      (do0
		       (comments "If there are more points than pixels on the screen, then I want to combine all the points under one pixel into three curves: the maximum, the mean and the minimum.")

		       
		       
		       (when (ImPlot--BeginPlot (? logScale
						   (string "FFT magnitude (dB)")
						   (string "FFT magnitude (linear)")))
			 ,(decimate-plots `(:x x :ys (y1) :idx (string "")))
			 #+nil (do0
			  (let ((pointsPerPixel (static_cast<int> (/ (x.size)
								     (dot (ImGui--GetContentRegionAvail)
									  x))))))
			  (if (<= pointsPerPixel 1)
			      (do0 ,@(loop for e in `(y1)
					   collect
					   `(ImPlot--PlotLine (string ,e)
							      (x.data)
							      (dot ,e (data))
							      windowSize)))

			      (do0 (comments "Calculate upper bound for the number of pixels, preallocate memory for vectors, initialize counter.")
				   (let ((pixels (/ (+ (x.size) pointsPerPixel -1)
						    pointsPerPixel))
					 (x_downsampled (std--vector<double> pixels))
					 #+extra (y_max (std--vector<double> pixels))
					 (y_mean (std--vector<double> pixels))
					 #+extra (y_min (std--vector<double> pixels))
					 (count 0))
				     (comments "Iterate over the data with steps of pointsPerPixel")
				     (for ((= "int i" 0)
					   (< i (x.size))
					   (incf i pointsPerPixel))
					  (let (#+extra (max_val (aref y1 i))
						#+extra (min_val (aref y1 i))
						(sum_val (aref y1 i)))
					    (comments "Iterate over the points under the same pixel")
					    (for ((= "int j" (+ i 1))
						  (logand (< j (+ i pointsPerPixel))
							  (< j (x.size)))
						  (incf j))
						 #+extra (do0 (setf max_val (std--max max_val (aref y1 j)))
							      (setf min_val (std--min min_val (aref y1 j))))
						 (incf sum_val  (aref y1 j)))
					    #+extra (setf (aref y_max count) max_val
							  (aref y_min count) min_val)
					    (setf (aref x_downsampled count) (aref x i)
						  
						  (aref y_mean count) (/ sum_val pointsPerPixel))
					    (incf count)))
				     ,@(loop for e in `(x_downsampled #+extra y_max #+extra y_min y_mean)
					     collect
					     `(dot ,e (resize count)))
				     ,@(loop for e in `(#+extra y_max #+extra y_min y_mean)
					     collect
					     `(ImPlot--PlotLine (string ,e)
								(x_downsampled.data)
								(dot ,e (data))
								(x_downsampled.size)))))))

			 
			 (do0 (comments "handle user input. clicking into the graph allow tuning the sdr receiver to the specified frequency.")
			      (let ((xmouse (dot (ImPlot--GetPlotMousePos) x))))
			      (when (logand (ImPlot--IsPlotHovered)
					    (logior (ImGui--IsMouseClicked 2)
						    (ImGui--IsMouseDragging 2)))
				
				(if realtimeDisplay
				    (do0
				     (setf centerFrequency xmouse)
				     (sdr.set_frequency centerFrequency))
				    (do0
				     
				     (setf lo_freq
					   (- xmouse
					      centerFrequency))))))
			 (ImPlot--EndPlot)
			 (ImGui--Text (string "lo_freq: %6.10f MHz")
				      (* lo_freq 1d-6))
			 (ImGui--Text (string "xmouse: %6.10f GHz")
				      (* xmouse 1d-9))
			 (ImGui--Text (string "gps_freq-xmouse: %6.10f MHz")
				      (* (- gps_freq xmouse) 1d-6))
			 (ImGui--Text (string "centerFrequency-xmouse: %6.10f MHz")
				      (* (- centerFrequency xmouse) 1d-6))
			 (ImGui--Text (string "centerFrequency-gps_freq: %6.10f MHz")
				      (* (- centerFrequency gps_freq) 1d-6))))

		      (let ((codesSize (dot (aref codes 0) (size))))
			(if (== windowSize codesSize)
			    (when (ImPlot--BeginPlot (string "Cross-Correlations with PRN sequences"))
			      (let ((x_corr (std--vector<double> (out.size))))
				(dotimes (i (x_corr.size))
				  (setf (aref x_corr i) (* 1d0 i))))
			      (dotimes (code_idx 32)
				(let ((code (aref codes code_idx))
				      (len (out.size))
				      (prod (std--vector<std--complex<double>> len))
				      (dopStart (static_cast<int> (/ (* -5000 (static_cast<int> len))
								     sampleRate)))
				      (dopEnd (static_cast<int> (/ (* 5000 len)
								   sampleRate))))
				  (for ((= "int dop" dopStart)
					(<= dop dopEnd)
					(incf dop))
				       (do0
					(dotimes (i (out.size))
					  (let ((i1 (% (+ i -dop len) len)))
					   (setf (aref prod i)
						 (* (std--conj (aref out i))
						    (aref code i1)))))
					(let ((corr (fftw.ifft prod (prod.size)))
					      (corrAbs2 (std--vector<double> (out.size))))
					  (dotimes (i (out.size))
					    (let ((v (std--abs (aref corr i))))
					      (setf (aref corrAbs2 i) (* 10 (std--log10
									     (/ (* v v)
										windowSize)))))))
					,(decimate-plots `(:x x_corr :ys (corrAbs2)
							   :idx (std--to_string (+ 100
										   (* 100 code_idx)
										   dop))
							   :points 100))))
				  ))
			      
			      (ImPlot--EndPlot))
			    (ImGui--Text (string "Don't perform correlation windowSize=%d codesSize=%ld")
					 windowSize codesSize)
			    )))))
	       ("const std::exception&" (e)
		 (ImGui--Text (string "Error while processing FFT: %s")
			      (e.what))))))))
     
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))

       (let ((fftw (FFTWManager 6))))
       (glfwSetErrorCallback glfw_error_callback)
       (when (== 0 (glfwInit))
	 ,(lprint :msg "glfw init failed")
	 (return 1))
       
       #+more
       (do0		 ;let ((glsl_version (string "#version 130")))
	(glfwWindowHint GLFW_CONTEXT_VERSION_MAJOR 3)
	(glfwWindowHint GLFW_CONTEXT_VERSION_MINOR 0))
       (let ((*window (glfwCreateWindow 800 600
					(string "imgui_dsp")
					nullptr nullptr)))
	 (when (== nullptr window)
	   ,(lprint :msg "failed to create glfw window")
	   (return 1))
	 (glfwMakeContextCurrent window)
	 ,(lprint :msg "enable vsync")
	 (glfwSwapInterval 1)
	 (IMGUI_CHECKVERSION)
	 ,(lprint :msg "create imgui context")
	 (ImGui--CreateContext)
	 (ImPlot--CreateContext)

	 (let ((&io (ImGui--GetIO)))
	   (setf io.ConfigFlags (or io.ConfigFlags
				    ImGuiConfigFlags_NavEnableKeyboard)))
					;(ImGui--StyleColorsDark)
	 (ImGui_ImplGlfw_InitForOpenGL window true)
	 (ImGui_ImplOpenGL3_Init (string "#version 130"))
	 (glClearColor  0 0 0 1)
	 )

       (let ((sampleRate ;5456d3
			 10.0d6
			 ))
	 (do0
	  (comments "based on Andrew Holme's code http://www.jks.com/gps/SearchFFT.cpp")
	  (let (
		(caFrequency 1.023d6)
		(caStep (/ caFrequency
			   sampleRate))
		(corrWindowTime_ms 1d0  ;6.55361
				   )
		(corrLength (static_cast<int> (/ (* corrWindowTime_ms sampleRate)
						 1000)))
		(caSequenceLength 1023))
	    ,(lprint :msg "prepare CA code chips"
		     :vars `(corrLength))
	    
	    (let ((codes ("std::vector<std::vector<std::complex<double>>>")))
	      
	      (dotimes (i 32)
		(comments "chatGPT decided to start PRN index with 1. I don't like it but leave it for now.")
		(let ((ca (GpsCACodeGenerator (+ i 1)))
		      (chips (ca.generate_sequence caSequenceLength))
		      (code (std--vector<std--complex<double>> corrLength))
		      (caPhase 0d0)
		      (chipIndex 0))

		  #+nil
		  (do0(<< std--cerr (+ i 1) std--endl)
		      (dotimes (j (chips.size))
			(<< std--cerr
			    (? (aref chips j) -1 1)
			    (string " ")))
		      (<< std--cerr std--endl))
		  
		  (dotimes (i corrLength)
				    (setf (aref code i) (? (aref chips (% chipIndex
									  caSequenceLength))
							   1d0 -1d0))
				    (incf caPhase caStep)
				    (when (<= 1 caPhase)
				      (decf caPhase 1d0)
				      (incf chipIndex)))
		  ,(lprint :msg "compute FFT"
			   :vars `(i))
		  (progn
		     (when (== i 0)
		       (comments "the first fft takes always long (even if wisdom is present). as a work around i just perform a very short fft. then it takes only a few milliseconds. subsequent large ffts are much faster")
		       (let ((mini (std--vector<std--complex<double>> 32)))
			 ,(benchmark `(fftw.fft mini 32))))
		     ,(benchmark `(let ((out (fftw.fft code corrLength)))))
		     ,(lprint :msg "codes"
			      :vars `(i (codes.size) (out.size)))
		     (codes.push_back out)
		    )))))))

       (handler-case
	   (do0
	    (let ((sdr (SdrManager 64512
				   1100000 
				   50000
				   5000)))
	      
	      (startDaemonIfNotRunning)
	      ,(lprint :msg "initialize sdr manager")
	      (sdr.initialize)
	      
	      (do0
	       (sdr.set_frequency 1575.42d6)
	       (sdr.set_sample_rate sampleRate)
	       (sdr.set_bandwidth 8d6))
	      (sdr.startCapture))
	    (let ((fn #+nil (string "/mnt5/capturedData_L1_rate10MHz_bw5MHz_iq_short.bin")
		      (string "/mnt5/gps.samples.cs16.fs5456.if4092.dat"))
		  (file (MemoryMappedComplexShortFile
			 fn)))
	      (when (file.ready)
	       ,(lprint :msg "first element"
			:vars `(fn (dot (aref file 0) (real)))))
	      #+nil  (let ((z0 (aref file 0)))))
	    (while (!glfwWindowShouldClose window)
		   (glfwPollEvents)
		   (ImGui_ImplOpenGL3_NewFrame)
		   (ImGui_ImplGlfw_NewFrame)
		   (ImGui--NewFrame)
		   (DrawPlot file sdr fftw codes)
		   (ImGui--Render)
		   (let ((w 0)
			 (h 0))
		     (glfwGetFramebufferSize window &w &h)
		     (glViewport 0 0 w h)
		     (glClear GL_COLOR_BUFFER_BIT))
		   (ImGui_ImplOpenGL3_RenderDrawData (ImGui--GetDrawData))
		   (glfwSwapBuffers window))
	    (do0 (sdr.stopCapture)
		 (sdr.close)))
	 
	 ("const std::runtime_error&" (e)
	   ,(lprint :msg "error 1422:"
		    :vars `((e.what)))
	   (return -1))
	 ("const std::exception&" (e)
	   ,(lprint :msg "error 1426:"
		    :vars `((e.what)))
	   (return -1)))

       #+nil (do0
	      (ImGui_ImplOpenGL3_Shutdown)
	      (ImGui_ImplGlfw_Shutdown)
	      (ImPlot--DestroyContext)
	      (ImGui--DestroyContext)
	      (glfwDestroyWindow window)
	      (glfwTerminate))
       
       
       
       (return 0)))
   :omit-parens t
   :format nil
   :tidy nil))
