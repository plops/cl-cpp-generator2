(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(setf *features* (set-difference *features* (list :more)))
(setf *features* (set-exclusive-or *features* (list :more)))


(progn
  (progn
    (defparameter *source-dir* #P"example/131_sdr/source02/src/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*)))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")

  (let ((name `ArgException)
	(members `((msg :type "const std::string&" :param t))))
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include<> exception
				   string)
			)
     :implementation-preamble
     `(do0
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
		 
		 )
	       (defmethod what ()
		 (declare (const)
			  (noexcept)
			  (override)
			  (values "[[nodiscard]] const char*"))
		 (return (msg_.c_str)))
	       "private:"
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (name &key type param initform initform-class) e
				    (let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					  (nname_ (format nil "~a_" (cl-change-case:snake-case (format nil "~a" name)))))
				      `(space ,type ,nname_)))))

	       )

	     ,(let ((name 'SdrException))
		`(defclass ,name "public std::exception"	 
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
						 `(,nname_ ,initform))
						)))))
		       )
		      (explicit)	    
		      (values :constructor))
		 
		     )
		   (defmethod what ()
		     (declare (const)
			      (noexcept)
			      (override)
			      (values "[[nodiscard]] const char*"))
		     (return (msg_.c_str)))
		   "private:"
		   ,@(remove-if #'null
				(loop for e in members
				      collect
				      (destructuring-bind (name &key type param initform initform-class) e
					(let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					      (nname_ (format nil "~a_" (cl-change-case:snake-case (format nil "~a" name)))))
					  (if initform-class
					      `(space ,type (setf ,nname_ ,initform-class))
					      `(space ,type ,nname_))))))

		   )))))

  (let* ((name `SdrManager)
	 (acq-type "std::complex<float>") (acq-sdr-type 'SOAPY_SDR_CF32)
	 ;;(acq-type "std::complex<short>") (acq-sdr-type 'SOAPY_SDR_CS16)
	
	 (members `(
		    (parameters :type "const Args&" :param t)
		    (sdr :type "SoapySDR::Device*" :param nil :initform-class nullptr)
		    (buf :type ,(format nil "std::vector<~a>" acq-type) :initform  (,(format nil "std::vector<~a>" acq-type) parameters_.bufferSize) :param nil )
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
		    (fifo :type "std::deque<std::complex<float>>" :initform nil :param nil)
		    
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
				   condition_variable)
			(include ArgParser.h)
			)
     :implementation-preamble
     `(do0
       (include<> SoapySDR/Device.hpp
					;SoapySDR/Types.hpp
		  SoapySDR/Formats.hpp
		  SoapySDR/Errors.hpp

		  )
       #+more
       (include<> chrono
		  iostream))
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
					     `(,nname_ ,initform)))))))
		   )
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
		       ,(lprint :msg "make failed")
		       (return false))
		     (let ((direction SOAPY_SDR_RX)
			   (channel 0))
		       #+more
		       ,@(loop for e in `((:fun listAntennas :el antenna :values antennas)
					  (:fun listGains :el gain :values gains)
					  (:fun listFrequencies :el element :values elements)
					  (:fun getFrequencyRange :el range :values ranges
					   :print ((range.minimum)
						   (range.maximum)))
					  )
			       collect
			       (destructuring-bind (&key fun print el values) e
				 `(let ((,values (-> sdr_ (,fun direction channel))))
				    (for-range
				     (,el ,values)
				     ,(lprint :msg (format nil "~a" fun)
					      :vars `(,@(if print
							    `(,@print)
							    `(,el))
						      direction
						      channel
						      ))))))
		       (let ((hasAutomaticGain (-> sdr_ (hasGainMode direction channel))))
			 #+more (do0
			  ,(lprint :msg "has automatic gain control"
				   :vars `(
					   hasAutomaticGain))
			  ,(lprint :msg "balance" ;; none 
				   :vars `((-> sdr_ (hasIQBalance direction channel))
					   (-> sdr_ (hasIQBalanceMode direction channel))
					   ))
			  ,(lprint :msg "offset" 
				   :vars `((-> sdr_ (hasDCOffset direction channel))
					   (-> sdr_ (hasDCOffsetMode direction channel)) ;; supported
					   )))
			 (when hasAutomaticGain
			   (let ((automatic false ;true
					    )
				 )
			     (-> sdr_ (setGainMode direction channel automatic))
			     (-> sdr_ (setGain direction channel (string "IFGR") 59))
			     (-> sdr_ (setGain direction channel (string "RFGR") 3))
			     #+more
			     (let ((ifgrGain (-> sdr_ (getGain direction channel (string "IFGR"))))
				   (ifgrGainRange (-> sdr_ (getGainRange direction channel (string "IFGR"))))
				   (rfgrGain (-> sdr_ (getGain direction channel (string "RFGR"))))
				   (rfgrGainRange (-> sdr_ (getGainRange direction channel (string "RFGR")))))
			       ,(lprint :msg "automatic gain"
					:vars `((-> sdr_ (getGainMode direction channel))
						ifgrGain
						(ifgrGainRange.minimum)
						(ifgrGainRange.maximum)
						rfgrGain
						(rfgrGainRange.minimum)
						(rfgrGainRange.maximum)
						))))))
		       #+more
		       ((lambda ()
			  (declare (capture "&"))
			  (let ((fullScale 0d0))
			    ,(lprint :vars `((-> sdr_ (getNativeStreamFormat direction channel fullScale))
					     fullScale)))))
		       
		       
		       (-> sdr_ (setSampleRate direction channel parameters_.sampleRate))
		       (-> sdr_ (setBandwidth direction channel parameters_.bandwidth))
		       (-> sdr_ (setFrequency direction channel parameters_.frequency))
		       ,(lprint :vars `((-> sdr_ (getSampleRate direction channel))
					(-> sdr_ (getBandwidth direction channel))
					(-> sdr_ (getFrequency direction channel))
					(-> sdr_ (getMasterClockRate)) ;; zero
					(-> sdr_ (getReferenceClockRate)) ;; zero

					) )

		       #+more
		       (do0 (for-range (rate (-> sdr_ (listSampleRates direction channel)))
				       ,(lprint :vars `(rate)))
			    (for-range (bw (-> sdr_ (listBandwidths direction channel)))
				       ,(lprint :vars `(bw)))
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
				 `(let ((,name (-> sdr_ (getFrequencyRange direction channel (string ,e)))))
				    (for-range (r ,name)
					       ,(lprint :msg (format nil "~a" name)
							:vars `((dot r (minimum))
								(dot r (maximum))) )))))

		       (do0
			(setf rx_stream_ (-> sdr_ (setupStream direction
							       ,acq-sdr-type)))
			(when (== nullptr rx_stream_)
			  ,(lprint :msg "stream setup failed")
			  (SoapySDR--Device--unmake sdr_)
			  (return false))
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
			 (let ((numElems parameters_.bufferSize)
			       #+more (numBytes (* parameters_.bufferSize
					    (sizeof ,acq-type))))
			   (setf buf_  
				 (,(format nil "std::vector<~a>" acq-type)
				  numElems))
			   ,(lprint :msg (format nil "allocate ~a buffer" acq-sdr-type) :vars `(numElems numBytes))
			   #+more (let ((expected_ms0 (/ (* 1000d0 numElems)
						  parameters_.sampleRate )))
			     (setf average_elapsed_ms_ expected_ms0))
			   )))
		       (return true)))))

	       (defmethod getBuf ()
		 (declare (values ,(format nil "const std::vector<~a>&" acq-type))
			  (const))
		 (return buf_))

	       
	       (defmethod capture ()
		 (declare (values int))
		 (let (;#+more (start (std--chrono--high_resolution_clock--now))
		       (numElems parameters_.bufferSize))
		   #+more (comments "choose alpha in [0,1]. for small values old measurements have less impact on the average"
				    ".04 seems to average over 60 values in the history")
		   (let ((buffs (std--vector<void*> (curly (buf_.data))))
			 (flags 0)
			 (time_ns 0LL)
			 (timeout_us
			   parameters_.timeoutUs
					;10000L
			   ;100000L
			   )
			 (readStreamRet (-> sdr_
				   (readStream rx_stream_
					       (buffs.data)
					       numElems
					       flags
					       time_ns
					       timeout_us)))
			 )
		     #+more (let ((end (std--chrono--high_resolution_clock--now))
				  (elapsed (std--chrono--duration<double> (- end start_)))
				  (elapsed_ms (* 1000 (elapsed.count)))
				  
				  (expected_ms (/ (* 1000d0 readStreamRet)
						  parameters_.sampleRate)))
			      (setf start_ end)
			      (setf average_elapsed_ms_
				    (+ (* alpha_ elapsed_ms)
				       (* (- 1d0 alpha_)
					  average_elapsed_ms_)))
			      ,(lprint :msg "data block acquired"
				       :vars `(elapsed_ms average_elapsed_ms_ expected_ms )))
		     (cond
		       ((space
			 #-more (setf "auto  readStreamRet"
			       (-> sdr_
				   (readStream rx_stream_
					       (buffs.data)
					       numElems
					       flags
					       time_ns
					       timeout_us)))
			 
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
		 (declare (values "std::deque<std::complex<float>>"))
		 (let ((lock (std--scoped_lock mtx_)))
		   (return fifo_)))

	       (defmethod processFifo (func &key (n (std--numeric_limits<std--size_t>--max)))
		 (declare (type "const std::function<void(const std::deque<std::complex<float>>&)> &" func)
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
			      (lastElements (std--deque<std--complex<float>> start (fifo_.end))))
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
			      (lastElements (std--deque<std--complex<float>> start (fifo_.end))))
			  (func lastElements))))))
	       
	       "private:"

	       (defmethod captureThread ()
		 ,(lprint :msg "captureThread")
		 (let ((captureSleepUs parameters_.captureSleepUs))
		  (while true
					;,(lprint :msg "get lock")
			 (progn
			   (let ((lock (std--scoped_lock mtx_)))
			  
			     (when stop_
			       ,(lprint :msg "stopping captureThread")
			       break))
			   (do0 (comments "capture and push to buffer")
					;,(lprint :msg "capture")
				(when (space (setf "auto numElems" (capture))
					     (< 0 numElems))
				  (comments "Insert new elements into the deque")
				  (dot fifo_ (insert (fifo_.end)
						     (buf_.begin)
						     (+
						      (buf_.begin)
						      numElems)))
				  ))

			   #+nil
			   ,(lprint :msg "Remove old elements if size exceeds <fifoSize>"
				    :vars `(parameters_.fifoSize (fifo_.size)))
			   #+nil (while (< parameters_.fifoSize (fifo_.size))
					(fifo_.pop_front))
			   (when (< parameters_.fifoSize (fifo_.size))
			     (fifo_.erase (fifo_.begin)
					  (+ (fifo_.begin)
					     (- (fifo_.size)
						parameters_.fifoSize )))))
			(when (< 0 captureSleepUs)
			 (std--this_thread--sleep_for (std--chrono--microseconds captureSleepUs))))))
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (name &key type param initform initform-class) e
				    (let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					  (nname_ (format nil "~a_" (cl-change-case:snake-case (format nil "~a" name)))))
				      (if initform-class
					  `(space ,type (= ,nname_ ,initform-class))
					  `(space ,type ,nname_))))))))))

  (let* ((name `ArgParser)
	 (cli-args `((:name sampleRate :short r :default 10d6 :type double :help "Sample rate in Hz" :parse "std::stod")
		     (:name bandwidth :short B :default  8d6 :type double :help "Bandwidth in Hz" :parse "std::stod")
		     (:name frequency :short f :default 433d6 :type double :help "Center frequency in Hz" :parse "std::stod")
		     (:name bufferSize :short b :default 512 :type int :help "Buffer Size (number of elements)" :parse "std::stoi")
		     (:name fifoSize :short F :default 2048 :type int :help "Fifo Buffer Size (number of elements)" :parse "std::stoi")
		     (:name timeoutUs :short T :default 20000 :type int :help "Timeout to read buffer of b samples (microseconds)" :parse "std::stoi")
		     (:name captureSleepUs :short t :default 5000 :type int :help "Delay to wait before next read attempt of b samples (microseconds)" :parse "std::stoi")
		     
					;(:name numberBuffers :short n :default 100 :type int :help "How many buffers to request" :parse "std::stoi")
		     )
		   )
	 (members `((cmdlineArgs :type "const std::vector<std::string>&" :param t)
		    (parsedArgs :type "Args" :param nil
				:initform (Args (designated-initializer
						 ,@(loop for e in cli-args
							 appending
							 (destructuring-bind (&key name short default type help parse) e
							   `(,name ,default)))))))))
    (write-class
     :dir (asdf:system-relative-pathname
	   'cl-cpp-generator2
	   *source-dir*)
     :name name
     :headers `()
     :header-preamble `(do0
			(include<> string
				   functional)
			(include "ArgException.h")
			(defstruct0 Args
			    ,@(loop for e in cli-args
				    collect
				    (destructuring-bind (&key name short default type help parse) e
				      `(,name ,type))))

			(defstruct0 Option
			    (longOpt "std::string")
			  (shortOpt "std::string")
			  (description "std::string")
			  (defaultValue "std::string")
			  (handler "std::function<void(const std::string&)>")))
     :implementation-preamble
     `(do0
       (include<> iostream
		  vector))
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
		 (processArgs cmdline_args_))
	       (defmethod getParsedArgs ()
		 (declare (values "[[nodiscard]] Args")
			  (const))
		 (return parsed_args_))
	       
	       "private:"
	       (defmethod printHelp (options)
		 (declare (type "const std::vector<Option>&" options)
					;(const)
			  (static))
		 (<< std--cout
		     (string "Usage: ./imgui_soapysdr [OPTIONS]")
		     std--endl)
		 (for-range (opt options)
			    (<< std--cout
				(string " ")
				opt.longOpt
				(string " or ")
				opt.shortOpt
				(string ": ")
				opt.description
				(string " default: ")
				opt.defaultValue
				std--endl)))
	       
	       (defmethod processArgs (args)
		 (declare (type "const std::vector<std::string>&" args))
		 (let ((options (std--vector<Option>
				 (curly
				  ,@(loop for e in cli-args
					  collect
					  (destructuring-bind (&key name short default type help parse) e
					    `(designated-initializer
					      :longOpt (string ,(format nil "--~a" name))
					      :shortOpt (string ,(format nil "-~a" short))
					      :description (string ,help)
					      :defaultValue (std--to_string (dot parsed_args_ ,name))
					      :handler
					      (paren
					       (lambda (x)
						 (declare (type "const std::string&" x)
							  (capture "this"))
						 (handler-case
						     (setf (dot this->parsed_args_ ,name)
							   (,parse x))
						   (const ("std::invalid_argument&")
						     (throw (ArgException
							     (string ,(format nil "Invalid value for --~a" name)))))))))))))))
		   (let ((it (args.begin)))
		     (while (!= it (args.end))
			    (if (logior (== *it (string "--help"))
					(== *it (string "-h")))
				(do0
				 (printHelp options)
				 (exit 0))
				(do0
				 (comments "Find matching option")
				 (let ((optIt (std--find_if (options.begin)
							    (options.end)
							    (lambda (opt)
							      (declare (type "const Option&" opt)
								       (capture "&it"))
							      (return (logior (== *it
										  opt.longOpt)
									      (== *it
										  opt.shortOpt)))))))
				   (when (== optIt (options.end))
				     (throw (ArgException (+ (string "Unknown argument: ")
							     *it))))
				   (comments "Move to next item, which should be the value for the option")
				   (incf it)
				   (when (== it
					     (args.end))
				     (throw (ArgException (+ (string "Expected value after ")
							     *it))))
				   (optIt->handler *it)
				   (comments "Move to next item, which should be the next option")
				   (incf it))))))))
	       	       
	       ,@(remove-if #'null
			    (loop for e in members
				  collect
				  (destructuring-bind (name &key type param initform initform-class) e
				    (let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					  (nname_ (format nil "~a_" (cl-change-case:snake-case (format nil "~a" name)))))
			              (if initform-class
					  `(space ,type (setf ,nname_ ,initform-class))
					  `(space ,type ,nname_))))))))))
  
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
	      SdrManager.h)
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
       (let ((x)
	     (y1)
	     (y2)
	     )
	 (declare (type "std::vector<double>" x y1 y2)))
       (do0
	(manager.processFifo
	 (lambda (fifo)
	   (declare (type "const std::deque<std::complex<float>>&" fifo)
		    (capture "&"))
	   ;;,(lprint :msg "processFifo_cb")
	   (let ((n (fifo.size)))
	     (dotimes (i n)
	       (x.push_back i)
	       (y1.push_back (dot  (aref fifo i) (real)) )
	       (y2.push_back (dot  (aref fifo i) (imag)) ))))
	 ,(expt 2 16))
	(do0 (ImGuiMd--Render (string "# This is a plot"))
	     (when (ImPlot--BeginPlot (string "Plot"))
	       ,@(loop for e in `(y1 y2)
		       collect
		       `(ImPlot--PlotLine (string ,e)
					  (x.data)
					  (dot ,e (data))
					  (x.size)))
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
       
       (let ((cmdlineArgs (std--vector<std--string> (+ argv 1)
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
   :tidy nil))

