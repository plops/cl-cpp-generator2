(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

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
	(members `((msg :type "const std::string&" :param t)
		   ;(wavetable :type "std::vector<double>" :param t)
		   ;(wavetable-size :type std--size_t :initform (wavetable.size))
		   ;(current-index :type double :initform 0d0)
		   ;(step :type double)
		   )))
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
					  (destructuring-bind (name &key type param (initform 0)) e
					    (let ((nname (intern
							  (string-upcase
							   (cl-change-case:snake-case (format nil "~a" name))))))
					      (when param
						nname))))))
		 (declare
		  ,@(remove-if #'null
			       (loop for e in members
				     collect
				     (destructuring-bind (name &key type param (initform 0)) e
				       (let ((nname (intern
						     (string-upcase
						      (cl-change-case:snake-case (format nil "~a" name))))))
					 (when param
					   
					   `(type ,type ,nname))))))
		  (construct
		   ,@(remove-if #'null
				(loop for e in members
				      collect
				      (destructuring-bind (name &key type param (initform 0)) e
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
				  (destructuring-bind (name &key type param (initform 0)) e
				    (let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					  (nname_ (format nil "~a_" (cl-change-case:snake-case (format nil "~a" name)))))
				      `(space ,type ,nname_)))))

	       ))))


  (let* ((name `ArgParser)
	(cli-args `((:name sampleRate :short r :default 10d6 :type double :help "Sample rate in Hz" :parse "std::stod")
		    (:name frequency :short f :default 433d6 :type double :help "Center frequency in Hz" :parse "std::stod")
		    (:name bufferSize :short b :default 512 :type int :help "Buffer Size (number of elements)" :parse "std::stoi")
		    (:name numberBuffers :short n :default 100 :type int :help "How many buffers to request" :parse "std::stoi"))
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
			  (handler "std::function<void(const std::string&)>"))
			)
     :implementation-preamble
     `(do0
       (include<> iostream
		  vector)
       )
     :code `(do0
	     (defclass ,name ()
	       "public:"
	       (defmethod ,name (,@(remove-if #'null
				    (loop for e in members
					  collect
					  (destructuring-bind (name &key type param (initform 0)) e
					    (let ((nname (intern
							  (string-upcase
							   (cl-change-case:snake-case (format nil "~a" name))))))
					      (when param
						nname))))))
		 (declare
		  ,@(remove-if #'null
			       (loop for e in members
				     collect
				     (destructuring-bind (name &key type param (initform 0)) e
				       (let ((nname (intern
						     (string-upcase
						      (cl-change-case:snake-case (format nil "~a" name))))))
					 (when param
					   
					   `(type ,type ,nname))))))
		  (construct
		   ,@(remove-if #'null
				(loop for e in members
				      collect
				      (destructuring-bind (name &key type param (initform 0)) e
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
		 (processArgs cmdline_args_)
		 )
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
		 
		 (let (
		       (options (std--vector<Option>
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
						     (throw (ArgException (string ,(format nil "Invalid value for --~a" name)))))))))))))))
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
				  (destructuring-bind (name &key type param (initform 0)) e
				    (let ((nname (cl-change-case:snake-case (format nil "~a" name)))
					  (nname_ (format nil "~a_" (cl-change-case:snake-case (format nil "~a" name)))))
				      `(space ,type ,nname_)))))))))
  
  (write-source 
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames "main.cpp"
		     *source-dir*))
   `(do0
     
     (include<>
      iostream
      string
      complex
      vector
      algorithm
      
      chrono

      filesystem
      unistd.h
      cstdlib
      
      SoapySDR/Device.hpp
					;SoapySDR/Types.hpp
      SoapySDR/Formats.hpp
      SoapySDR/Errors.hpp
      )
     (include
      immapp/immapp.h
      implot/implot.h
      imgui_md_wrapper.h
      )
     (include ArgException.h
	      ArgParser.h)
					;(include "cxxopts.hpp")

     (comments "./my_project -b $((2**10))")

     (defun Gui ()
       (ImGuiMd--RenderUnindented
	(string-r "Bla")))

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
		 (shm_files_exist (logand
				   ,@(loop for e in daemon-shm-files
					   collect
					   `(std--filesystem--exists (string ,e))))))
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
		    (parameters (argParser.getParsedArgs)))
		(startDaemonIfNotRunning)
		(let (
		      (runnerParams (HelloImGui--SimpleRunnerParams (designated-initializer
								     :guiFunction Gui
								     :windowTitle (string "imgui_soapysdr")
								     :windowSize (curly 800 600)
								     )))
		      (addOnsParams (ImmApp--AddOnsParams (designated-initializer
							   :withImplot true
							   :withMarkdown true
							   ))))
		  (ImmApp--Run runnerParams
			       addOnsParams))
		
		(let ((sdrResults (SoapySDR--Device--enumerate)))
		  (dotimes (i (sdrResults.size))
		    (declare (type "unsigned long" i))
		    ,(lprint :msg "found device"
			     :vars `(i)))
		  (let ((soapyDeviceArgs (aref sdrResults 0))
			(sdr (SoapySDR--Device--make soapyDeviceArgs)))
		    (declare (type "const auto&" soapyDeviceArgs))
		    (when (== nullptr sdr)
		      ,(lprint :msg "make failed")
		      (return -1))
		    (let ((direction SOAPY_SDR_RX)
			  (channel 0))
		      ,@(loop for e in `((:fun listAntennas :el antenna :values antennas)
					 (:fun listGains :el gain :values gains)
					 (:fun getFrequencyRange :el range :values ranges
					  :print ((range.minimum)
						  (range.maximum))))
			      collect
			      (destructuring-bind (&key fun print el values) e
				`(let ((,values (-> sdr (,fun direction channel))))
				   (for-range
				    (,el ,values)
				    ,(lprint :msg (format nil "~a" fun)
					     :vars `(,@(if print
							   `(,@print)
							   `(,el))
						     direction
						     channel
						     ))))))
		      ((lambda ()
			 (declare (capture "&"))
			 (let ((fullScale 0d0))
			   ,(lprint :vars `((-> sdr (getNativeStreamFormat direction channel fullScale))
					    fullScale))
			   )))
		      (-> sdr (setSampleRate direction channel parameters.sampleRate))
		      (-> sdr (setFrequency direction channel parameters.frequency))

		      ,(let ((acq-type "std::complex<float>") (acq-sdr-type 'SOAPY_SDR_CF32)
					;(acq-type "std::complex<short>") (acq-sdr-type 'SOAPY_SDR_CS16)
			     )
			 `(do0
			   (let ((rx_stream (-> sdr (setupStream direction
								 ,acq-sdr-type))))
			     (when (== nullptr rx_stream)
			       ,(lprint :msg "stream setup failed")
			       (SoapySDR--Device--unmake sdr)
			       (return -1))
			     ,(lprint :vars `((-> sdr
						  (getStreamMTU rx_stream))))
			     ((lambda ()
				(declare (capture "&"))
				(let ((flags 0)
				      (timeNs 0)
				      (numElems 0))
				  (-> sdr (activateStream rx_stream flags timeNs numElems))))))
			   (do0
			    (comments "reusable buffer of rx samples")
			    (let ((numElems parameters.bufferSize)
				  (numBytes (* parameters.bufferSize
					       (sizeof ,acq-type)))
				  (buf  
				    (,(format nil "std::vector<~a>" acq-type)
				     numElems)))
			      ,(lprint :msg (format nil "allocate ~a buffer" acq-sdr-type) :vars `(numElems numBytes))
			      (let ((start (std--chrono--high_resolution_clock--now))
				    (expected_ms0 (/ (* 1000d0 numElems)
						     parameters.sampleRate ))
				    (expAvgElapsed_ms expected_ms0)
				    (alpha .01))
				(comments "choose alpha in [0,1]. for small values old measurements have less impact on the average"
					  ".04 seems to average over 60 values in the history")
				(dotimes (i parameters.numberBuffers)		  
				  (let ((buffs (std--vector<void*> (curly (buf.data))))
					(flags 0)
					(time_ns 0LL)
			     
					(readStreamRet
					  (-> sdr
					      (readStream rx_stream
							  (buffs.data)
							  numElems
							  flags
							  time_ns
							  1e5)))
					(end (std--chrono--high_resolution_clock--now))
					(elapsed (std--chrono--duration<double> (- end start)))
					(elapsed_ms (* 1000 (elapsed.count)))
					(expected_ms (/ (* 1000d0 readStreamRet)
							parameters.sampleRate ))
			     
					)
				    (setf expAvgElapsed_ms (+ (* alpha elapsed_ms
								 )
							      (* (- 1d0 alpha)
								 expAvgElapsed_ms)))
				    ,(lprint :msg "data block acquisition took"
					     :vars `(i elapsed_ms expAvgElapsed_ms expected_ms ))
				    (setf start end)
				    (when (== readStreamRet SOAPY_SDR_TIMEOUT)
				      ,(lprint :msg "warning: timeout"))
				    (when (== readStreamRet SOAPY_SDR_OVERFLOW)
				      ,(lprint :msg "warning: overflow"))
				    (when (== readStreamRet SOAPY_SDR_UNDERFLOW)
				      ,(lprint :msg "warning: underflow"))
				    (when (< readStreamRet 0)
				      ,(lprint :msg "readStream failed"
					       :vars `(readStreamRet
						       (SoapySDR--errToStr readStreamRet)))
				      (do0 (-> sdr (deactivateStream rx_stream 0 0))
					   (-> sdr (closeStream rx_stream))
					   (SoapySDR--Device--unmake sdr))
				      (exit -1))
				    (when (!= readStreamRet numElems)
				      ,(lprint :msg "warning: readStream returned unexpected number of elements" :vars `(readStreamRet flags time_ns)))))))))
			 )
		      (do0 (-> sdr (deactivateStream rx_stream 0 0))
			   (-> sdr (closeStream rx_stream))
			   (SoapySDR--Device--unmake sdr)))))))
	  
	   ("const ArgException&" (e)
	     (do0 ,(lprint :msg "Error processing command line arguments"
			   :vars `((e.what)))
		  (return -1)))))
       
       
       (return 0)))
   :omit-parens t
   :format nil
   :tidy nil))

