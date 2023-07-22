(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (progn
    (defparameter *source-dir* #P"example/131_sdr/source01/")
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
	(members `((msg :type "std::string" :param t)
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
			  (values "const char*"))
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
      SoapySDR/Device.hpp
					;SoapySDR/Types.hpp
      SoapySDR/Formats.hpp
      )
     (include "ArgException.h")
					;(include "cxxopts.hpp")

     ,(let ((cli-args `((:name sampleRate :short r :default 10d6 :type double :help "Sample rate in Hz" :parse "std::stod")
			(:name frequency :short f :default 433d6 :type double :help "Center frequency in Hz" :parse "std::stod")
			(:name bufferSize :short b :default 512 :type int :help "Buffer Size (number of elements)" :parse "std::stoi")
			  
			)
		      ))
	`(do0
	  (defstruct0 Args
	      ,@(loop for e in cli-args
		      collect
		      (destructuring-bind (&key name short default type help parse) e
			`(,name ,type)))
	    )
	  (defun processArgs (args)
	    (declare (type "const std::vector<std::string>&" args)
		     
		     (values Args))
	    (let ((result (Args #+nil (curly ,@(loop for e in cli-args
					       collect
					       (destructuring-bind (&key name short default type help parse) e
						 default)))
				(designated-initializer
				 ,@(loop for e in cli-args
					 appending
					 (destructuring-bind (&key name short default type help parse) e
					   `(,name ,default)))))))
	      (for-range
	       (arg args)
	       ,@(loop for e in cli-args
		       collect
		       (destructuring-bind (&key name short default type help parse) e
			 `(if (== arg (string ,(format nil "--~a" name)))
			      (handler-case
				  (do0
				   (when (== &arg (&args.back))
				     (throw (ArgException (string ,(format nil "Expected value after --~a" name)))))
				   (setf (dot result ,name)
					 (,parse (aref (std--find &arg (args.end) arg) 1)))
				   )
				(const ("std::invalid_argument&")
				  (throw (ArgException (string ,(format nil "Invalid value for --~a" name)))))))))
	       (throw (ArgException (+ (string "Unknown argument: ")
				       arg)))
	       )
	      (return result))
       
	    #+nil (let ((options (cxxopts--Options (string "SDR Application")
						   (string "my_project"))))
		    ,@(loop for e in cli-args
			    collect
			    (destructuring-bind (&key name short default type help) e
			      `(let ((,name ,default))
				 (declare (type ,type ,name))
				 (dot options ("add_options()"
					       (string ,(format nil "~a,~a" short name))
					       (string ,help)
					       (,(format nil "cxxopts::value<~a>" type) ,name))
				      ))))
		    (let ((cmd_result (options.parse argc argv)))))
	    )))
     
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))
       #+nil (do0 "(void) argc;"
		  "(void) argv;")
       ;"std::vector<std::string> args(argv+1,argv+argc);"
       (let ((cmdlineargs (std--vector<std--string> (+ argv 1)
					     (+ argv argc))))
	(handler-case
	    (do0
	     (let ((parameters (processArgs cmdlineargs)))
	       (let ((results (SoapySDR--Device--enumerate)))
	 (dotimes (i (results.size))
	   (declare (type "unsigned long" i))
	   ,(lprint :msg "found device"
		    :vars `(i)))
	 (let ((sdrargs (aref results 0))
	       (sdr (SoapySDR--Device--make sdrargs)))
	   (declare (type "const auto&" sdrargs)
					;(type "const auto*" sdr)
		    )
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
	     (when true
	       (let ((fullScale 0d0))
		 ,(lprint :vars `((-> sdr (getNativeStreamFormat direction channel fullScale))
				  fullScale))
		 ))
	     (-> sdr (setSampleRate direction channel parameters.sampleRate))
	     (-> sdr (setFrequency direction channel parameters.frequency))
	     (do0
					;(comments "read complex floats")
	      (let ((rx_stream (-> sdr (setupStream direction
						    SOAPY_SDR_CF32
					; SOAPY_SDR_CS16
						    ))))
		(when (== nullptr rx_stream)
		  ,(lprint :msg "stream setup failed")
		  (SoapySDR--Device--unmake sdr)
		  (return -1))
		(when true
		  (let ((flags 0)
			(timeNs 0)
			(numElems 0))
		    (-> sdr (activateStream rx_stream flags timeNs numElems))))))
	     (do0
	      (comments "reusable buffer of rx samples")
	      (let ((numElems parameters.bufferSize)
		    (buf  ;(std--vector<std--complex<short>> numElems)
		      (std--vector<std--complex<float>> numElems)
					;("std::array<std::complex<float>,numElems>")
		      ))
		(declare (type "const auto" n))
		(dotimes (i 10)		  
		  (let ((buffs #+nil ("std::array<void*,1>" (curly (buf.data)
								   ))
			       (std--vector<void*> (curly (buf.data)
							  )))
			(flags 0)
			(time_ns 0LL)
			
			(ret (-> sdr
				 (readStream rx_stream
					     (buffs.data)
					     numElems
					     flags
					     time_ns
					     1e5))))
		    ,(lprint :vars `(ret flags time_ns))))))
	     (-> sdr (deactivateStream rx_stream 0 0)
		 )
	     (-> sdr (closeStream rx_stream))
	     (SoapySDR--Device--unmake sdr)
	     
	     )
	   ))))
	  
	  ("const ArgException&" (e)
	    (do0 ,(lprint :msg "Error processing command line arguments"
			  :vars `((e.what)))
		 (return -1)))))
       
       
       (return 0)))
   :omit-parens t
   :format nil
   :tidy nil))

