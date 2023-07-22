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


  (defun share (name)
    (format nil "std::shared_ptr<~a>" name))
  (defun uniq (name)
    (format nil "std::unique_ptr<~a>" name))
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
      
      SoapySDR/Device.hpp
      ;SoapySDR/Types.hpp
      SoapySDR/Formats.hpp
      )
     (include "cxxopts.hpp")
     
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))
       #+nil (do0 "(void) argc;"
	    "(void) argv;")

       ,(let ((cli-args `((:name sampleRate :short r :default 10e6 :type float :help "Sample rate in Hz")
			  (:name frequency :short f :default 433e6 :type float :help "Center frequency in Hz")
			  (:name bufferSize :short b :default 512 :type int :help "Buffer Size (number of elements)")
			  
			  )
			))
	 `(do0
       
	   (let ((options (cxxopts--Options (string "SDR Application")
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
	     (let ((cmd_result (options.parse argc argv)))))))
       
       (let ((results (SoapySDR--Device--enumerate)))
	 (dotimes (i (results.size))
	   (declare (type "unsigned long" i))
	   ,(lprint :msg "found device"
		    :vars `(i)))
	 (let ((args (aref results 0))
	       (sdr (SoapySDR--Device--make args)))
	   (declare (type "const auto&" args)
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
		       `(let ((,values ((-> sdr ,fun) direction channel)))
			  (for-range
			   (,el ,values)
			   ,(lprint :msg (format nil "~a" fun)
				    :vars `(,@(if print
						  `(,@print)
						  `(,el))
					    direction
					    channel
					    ))))))
	     (progn
	       (let ((fullScale 0d0))
		 ,(lprint :vars `((-> sdr (getNativeStreamFormat direction channel fullScale))
				  fullScale))
		 ))
	     (-> sdr (setSampleRate direction channel 10e6))
	     (-> sdr (setFrequency direction channel 433e6))
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
		(progn
		  (let ((flags 0)
			(timeNs 0)
			(numElems 0))
		    (-> sdr (activateStream rx_stream flags timeNs numElems))))))
	     (do0
	      (comments "reusable buffer of rx samples")
	      (let ((n 512)
		    (buf	 ;(std--vector<std--complex<short>> n)
		      (std--vector<std--complex<float>> n)
					;("std::array<std::complex<float>,n>")
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
					     n
					     flags
					     time_ns
					     1e5))))
		    ,(lprint :vars `(ret flags time_ns))))))
	     (-> sdr (deactivateStream rx_stream 0 0)
		 )
	     (-> sdr (closeStream rx_stream))
	     (SoapySDR--Device--unmake sdr)
	     
	     )
	   ))
       (return 0)))
   :omit-parens t
   :format nil
   :tidy nil))


