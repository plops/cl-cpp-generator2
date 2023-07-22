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
     
     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))
       "(void) argc;"
       "(void) argv;"
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
	     (-> sdr (setSampleRate direction channel 10e6))
	     (-> sdr (setFrequency direction channel 433e6))
	     (do0
	      (comments "read complex floats")
	      (let ((rx_stream (-> sdr (setupStream direction SOAPY_SDR_CF32))))
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
		    (buf (std--vector<std--complex<float>> n) ;("std::array<std::complex<float>,n>")
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


