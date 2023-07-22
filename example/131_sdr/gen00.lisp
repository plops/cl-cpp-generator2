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
      vector
      map
      SoapySDR/Device.hpp
      SoapySDR/Types.hpp
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
	   ,(lprint :msg "found device"
		    :vars `(i)))
	 (let ((args (aref results 0))
	       (*sdr (SoapySDR--Device--make args)))
	   (when (== nullptr sdr)
	     ,(lprint :msg "make failed")
	     (return -1))
	   ,@(loop for e in `((:fun listAntennas)
			      (:fun listGains)
			      (:fun getFrequencyRange :print ((val.minimum)
							      (val.maximum))))
		   collect
		   (destructuring-bind (&key fun print) e
		    `(progn
		       (let ((vals ((-> sdr ,fun) SOAPY_SDR_RX 0)))
			 (for-range
			  (val vals)
			  ,(lprint :msg (format nil "~a" fun)
				   :vars (if print
					     `(,@print)
					     `(val))))))))
	   ))
       (return 0)))
   :omit-parens t
   :format nil
   :tidy nil))


