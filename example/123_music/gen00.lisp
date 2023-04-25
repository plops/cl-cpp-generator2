(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)


(progn
   (progn
    (defparameter *source-dir* #P"example/123_music/source00/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*)))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")

  (let ((name `WavetableOscillator)
	(members `((sample-rate :type double :param t)
		   (wavetable :type "std::vector<double>" :param t)
		   (wavetable-size :type std--size_t :initform (wavetable.size))
		   (current-index :type double :initform 0d0)
		   (step :type double))))
      (write-class
       :dir (asdf:system-relative-pathname
	     'cl-cpp-generator2
	     *source-dir*)
       :name name
       :headers `()
       :header-preamble `(do0
			  (include<> vector
				     cstdint)
			  )
       :implementation-preamble
       `(do0
	 #+nil (space "extern \"C\" "
		(progn
		  ))
	 (do0
	  "#define FMT_HEADER_ONLY"
	  (include "core.h"))
	 (include<>
	  iostream
	  vector
	  cmath
	  cstdint
	  stdexcept))
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
		   (when (wavetable.empty)
		     (throw (std--invalid_argument ,(sprint :msg "Wavetable cannot be empty."))))
		   )

		 (defmethod set_frequency (frequency)
		   (declare (type double frequency))
		   (setf step_ (/ (* frequency wavetable_size_)
				  sample_rate_)))

		 (defmethod next_sample ()
		   (declare (values double))
		   (let ((index_1 (static_cast<std--size_t> current_index_))
			 (index_2 (% (+ index_1
					1)
				     wavetable_size_))
			 (fraction (- current_index_
				      index_1))
			 (sample (+ (* (aref wavetable_ index_1)
				       (- 1d0 fraction))
				    (* (aref wavetable_ index_2)
				       fraction))))
		     (incf current_index_ step_)
		     (when (< wavetable_size_ current_index_)
		       (decf current_index_ wavetable_size_))
		     (return sample)))
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
    (merge-pathnames #P"main.cpp"
		     *source-dir*))
   `(do0
     
     (include<> vector
		)
     (include "WavetableOscillator.h")

     (do0
      "#define FMT_HEADER_ONLY"
      (include "core.h"))

     

     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))
       ,(lprint :msg (multiple-value-bind
			   (second minute hour date month year day-of-week dst-p tz)
			 (get-decoded-time)
		       (declare (ignorable dst-p))
		       (format nil "generation date ~2,'0d:~2,'0d:~2,'0d of ~a, ~d-~2,'0d-~2,'0d (GMT~@d)"
			       hour
			       minute
			       second
			       (nth day-of-week *day-names*)
			       year
			       month
			       date
			       (- tz))))
       
       (let ((sample_rate 44100d0)
	     (wavetable_size 1024u)
	     (wavetable ((lambda (size)
			   (let ((wavetable (std--vector<double> size)))
			     (dotimes (i size)
			       (setf (aref wavetable i)
				     (std--sin (/ (* 2 M_PI i)
						  (static_cast<double> size)))))
			     (return wavetable)))
			 wavetable_size))
	     (osc (WavetableOscillator sample_rate wavetable)))
	 (osc.set_frequency 440d0)
	 (dotimes (i 100)
	   ,(lprint :vars `(i (osc.next_sample)))))
       (return 0)))))



