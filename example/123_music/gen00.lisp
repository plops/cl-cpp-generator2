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
	 #+log
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
		     (throw (std--invalid_argument (string "Wavetable cannot be empty."))))
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

  (let ((name `EnvelopeGenerator)
	(members `(,@(loop for e in `(sample-rate attack decay sustain release)
			   collect
			   `(,e :type double :param t))
		   (current-state :type EnvelopeGeneratorState :initform EnvelopeGeneratorState--Idle)
		   ,@(loop for e in `(current-amplitude
				      attack-increment
				      decay-increment
				      release-increment)
			   collect
			   `(,e :type double :param nil :initform 0d0)))))
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
	 #+log
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
		   
		   )

		 (defmethod note_on ()
		   (setf current_state_ EnvelopeGeneratorState--Attack
			 attack_increment_ (/ 1d0
					      (* sample_rate_
						 attack_))))

		 (defmethod note_off ()
		   (setf current_state_ EnvelopeGeneratorState--Release
			 release_increment_ (/ (- current_amplitude_
						  0d0)
					      (* sample_rate_
						 release_))))
		 (defmethod next_amplitude ()
		   (declare (values double))
		   (case current_state_
		     (EnvelopeGeneratorState--Attack
		      (incf current_amplitude_ attack_increment_)
		      (when (<= 1d0 current_amplitude_)
			(setf current_amplitude_ 1d0
			      current_state_ EnvelopeGeneratorState--Decay
			      decay_increment_ (/ (- 1d0 sustain_)
						  (* sample_rate_
						     decay_)))))
		     (EnvelopeGeneratorState--Decay
		      (decf current_amplitude_ decay_increment_)
		      (when (<= current_amplitude_ sustain_)
			(setf current_amplitude_ sustain_
			      current_state_ EnvelopeGeneratorState--Sustain)))
		     (EnvelopeGeneratorState--Sustain
		      (comments "amplitude remains constant"))
		     (EnvelopeGeneratorState--Release
		      (decf current_amplitude_ release_increment_)
		      (when (<= current_amplitude_ 0d0)
			(setf current_amplitude_ 0d0
			      current_state_ EnvelopeGeneratorState--Idle)))
		     (EnvelopeGeneratorState--Idle
		      (comments "amplitude remains zero"))
		     (t
		      (comments "amplitude remains zero"))
		     
		     )
		   (return current_amplitude_))
		 

		 "private:"

		 (space enum class EnvelopeGeneratorState
			(curly
			 Idle Attack Decay Sustain Release))
		 
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
     (include "WavetableOscillator.h"
	      "EnvelopeGenerator.h")

     ;#+log
     (do0
      "#define FMT_HEADER_ONLY"
      (include "core.h"))
     (include<> cmath
		iostream)


     (include<> portaudio.h)


     (defun paCallback (input_buffer
			output_buffer
			frames_per_buffer
			time_info
			status_flags
			user_data)
       (declare (type "const void*" input_buffer )
		(type "unsigned long" frames_per_buffer)
		(type "const PaStreamCallbackTimeInfo*" time_info)
		(type PaStreamCallbackFlags status_flags)
		(type "void*" user_data output_buffer)
		(values "static int"))
       (let ((data ("static_cast<std::pair<WavetableOscillator*,EnvelopeGenerator*>*>" user_data))
	     (osc data->first)
	     (env data->second)
	     (out (static_cast<float*> output_buffer)))
	 (dotimes (i frames_per_buffer)
	   (let ((osc_ (osc->next_sample))
		 (env_ (env->next_amplitude))
		 (out_ (* osc_ env_)))
	     (comments "left and right channel")
	     (setf *out++ (static_cast<float> out_)
		   *out++ (static_cast<float> out_))))
	 (return paContinue)))

     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))
       #+log ,(lprint :msg (multiple-value-bind
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
	 (let ((attack 0.01d0)
	       (decay 0.2d0)
	       (sustain 0.6d0)
	       (release 0.5d0)
	       (env (EnvelopeGenerator
		     sample_rate
		     attack
		     decay
		     sustain
		     release))))

	 ,(flet ((pa (code)
		   `(progn
		      (let ((err ,code))
			(unless (== paNoError err)
			  ,(lprint :vars `((Pa_GetErrorText err)))
			  (return 1))))))
	    `(do0
	     ,(pa `(Pa_Initialize))
	     
	     (let ((stream nullptr)
		   (userData (std--make_pair &osc &env)))
	       (declare (type PaStream* stream))

	       ,(pa `(Pa_OpenDefaultStream
		      &stream
		      0 2 paFloat32 sample_rate
		      256
		      paCallback
		      &userData))

	       ,(pa `(Pa_StartStream stream))
	       (do0
	  (env.note_on)
	  (Pa_Sleep 1000)
	  (env.note_off)
	  (Pa_Sleep 2000)
	  ,(pa `(Pa_StopStream stream))
	  ,(pa `(Pa_CloseStream stream))
	  (Pa_Terminate))
	       )))
	 
	 
	 #+nil
	 (do0
	  (let ((count 0))
	    (dotimes (i 2000)
	      (let ((osc_output (osc.next_sample))
		    (env_amplitude (env.next_amplitude))
		    (output_sample (* osc_output env_amplitude)))
		(do0
		 (<< std--cout
		     count
		     (string " ")
		     output_sample
		     std--endl)
		 (incf count)))
	      #+log ,(lprint :vars `(i (osc.next_sample)))))
	  (env.note_off)
	  (dotimes (i 22050)
	    (let ((osc_output (osc.next_sample))
		  (env_amplitude (env.next_amplitude))
		  (output_sample (* osc_output env_amplitude)))
	      (do0
	       (<< std--cout
		   count
		   (string " ")
		   output_sample
		   std--endl)
	       (incf count)))
	    #+log ,(lprint :vars `(i (osc.next_sample))))))
       (return 0)))))



