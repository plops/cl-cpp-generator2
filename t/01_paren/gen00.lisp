(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-cpp-generator2")
  (ql:quickload "cl-ppcre")
  (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
   (progn
    (defparameter *source-dir* #P"t/01_paren/source00/")
    (defparameter *full-source-dir* (asdf:system-relative-pathname
				     'cl-cpp-generator2
				     *source-dir*)))
  (defparameter *day-names*
    '("Monday" "Tuesday" "Wednesday"
      "Thursday" "Friday" "Saturday"
      "Sunday"))
  (ensure-directories-exist *full-source-dir*)
  
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
       (let ((out (static_cast<float*> output_buffer)))
	(let #+nil ((data ("static_cast<std::pair<WavetableOscillator*,EnvelopeGenerator*>*>" user_data))
	      (osc data->first)
	      (env data->second)
	      )
	  (((bracket osc env) (deref ("static_cast<std::pair<WavetableOscillator*,EnvelopeGenerator*>*>" user_data)))
	      )
	  (dotimes (i frames_per_buffer)
	    (let ((osc_ (osc->next_sample))
		  (env_ (env->next_amplitude))
		  (out_ (* osc_ env_)))
	      (comments "left and right channel")
	      (setf *out++ (static_cast<float> out_)
		    *out++ (static_cast<float> out_))))
	  (return paContinue))))

     (defun main (argc argv)
       (declare (values int)
		(type int argc)
		(type char** argv))
       ;#+log
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
		      4096 ; 256
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



