(eval-when (:compile-toplevel :execute :load-toplevel)
	   (ql:quickload "cl-cpp-generator2")
	   (ql:quickload "cl-ppcre")
	   (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *source-dir* #P"example/111_audio/source02/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  (load "util.lisp")
  
  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"main.cpp"
		     *source-dir*))
   `(do0
     
					;(include "fatheader.hpp")
     (include
  
      ,@(loop for e in `(; iostream
			 algorithm
			; cstdlib
			 cstring
			 cmath
			 ;spa/pod/parser.h
					;spa/pod/builder.h
			 spa/param/audio/format-utils.h
			 pipewire/pipewire.h
			 fmt/core.h)
	      collect
	      (format nil "<~a>" e))
      )
     (include "../source00/c_resource.hpp")
				
     (comments "https://docs.pipewire.org/tutorial4_8c-example.html")
     ,@(loop for (e f type) in `((DEFAULT_RATE 44100 int)
			    (DEFAULT_CHANNELS 2 int)
			    (DEFAULT_VOLUME 0.7 double)
				    )
		     collect
		     (format nil "constexpr ~a ~a = ~a;" type e f))
     ,@(loop for e in `(main-loop context
					;core registry properties filter global map-insert
					;protocol resource stream thread-loop work-queue
				  )
	     collect
	     (let* ((str (format nil "~a" e))
		    (pascal (cl-change-case:pascal-case str))
		    (snake (cl-change-case:snake-case str)))
	       `(using ,pascal
		       ,(format nil "stdex::c_resource< pw_~a, pw_~a_new, pw_~a_destroy >"
				snake snake snake))))

     (defun local_pw_registry_destroy (registry)
       (declare (type pw_registry* registry))
       ,(lprint :msg "local_pw_registry_destroy")
       (pw_proxy_destroy (reinterpret_cast<pw_proxy*> registry)))
     ,@(loop for (name type create destroy) in
	     `((Core pw_core pw_context_connect pw_core_disconnect)
	       (Registry pw_registry pw_core_get_registry local_pw_registry_destroy))
	     collect
	     `(using ,name
		     ,(format nil "stdex::c_resource< ~a, ~a, ~a >"
			      type create destroy)))

     (space struct
	    Data
	    (progn
	      "pw_main_loop* loop;"
	      "pw_stream* stream;"
	      "double accumulator"))

     (defun on_process (userdata)
       (declare (type void* userdata))
       ,(lprint :msg "on_process")
       (let ((data (reinterpret_cast<Data*> userdata))
	     (b (pw_stream_dequeue_buffer data->stream)))
	 (when (== nullptr b)
	   ,(lprint :msg "out of buffers")
	   (return))
	 (let ((buf b->buffer)
	       (dst (dot (aref buf->datas 0)
			 data)))
	   (when (== nullptr
		     dst)
	     (return))
	   (let ((stride (* DEFAULT_CHANNELS (sizeof int16_t)))
		 (n_frames (/ (dot (aref buf->datas 0)
				   maxsize)
			      stride)))
	     (dotimes (i n_frames)
	       (incf data->accumulator (/ ,(* 2 pi 440 )
					  DEFAULT_RATE))
	       (when (<= ,(* 2 pi) data->accumulator)
		 (decf data->accumulator ,(* 2 pi)))
	       (setf val (* DEFAULT_VOLUME (sin data->accumulator)
			    16767))
	       (dotimes (c DEFAULT_CHANNELS)
		 (setf *dst++ val)))
	     ,@(loop for (e f) in `((offset 0)
				    (stride stride)
				    (size (* stride n_frames)))
		     collect
		     `(setf
		       (-> (dot (aref buf->datas 0)
				chunk)
			   ,e)
		       ,f))
	     (pw_stream_queue_buffer data->stream
				     b)))
	 ))

     
     (defun main (argc argv)
	       (declare (type int argc)
			(type char** argv)
			(values int))

	       (pw_init &argc &argv)
	       ,(lprint :vars `((pw_get_headers_version)
				(pw_get_library_version)))
	      
					
	       (let ((main_loop (MainLoop nullptr))
		     (context (Context (pw_main_loop_get_loop main_loop)
				       nullptr
				       0))
		     (core ((lambda ()
			      (declare (capture "&context"))
			      (let ((v (Core context nullptr 0)))
				(when (== nullptr v)
				  ,(lprint :msg "error: connection with pipewire daemon failed"))
				(return v)))))
		     (registry (Registry core PW_VERSION_REGISTRY 0))
		     (registry_listener (spa_hook)))
		 
		 (spa_zero registry_listener)
		 (setf "pw_stream_events stream_events" (designated-initializer
						  version PW_VERSION_STREAM_EVENTS
						  process on_process))
		 (pw_registry_add_listener
		  (reinterpret_cast<spa_interface*> (registry.get))
		  
		       &registry_listener
		       &registry_events
		       nullptr)
		 (roundtrip core main_loop)
					
		 )
	       
	       (return 0)))))


  
