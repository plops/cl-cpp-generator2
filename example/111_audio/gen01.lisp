(eval-when (:compile-toplevel :execute :load-toplevel)
	   (ql:quickload "cl-cpp-generator2")
	   (ql:quickload "cl-ppcre")
	   (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *source-dir* #P"example/111_audio/source01/")
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
			 ;spa/pod/parser.h
			 ;spa/pod/builder.h
			 pipewire/pipewire.h
			 fmt/core.h)
	      collect
	      (format nil "<~a>" e))
      )
     (include "../source00/c_resource.hpp")
				
     (comments "https://docs.pipewire.org/tutorial3_8c-example.html")
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
	    RoundtripData
	    (progn
	      "int pending;"
	      "pw_main_loop* loop;"))

     (defun on_core_done (data id seq)
       (declare (type void* data)
		(type uint32_t id)
		(type int seq))
       ,(lprint :msg "on_core_done")
       (let ((d (reinterpret_cast<RoundtripData*> data)))
	 (when (logand  (== PW_ID_CORE id)
			(== d->pending seq))
	   (pw_main_loop_quit d->loop))))

     (defun roundtrip (core loop)
       (declare (type pw_core* core)
		(type pw_main_loop* loop))
       ,(lprint :msg "roundtrip")
       (setf "static const pw_core_events core_events"
	     (designated-initializer
	      version PW_VERSION_CORE_EVENTS
	      done on_core_done))
       (setf "RoundtripData d" (designated-initializer
				loop loop))
       "spa_hook core_listener;"
       (pw_core_add_listener core
			     &core_listener
			     &core_events
			     &d)
       (setf d.pending
	     (pw_core_sync core
			   PW_ID_CORE
			   0))
       (pw_main_loop_run loop)
       (spa_hook_remove &core_listener))
     
     (defun registry_event_global (data id permissions type version props)
       (declare (type void* data)
		(type uint32_t id permissions version)
		(type "const char*" type)
		(type "const struct spa_dict*" props))
       ,(lprint :vars `(id type version)))
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
		 (setf "pw_registry_events registry_events" (designated-initializer
						  version PW_VERSION_REGISTRY_EVENTS
						  global registry_event_global))
		 (pw_registry_add_listener
		  (reinterpret_cast<spa_interface*> (registry.get))
		  
		       &registry_listener
		       &registry_events
		       nullptr)
		 (roundtrip core main_loop)
					
		 )
	       
	       (return 0)))))


  
