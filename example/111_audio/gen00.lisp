(eval-when (:compile-toplevel :execute :load-toplevel)
	   (ql:quickload "cl-cpp-generator2")
	   (ql:quickload "cl-ppcre")
	   (ql:quickload "cl-change-case"))

(in-package :cl-cpp-generator2)

(progn
  (defparameter *source-dir* #P"example/111_audio/source00/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)
  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"fatheader.hpp"
		     *source-dir*))
   `(do0
     "#pragma once"
     (include
  
      ,@(loop for e in `(iostream
			 algorithm
			 cstdlib
			 cstring
			 spa/pod/parser.h
			 spa/pod/builder.h
			 pipewire/pipewire.h)
	      collect
	      (format nil "<~a>" e))
      )))
  

  (write-source
   (asdf:system-relative-pathname
    'cl-cpp-generator2
    (merge-pathnames #P"main.cpp"
		     *source-dir*))
   `(do0
     
     (include "fatheader.hpp")
					;"import fatheader;"

     ,@(loop for (e f) in `((SAMPLE_RATE 44100)
			    (CHANNELS 2)
			    (BUFFER_SIZE 8192))
	     collect
	     (format nil "constexpr int ~a = ~a;" e f))
     (defun main (argc argv)
       (declare (type int argc)
		(type char** argv)
		(values int))
       (spa_handle_factory_enum &factory
				SPA_TYPE_INTERFACE_Node
				0
				0)
       (return 0)))))


  
