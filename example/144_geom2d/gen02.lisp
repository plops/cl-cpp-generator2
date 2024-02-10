(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))

(in-package :cl-py-generator)

(progn
  (defparameter *source-dir* #P"example/144_geom2d/source01/python/")
  (defparameter *full-source-dir* (asdf:system-relative-pathname
				   'cl-cpp-generator2
				   *source-dir*))
  (ensure-directories-exist *full-source-dir*)

  (defun lprint (&key msg vars)
    `(do0 ;when args.verbose
       (print (dot (string ,(format nil "{} ~a ~{~a={}~^ ~}"
				    msg
				    (mapcar (lambda (x)
					      (emit-py :code x))
					    vars)))
                   (format  (- (time.time) start_time)
                            ,@vars)))))

  
  
  (let* ((notebook-name "split")
	 (cli-args `((:short "c" :long "chunk_size" :type int :default 500
		      :help "Approximate number of words per chunk")
		     (:short "p" :long "prompt" :type str
		      :default (string "Summarize the following video transcript as a bullet list.")
		      :help "The prompt to be prepended to the output file(s)."))))
    (write-source
     (format nil "~a/setup" *full-source-dir* )
     `(do0
       "#!/usr/bin/env python3"
       (imports-from  (setuptools setup Extension))
       (setf ext_module (Extension
			 (string "pygeometry")
			 :sources (list (string "pygeom.cpp"))
			 :include_dirs (list (pybind11.get_include))
			 :language (string "c++20")))
       (setup :name (string "pygeometry")
	      :ext_modules (list ext_module))))))

