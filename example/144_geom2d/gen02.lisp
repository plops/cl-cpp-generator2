(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "alexandria"))

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
	 (module-name "olcUTIL_Geometry2D_py")
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
       (imports-from  (pybind11.setup_helpers Pybind11Extension))
       (setf ext_modules (list
			  (Pybind11Extension
			   (string ,module-name)
			   :sources (list (string ,(format nil "~a.cpp" module-name)))
			   ;:include_dirs (list (pybind11.get_include))
			   :language (string "c++20"))))
       (setup :name (string ,module-name)
	      :version (string "0.0.1")
	      :author (string "Developers of olcUTIL_Geometry2D")
	      :description (string "Python bindings for olcUTIL_Geometry2D library using Pybind11")
	      :ext_modules ext_modules)))

    (write-source
     (format nil "~a/test_python" *full-source-dir* )
     `(do0
       "#!/usr/bin/env python3"
       
       (imports-from  (,module-name v circle rect line triangle contains closest overlaps intersects envelope_c envelope_r))
       (setf p (v 1 2)
	     c (circle (v 0 0) 5)
	     l (line (v 0 0) (v 1 1))
	     r (rect (v 0 0) (v 1 1))
	     d (triangle (v 0 0) (v 1 1) (v 0 1)))

       ,@(loop for e in `(p c l r d)
	       collect
	       `(print (dot (string ,(format nil "circle around ~a {}" e))
			    (format (envelope_c ,e)))))
       ,@(loop for e in `(p c l r d)
	       collect
	       `(print (dot (string ,(format nil "bbox around ~a {}" e))
			    (format (envelope_r ,e)))))
       ,@(loop for (a b) in 
			 (let ((res))
			   (alexandria:map-permutations #'(lambda (x)
							    (push x res))
							`(p c l r d)
							:length 2)
			   res)
	       collect
	       `(if (contains ,a ,b)
		    (print (string ,(format nil "~a is inside ~a" a b)))
		    (print (string ,(format nil "~a is not inside ~a" a b)))))

       ,@(loop for (a b) in 
		(let ((res))
		  (alexandria:map-permutations #'(lambda (x)
						   (push x res))
					       `(p c l r d)
					       :length 2)
		  res)
		collect
		`(print (dot (string ,(format nil "~a intersects ~a in {}" a b))
			     (format (intersects ,a ,b))))
		 )
       ))))

 
